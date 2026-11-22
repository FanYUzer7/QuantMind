"""Check checkpoint selection and reconstruct scored rows in a fresh process."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from model import StockRelationNet
from reference import load_reference, sha256


def verify(workspace, run_id):
    run = workspace / "experiments/runs" / run_id
    output = run / "independent-model-verification"
    output.mkdir(exist_ok=False)
    cfg = json.loads((run / "research-config.json").read_text())
    ref = load_reference(workspace, cfg)
    data, prep, _ = ref.prepare(cfg, output)
    if prep != json.loads((run / "preprocessing.json").read_text()):
        raise ValueError("Rebuilt preprocessing changed")
    trace = pd.read_csv(run / "training-trace.csv")
    selection = pd.read_csv(run / "checkpoint-selection.csv").set_index("model")
    metadata = json.loads((run / "model-verification.json").read_text())
    models = Path(json.loads((run / "provenance.json").read_text())["model_directory"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors = ref.tensor_data(data, device)
    scored = pd.read_parquet(run / "scores.parquet")
    available = data["dates"].get_indexer(scored.trade_date.unique())
    indexes = np.unique(available[np.linspace(0, len(available) - 1, 16, dtype=int)])
    selected_dates = data["dates"][indexes]
    rows = np.isin(data["row_dates"], indexes)
    reconstructed = data["keys"].loc[rows, ["trade_date", "symbol"]].copy()
    row_day_positions = np.searchsorted(indexes, data["row_dates"][rows])
    selected_epochs = {}
    for seed in cfg["seeds"]:
        permutation = ref.permutation_tensor(data, seed, device)
        for variant in cfg["variants"]:
            name = f"v2_{variant}_s{seed}"
            history = trace.loc[trace.model.eq(name)].sort_values("epoch")
            if history.epoch.tolist() != list(range(1, cfg["epochs"] + 1)):
                raise ValueError("Incomplete epoch trace")
            if not np.isfinite(history[["train_mse", "valid_mse"]]).all().all():
                raise ValueError("Nonfinite epoch metric")
            expected_epoch = int(history.loc[history.valid_mse.idxmin(), "epoch"])
            if int(selection.loc[name, "selected_epoch"]) != expected_epoch:
                raise ValueError("Validation selection mismatch")
            selected_epochs[name] = expected_epoch
            for checkpoint, suffix, epoch in (
                ("best", "", expected_epoch),
                ("last", "_last", cfg["epochs"]),
            ):
                path = models / f"{name}_{checkpoint}.pt"
                column = name + suffix
                if sha256(path) != metadata[column]["sha256"]:
                    raise ValueError("Weight checksum mismatch")
                saved = torch.load(path, map_location=device, weights_only=True)
                if saved["epoch"] != epoch or saved["config"] != cfg:
                    raise ValueError("Wrong saved epoch/config")
                model = StockRelationNet(
                    prep["input_dim"],
                    prep["market_dim"],
                    prep["known_stocks"],
                    cfg,
                    variant,
                ).to(device)
                model.load_state_dict(saved["state_dict"])
                # Replay the original contiguous batches before sampling probe dates.
                # Regrouping arbitrary dates can perturb near-tied float32 logits.
                prediction = ref.predict_days(
                    model,
                    tensors,
                    permutation,
                    np.arange(len(data["dates"])),
                    cfg["batch_dates"],
                )[indexes]
                reconstructed[column] = prediction[
                    row_day_positions, data["row_stocks"][rows]
                ]
                del model
    for variant in cfg["variants"]:
        for suffix in ("", "_last"):
            columns = [f"v2_{variant}_s{s}{suffix}" for s in cfg["seeds"]]
            reconstructed[f"v2_{variant}{suffix}"] = (
                reconstructed.groupby("trade_date")[columns].rank(pct=True).mean(axis=1)
            )
    expected = scored.loc[scored.trade_date.isin(selected_dates)]
    combined = expected.merge(
        reconstructed,
        on=["trade_date", "symbol"],
        suffixes=("_saved", "_rebuilt"),
        validate="one_to_one",
    )
    if len(combined) != len(expected):
        raise ValueError("Missing verification rows")
    errors = {
        c: float((combined[c + "_saved"] - combined[c + "_rebuilt"]).abs().max())
        for c in reconstructed
        if c.startswith("v2_")
    }
    if not np.isfinite(list(errors.values())).all() or max(errors.values()) > 3e-6:
        raise ValueError(f"Reconstructed prediction mismatch: {errors}")
    result = {
        "status": "passed",
        "probe_dates": len(indexes),
        "scored_rows": len(combined),
        "signals": len(errors),
        "max_prediction_errors": errors,
        "selected_epochs": selected_epochs,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace, args.run_id), indent=2))
