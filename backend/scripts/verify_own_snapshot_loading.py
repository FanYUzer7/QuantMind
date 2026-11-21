"""Read-only native loading/label verification; does not train or predict a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.loading import load_data
from backend.services.engine.inference.data_loader import load_date_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("/quant"))
    ap.add_argument("--dataset", default="datasets/own-daily-research-v002")
    ap.add_argument(
        "--output",
        default="experiments/data-integration/own-data-continuous-20261101-01/native-loading-02.json",
    )
    args = ap.parse_args()
    folder, output = args.workspace / args.dataset, args.workspace / args.output
    if output.exists():
        raise SystemExit("Use a new verification output path")
    meta = json.loads((folder / "manifest.json").read_text())
    if not meta["research_only"] or meta["production_enabled"]:
        raise ValueError(
            "This verifier expects an explicitly isolated research snapshot"
        )
    required = meta["features"]
    path = folder / "model_features_2025.parquet"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "dataset": args.dataset,
        "model_training_executed": False,
        "model_prediction_executed": False,
        "backtest_executed": False,
        "status": "validating",
        "research_only": True,
        "dataset_sha256": before,
    }
    try:
        if before != meta["files"][path.name]:
            raise ValueError("Snapshot hash does not match manifest")
        frame = (
            pd.read_parquet(path)
            .sort_values(["symbol", "trade_date"])
            .reset_index(drop=True)
        )
        native, used = load_data(
            "2025-07-01",
            "2025-08-15",
            required,
            target_horizon_days=5,
            target_mode="lab_return",
            valid_end="2025-09-05",
            test_end="2025-09-19",
            local_dir=str(folder),
            market="CN",
        )
        if used != required:
            raise AssertionError(
                "Native loader silently ignored/reordered requested features"
            )
        future = frame.groupby("symbol")["close"].shift(-6)
        execution = frame.groupby("symbol")["close"].shift(-1)
        frame["independent_label"] = (future / execution - 1).clip(-0.3, 0.3)
        joined = native.merge(
            frame[["symbol", "trade_date", "independent_label"]],
            on=["symbol", "trade_date"],
            validate="one_to_one",
        )
        if len(joined) != len(native) or joined["independent_label"].isna().any():
            raise AssertionError(
                "Native label dates do not align with independent price calculation"
            )
        errors = abs(joined["label"] - joined["independent_label"])
        if errors.max() > 1e-7:
            raise AssertionError("T+1 execution / H=5 native label mismatch")
        expected = frame.loc[
            frame["trade_date"].between("2025-07-01", "2025-09-19")
            & frame["independent_label"].notna()
        ]
        if len(native) != len(expected):
            raise AssertionError(
                "Unexpected native filtering of the verified sample window"
            )
        date = "2025-09-19"
        inferred = load_date_data(
            date,
            folder,
            {
                "context": {"market": "CN"},
                "data_source": "parquet",
                "feature_columns": required,
                "features": required,
            },
        )
        source = frame.loc[frame["trade_date"] == date].sort_values("symbol")
        if inferred is None:
            raise AssertionError("Native inference loader returned no data")
        inferred = inferred.sort_values("symbol")
        if source["symbol"].tolist() != inferred["symbol"].tolist() or not np.allclose(
            source[required], inferred[required], rtol=1e-7, atol=1e-9
        ):
            raise AssertionError(
                "Native inference feature values or instruments changed"
            )
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if after != before:
            raise AssertionError("Read-only verification mutated snapshot")
        report.update(
            status="passed",
            native_training_rows=len(native),
            native_features=used,
            native_training_first=str(native["trade_date"].min().date()),
            native_training_last=str(native["trade_date"].max().date()),
            label_formula="clip(qfq_close[T+6]/qfq_close[T+1]-1,-0.3,0.3)",
            label_max_absolute_error=float(errors.max()),
            inference_date=date,
            inference_rows=len(inferred),
            inference_features_match=True,
            source_hash_unchanged=True,
        )
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        raise
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
