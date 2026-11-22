"""V2 identity-path controls with validation-selected and fixed-last checkpoints."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import StockRelationNet, equal_date_mse
from reference import load_reference, sha256


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        + "\n"
    )


def train_models(data, prep, cfg, run, models, reference):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = reference.tensor_data(data, device)
    train_days = np.flatnonzero((data["split"] == 1) & data["observed"].any(1))
    valid_days = np.flatnonzero((data["split"] == 2) & data["observed"].any(1))
    predictions = data["keys"].copy()
    epochs, updates, selection, checks = [], [], [], {}
    for seed in cfg["seeds"]:
        permutation = reference.permutation_tensor(data, seed, device)
        for variant in cfg["variants"]:
            name = f"v2_{variant}_s{seed}"
            torch.manual_seed(seed)
            model = StockRelationNet(
                prep["input_dim"],
                prep["market_dim"],
                prep["known_stocks"],
                cfg,
                variant,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
            best_loss, best_epoch, step = float("inf"), None, 0
            print(f"Training {name}", flush=True)
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                order = np.random.default_rng(seed + epoch - 1).permutation(train_days)
                for offset in range(0, len(order), cfg["batch_dates"]):
                    indexes = order[offset : offset + cfg["batch_dates"]]
                    optimizer.zero_grad(set_to_none=True)
                    pred, regularizer = model(
                        tensors["x"][indexes],
                        tensors["market"][indexes],
                        tensors["ids"],
                        tensors["active"][indexes],
                        permutation[indexes],
                    )
                    mse = equal_date_mse(
                        pred, tensors["target"][indexes], tensors["observed"][indexes]
                    )
                    objective = mse + regularizer
                    if not torch.isfinite(objective):
                        raise ValueError("Nonfinite training loss")
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["gradient_clip"]
                    )
                    optimizer.step()
                    step += 1
                    updates.append(
                        {
                            "model": name,
                            "seed": seed,
                            "variant": variant,
                            "epoch": epoch,
                            "step": step,
                            "mse": float(mse.detach()),
                            "routing_penalty": float(regularizer.detach()),
                            "objective": float(objective.detach()),
                            "gradient_norm_before_clip": float(norm),
                            "learning_rate": cfg["learning_rate"],
                        }
                    )
                # Both losses use this SAME epoch-end checkpoint, unlike v1's online train MSE.
                loss_by_split = {}
                for label, indexes in (("train", train_days), ("valid", valid_days)):
                    scores = reference.predict_days(
                        model, tensors, permutation, indexes, cfg["batch_dates"]
                    )
                    observed = data["observed"][indexes]
                    loss_by_split[label] = float(
                        (
                            ((scores - data["target"][indexes]) ** 2 * observed).sum(1)
                            / observed.sum(1)
                        ).mean()
                    )
                epochs.append(
                    {
                        "model": name,
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "train_mse": loss_by_split["train"],
                        "valid_mse": loss_by_split["valid"],
                    }
                )
                checkpoint = {
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "config": cfg,
                    "input_dim": prep["input_dim"],
                    "market_dim": prep["market_dim"],
                    "known_stocks": prep["known_stocks"],
                }
                if loss_by_split["valid"] < best_loss:
                    best_loss, best_epoch = loss_by_split["valid"], epoch
                    torch.save(checkpoint, models / (name + "_best.pt"))
                if epoch == cfg["epochs"]:
                    torch.save(checkpoint, models / (name + "_last.pt"))
                print(
                    f"{name} epoch={epoch} train={loss_by_split['train']:.6f} valid={loss_by_split['valid']:.6f}",
                    flush=True,
                )
            selection.append(
                {
                    "model": name,
                    "variant": variant,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selected_valid_mse": best_loss,
                    "last_epoch": cfg["epochs"],
                }
            )
            for checkpoint_name in ("best", "last"):
                path = models / (name + "_" + checkpoint_name + ".pt")
                saved = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(saved["state_dict"])
                scores = reference.predict_days(
                    model,
                    tensors,
                    permutation,
                    np.arange(len(data["dates"])),
                    cfg["batch_dates"],
                )
                column = name if checkpoint_name == "best" else name + "_last"
                predictions[column] = scores[data["row_dates"], data["row_stocks"]]
                check_days = np.linspace(0, len(data["dates"]) - 1, 16, dtype=int)
                second = StockRelationNet(
                    prep["input_dim"],
                    prep["market_dim"],
                    prep["known_stocks"],
                    cfg,
                    variant,
                ).to(device)
                second.load_state_dict(
                    torch.load(path, map_location=device, weights_only=True)[
                        "state_dict"
                    ]
                )
                rebuilt = reference.predict_days(
                    second, tensors, permutation, check_days, cfg["batch_dates"]
                )
                error = float(np.max(np.abs(rebuilt - scores[check_days])))
                if error > 3e-6 or not np.isfinite(predictions[column]).all():
                    raise ValueError("Checkpoint reload mismatch")
                checks[column] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "reload_error": error,
                    "epoch": saved["epoch"],
                }
                del second
            pd.DataFrame(epochs).to_csv(run / "training-trace.csv", index=False)
            pd.DataFrame(updates).to_parquet(
                run / "optimizer-trace.parquet", index=False
            )
            pd.DataFrame(selection).to_csv(
                run / "checkpoint-selection.csv", index=False
            )
            write_json(run / "model-verification.json", checks)
            del model, optimizer
    for variant in cfg["variants"]:
        for suffix in ("", "_last"):
            names = [f"v2_{variant}_s{s}{suffix}" for s in cfg["seeds"]]
            predictions[f"v2_{variant}{suffix}"] = (
                predictions.groupby("trade_date")[names].rank(pct=True).mean(axis=1)
            )
    root = reference.ROOT
    old = root / "models" / cfg["feature_parent"] / "lgb_cluster.joblib"
    verification = json.loads(
        (
            root
            / "experiments/runs"
            / cfg["feature_parent"]
            / "model-verification.json"
        ).read_text()
    )
    if sha256(old) != verification["lgb_cluster"]["sha256"]:
        raise ValueError("Reference tree changed")
    artifact = joblib.load(old)
    predictions["lgb_cluster"] = artifact["model"].predict(
        data["raw_rank"][artifact["features"]].fillna(0)
    )
    labeled = data["samples"].merge(
        predictions.drop(columns="is_st"),
        on=["trade_date", "symbol"],
        validate="one_to_one",
    )
    labeled.to_parquet(run / "predictions.parquet", index=False)
    from model_portfolio_backtest import is_main_board

    scores = predictions.loc[
        predictions.symbol.map(is_main_board)
        & predictions.trade_date.between("2023-12-13", "2025-12-31")
    ].copy()
    scores["eligible"] = scores.is_st.eq(0).fillna(False)
    scores.to_parquet(run / "scores.parquet", index=False)
    del tensors
    torch.cuda.empty_cache()
    return labeled, scores


def evaluate(predictions, cfg, run, reference):
    from factor_research import daily_metrics, summarize, block_interval

    # Preserve seed-level predictions; evaluate all saved heads/checkpoints without selection on test.
    columns = [c for c in predictions if c.startswith("v2_") or c == "lgb_cluster"]
    daily = pd.concat(
        [daily_metrics(predictions, c) for c in columns], ignore_index=True
    )
    daily.to_parquet(run / "daily-ic.parquet", index=False)
    summarize(daily).to_csv(run / "ic-summary.csv", index=False)
    pairs = []
    for split in ("valid", "test"):
        pivot = daily.loc[daily.split.eq(split)].pivot(
            index="trade_date", columns="signal", values="rank_ic"
        )
        for name, base in cfg["comparisons"]:
            for suffix in ("", "_last"):
                delta = (pivot[name + suffix] - pivot[base + suffix]).dropna()
                low, high = block_interval(delta, cfg)
                pairs.append(
                    {
                        "split": split,
                        "model": name + suffix,
                        "baseline": base + suffix,
                        "delta": delta.mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    pd.DataFrame(pairs).to_csv(run / "paired-ic.csv", index=False)
    # Fixed-last checkpoints must reproduce v1 for the unchanged branches.
    previous = pd.read_parquet(
        reference.ROOT
        / "experiments/runs"
        / cfg["reference_run"]
        / "predictions.parquet"
    )
    shared = predictions.merge(
        previous,
        on=["trade_date", "symbol", "split"],
        suffixes=("_v2", "_v1"),
        validate="one_to_one",
    )
    if len(shared) != len(predictions) or len(shared) != len(previous):
        raise ValueError("Changed labeled sample")
    checks = []
    for variant in ("local", "market", "relation"):
        for seed in cfg["seeds"]:
            error = float(
                (shared[f"v2_{variant}_s{seed}_last"] - shared[f"nn_{variant}_s{seed}"])
                .abs()
                .max()
            )
            if error > 1e-5:
                raise ValueError(
                    f"Unchanged last checkpoint did not reproduce v1: {variant}/{seed}/{error}"
                )
            checks.append(
                {"variant": variant, "seed": seed, "max_prediction_error": error}
            )
    pd.DataFrame(checks).to_csv(run / "v1-reproduction.csv", index=False)


def record_run(run, cfg, workspace, status):
    from paper_records import markdown_table, update_manifest, update_index

    record = workspace / "experiments/paper-records" / run.name
    record.mkdir(exist_ok=False)
    for folder in ("results", "evidence", "code", "figures", "verification"):
        (record / folder).mkdir()
    result_names = [
        "training-trace.csv",
        "optimizer-trace.parquet",
        "checkpoint-selection.csv",
        "model-verification.json",
        "predictions.parquet",
        "scores.parquet",
        "ic-summary.csv",
        "daily-ic.parquet",
        "paired-ic.csv",
        "v1-reproduction.csv",
        "metrics.csv",
        "daily-results.parquet",
        "orders.parquet",
        "holdings.parquet",
        "target-selections.parquet",
        "yearly-metrics.csv",
    ]
    for name in result_names:
        if (run / name).exists():
            shutil.copy2(run / name, record / "results" / name)
    for name in (
        "config.json",
        "research-config.json",
        "provenance.json",
        "preprocessing.json",
        "sample-audit.json",
        "run.log",
        "status.json",
        "failure.txt",
    ):
        if (run / name).exists():
            shutil.copy2(run / name, record / "evidence" / name)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, record / "code" / source.name)
    for source in Path(__file__).parent.glob("*.md"):
        shutil.copy2(source, record / "code" / source.name)
    if (run / "independent-verification").exists():
        shutil.copytree(
            run / "independent-verification", record / "verification" / "portfolio"
        )
    if (run / "independent-model-verification").exists():
        shutil.copytree(
            run / "independent-model-verification", record / "verification" / "model"
        )
    if (run / "provenance.json").exists():
        shutil.copy2(run / "provenance.json", record / "references.json")
    text = (Path(__file__).parent / "PROTOCOL.md").read_text()
    text = (
        f"# 股票关系第二轮：身份通道与状态路由\n\n运行：{run.name}；状态：{status}。\n\n"
        + text
    )
    if status == "complete":
        summary = pd.read_csv(run / "ic-summary.csv")
        summary = summary.loc[
            summary.split.eq("test")
            & ~summary.signal.str.contains(r"_s\d", regex=True),
            ["signal", "rank_ic"],
        ]
        metrics = pd.read_csv(run / "metrics.csv")
        base = metrics.loc[
            metrics.cost_scenario.eq("base_10bp"),
            [
                "model",
                "total_return",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown",
                "average_cash_weight",
            ],
        ]
        text += (
            "\n\n## 探索性测试结果\n\n"
            + markdown_table(summary)
            + "\n\n"
            + markdown_table(base)
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trace = pd.read_csv(run / "training-trace.csv")
        fig, axes = plt.subplots(
            2, 3, figsize=(13, 7), sharex=True, sharey=True, layout="constrained"
        )
        for variant, ax in zip(cfg["variants"], axes.flat, strict=False):
            g = (
                trace.loc[trace.variant.eq(variant)]
                .groupby("epoch")[["train_mse", "valid_mse"]]
                .mean()
            )
            ax.plot(g.index, g.train_mse, label="Training")
            ax.plot(g.index, g.valid_mse, label="Validation")
            ax.set_title(variant)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Same-checkpoint MSE")
        axes.flat[0].legend()
        for ext in ("png", "svg"):
            fig.savefig(record / "figures" / ("01-loss." + ext), dpi=160)
        plt.close(fig)
        daily = pd.read_parquet(run / "daily-results.parquet")
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for name, g in daily.loc[
            daily.cost_scenario.isin(["base_10bp", "benchmark"])
        ].groupby("model"):
            ax.plot(g.trade_date, g.nav, label=name)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylabel("NAV, 10bp per-side friction")
        for ext in ("png", "svg"):
            fig.savefig(record / "figures" / ("02-nav." + ext), dpi=160)
        plt.close(fig)
        text += "\n\n![损失](figures/01-loss.png)\n\n![净值](figures/02-nav.png)\n"
    (record / "paper.md").write_text(text)
    write_json(
        record / "record-status.json",
        {
            "status": "complete",
            "experiment_status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    update_manifest(record)
    update_index(record.parent)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config-v2.json")
    )
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unsafe run id")
    workspace = args.workspace.resolve()
    cfg = json.loads(args.config.read_text())
    reference = load_reference(workspace, cfg)
    run = workspace / "experiments/runs" / args.run_id
    models = workspace / "models" / args.run_id
    if models.exists():
        raise FileExistsError(models)
    run.mkdir(exist_ok=False)
    models.mkdir(exist_ok=False)
    from factor_research import Tee

    logfile = (run / "run.log").open("x", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)
    write_json(run / "research-config.json", cfg)
    write_json(
        run / "status.json",
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )
    execution = json.loads((workspace / "configs/model-backtest-v1.json").read_text())
    execution["models"] = [
        "v2_" + v + suffix for suffix in ("", "_last") for v in cfg["variants"]
    ] + ["lgb_cluster"]
    write_json(run / "config.json", execution)
    try:
        data, prep, provenance = reference.prepare(cfg, run)
        provenance.update(
            model_directory=str(models),
            code_sha256=sha256(__file__),
            model_code_sha256=sha256(Path(__file__).with_name("model.py")),
            config_sha256=sha256(args.config),
        )
        write_json(run / "provenance.json", provenance)
        write_json(run / "preprocessing.json", prep)
        write_json(models / "preprocessing.json", prep)
        predictions, scores = train_models(data, prep, cfg, run, models, reference)
        del data
        evaluate(predictions, cfg, run, reference)
        del predictions
        checked = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_training.py")),
                "--workspace",
                str(workspace),
                "--run-id",
                args.run_id,
            ],
            capture_output=True,
            text=True,
        )
        (run / "independent-model-verification").mkdir(exist_ok=True)
        (run / "independent-model-verification" / "verification.log").write_text(
            checked.stdout + checked.stderr
        )
        if checked.returncode:
            raise RuntimeError("Independent model verification failed")
        reference.run_portfolios(scores, execution, provenance, run)
        verifier = (
            workspace
            / "experiments/paper-records"
            / cfg["reference_run"]
            / "code/verify_model_portfolio_backtest.py"
        )
        result = subprocess.run(
            [sys.executable, str(verifier), str(run)], capture_output=True, text=True
        )
        (run / "independent-verification").mkdir(exist_ok=True)
        (run / "independent-verification" / "verification.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode:
            raise RuntimeError("Independent portfolio verification failed")
        write_json(
            run / "status.json",
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except BaseException:
        (run / "failure.txt").write_text(traceback.format_exc())
        write_json(
            run / "status.json",
            {"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat()},
        )
        record_run(run, cfg, workspace, "failed")
        raise
    print(record_run(run, cfg, workspace, "complete") / "paper.md", flush=True)


if __name__ == "__main__":
    main()
