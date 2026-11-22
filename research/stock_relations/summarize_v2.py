"""Add seed-level losses and paired best/last results to a completed v2 record."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reference import load_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    run = root / "experiments/runs" / args.run_id
    record = root / "experiments/paper-records" / args.run_id
    cfg = json.loads((run / "research-config.json").read_text())
    if json.loads((run / "status.json").read_text())["status"] != "complete":
        raise ValueError("Only summarize a completed run")
    load_reference(root, cfg)
    from paper_records import markdown_table, update_manifest

    output = run / "supplement"
    output.mkdir(exist_ok=False)
    predictions = pd.read_parquet(run / "predictions.parquet")
    predictions["rank_target"] = (
        predictions.groupby("trade_date").future_return.rank(pct=True) - 0.5
    )
    losses = []
    for variant in cfg["variants"]:
        for checkpoint, suffix in (("best", ""), ("last", "_last")):
            for seed in cfg["seeds"]:
                column = f"v2_{variant}_s{seed}{suffix}"
                squared = (predictions[column] - predictions.rank_target) ** 2
                daily = squared.groupby(
                    [predictions.split, predictions.trade_date]
                ).mean()
                for split, value in daily.groupby(level="split").mean().items():
                    losses.append(
                        {
                            "variant": variant,
                            "checkpoint": checkpoint,
                            "seed": seed,
                            "split": split,
                            "mse": value,
                        }
                    )
    pd.DataFrame(losses).to_csv(output / "checkpoint-mse.csv", index=False)
    summary = (
        pd.DataFrame(losses)
        .groupby(["variant", "checkpoint", "split"])
        .mse.mean()
        .unstack("split")
        .reset_index()
    )
    metrics = pd.read_csv(run / "metrics.csv")
    ic = pd.read_csv(run / "ic-summary.csv")
    summary["model"] = (
        "v2_" + summary.variant + summary.checkpoint.map({"best": "", "last": "_last"})
    )
    summary = summary.merge(
        ic.loc[ic.split.eq("test"), ["signal", "rank_ic"]],
        left_on="model",
        right_on="signal",
    ).drop(columns="signal")
    summary = summary.merge(
        metrics.loc[
            metrics.cost_scenario.eq("base_10bp"),
            [
                "model",
                "total_return",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown",
                "average_cash_weight",
            ],
        ],
        on="model",
    )
    summary.to_csv(output / "best-last-comparison.csv", index=False)
    trace = pd.read_csv(run / "training-trace.csv")
    selected = (
        pd.read_csv(run / "checkpoint-selection.csv").set_index("model").selected_epoch
    )
    fig, axes = plt.subplots(
        len(cfg["variants"]),
        len(cfg["seeds"]),
        figsize=(12, 16),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for i, variant in enumerate(cfg["variants"]):
        for j, seed in enumerate(cfg["seeds"]):
            name = f"v2_{variant}_s{seed}"
            part = trace.loc[trace.model.eq(name)]
            ax = axes[i, j]
            ax.plot(part.epoch, part.train_mse, label="Train")
            ax.plot(part.epoch, part.valid_mse, label="Validation")
            ax.axvline(
                selected[name], linestyle=":", color="0.4", label="Selected epoch"
            )
            ax.set_title(f"{variant}, seed {seed}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Same-checkpoint MSE")
    axes[0, 0].legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(record / "figures" / f"03-loss-by-seed.{ext}", dpi=160)
    plt.close(fig)
    for path in output.iterdir():
        shutil.copy2(path, record / "results" / path.name)
    shutil.copy2(Path(__file__), record / "code" / Path(__file__).name)
    report = (
        "# 第二轮选中与末轮配对结果\n\nMSE 是三个种子各自原始回归分数的等日期损失再取平均；Rank IC 和组合指标属于三种子排名集成，不能将这两种统计对象混同。best 使用验证集选择，last 固定第 12 轮；2024–2025 仍是已观察过的探索性测试。收益及回撤为小数比例，成本为每边 10bp。\n\n"
        + markdown_table(
            summary[["variant", "checkpoint", "train", "valid", "test", "rank_ic"]]
        )
        + "\n\n"
        + markdown_table(
            summary[
                [
                    "variant",
                    "checkpoint",
                    "total_return",
                    "annualized_return",
                    "sharpe_rf0",
                    "max_drawdown",
                    "average_cash_weight",
                ]
            ]
        )
        + "\n\n![各种子损失](figures/03-loss-by-seed.png)\n\n制图数据为 results/training-trace.csv 与 results/checkpoint-selection.csv；损失复算为 results/checkpoint-mse.csv；生成代码为 code/summarize_v2.py。\n"
    )
    (record / "paired-results.md").write_text(report)
    update_manifest(record)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
