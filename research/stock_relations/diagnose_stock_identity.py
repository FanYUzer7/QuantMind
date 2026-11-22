"""Frozen diagnostics for whether stock identity memorizes training rank.

This script does not train. It reads the stock-relation checkpoints and the
same purged, equal-date sample used by stock-relation-20261120-001. Outputs go
to a new run directory. Original runs, weights, and the parent paper are left
unchanged.

Three different claims are kept apart:

- Observation: train-period target means, later target means, and checkpoint loss.
- Diagnostic intervention: at a frozen checkpoint, zero only the direct
  identity input to the head, or replace only routing weights with the uniform
  distribution. These are off-distribution edits, not retrained causal effects.
- Not claimed here: that a coordinate of e_i is a rank, or that the intervention
  equals a model trained without that path.
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd
import torch

from model import StockRelationNet, peer_context
from reference import load_reference

ROOT = Path(__file__).resolve().parents[3]
PARENT_RUN_ID = "stock-relation-20261120-001"
DEFAULT_RUN_ID = "stock-identity-audit-20261122-001"
MIN_DAYS = 50
IDENTITY_VARIANTS = ("identity", "relation", "shuffled")
ROUTING_VARIANTS = ("relation", "shuffled")
INTERVENTIONS = ("original", "zero_direct_identity", "uniform_routing", "both")


def intervention_flags(name):
    if name == "original":
        return False, False
    if name == "zero_direct_identity":
        return True, False
    if name == "uniform_routing":
        return False, True
    if name == "both":
        return True, True
    raise ValueError(name)


def forward_parts(
    model,
    x,
    market,
    ids,
    active,
    permutation,
    *,
    zero_direct_identity=False,
    uniform_routing=False,
):
    """Replay StockRelationNet.forward with optional isolated path edits.

    zero_direct_identity replaces only the embedding concatenated into the head.
    uniform_routing replaces only query and source assignment weights, after
    those weights have been read from the frozen embedding. Leave-self-out is
    still applied inside peer_context. Local and market variants do not gain a
    path they did not use in training.
    """
    h = model.encoder(x)
    identity = model.identity(ids).unsqueeze(0).expand(x.shape[0], -1, -1)
    market_to_head = market.unsqueeze(1).expand(-1, x.shape[1], -1)
    context = torch.zeros_like(h)
    query = None
    if model.variant in ROUTING_VARIANTS:
        query = model.assignments(ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        if uniform_routing:
            query = torch.full_like(query, 1.0 / query.shape[-1])
        if model.variant == "relation":
            source = query
        else:
            index = permutation.unsqueeze(-1).expand(-1, -1, query.shape[-1])
            source = torch.gather(query, 1, index)
        context = peer_context(h, query, source, active)
    if model.variant == "local":
        market_to_head = torch.zeros_like(market_to_head)
    if model.variant not in IDENTITY_VARIANTS or zero_direct_identity:
        identity_to_head = torch.zeros_like(identity)
    else:
        identity_to_head = identity
    prediction = model.head(
        torch.cat([h, market_to_head, identity_to_head, context], dim=-1)
    ).squeeze(-1)
    return {
        "prediction": prediction,
        "encoded": h,
        "market_to_head": market_to_head,
        "identity_to_head": identity_to_head,
        "context": context,
        "query_assignment": query,
    }


def weighted_stock_means(target, observed):
    """Stock-specific intercept that minimizes equal-date squared error.

    The loss averages squared error inside each labeled day, then averages days
    equally. The optimum is therefore not the unweighted mean of a stock's
    targets. Each labeled row of stock i on day t has weight 1/n_t, where n_t
    is the number of labeled names that day. A stock with no labeled row is NaN.
    """
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if target.shape != observed.shape:
        raise ValueError("target and observed shapes differ")
    day_count = observed.sum(axis=1).astype(np.float64)
    weights = np.zeros(target.shape, dtype=np.float64)
    labeled_days = day_count > 0
    weights[labeled_days] = observed[labeled_days] / day_count[labeled_days, None]
    weight_sum = weights.sum(axis=0)
    weighted_target = (weights * target).sum(axis=0)
    means = np.full(target.shape[1], np.nan, dtype=np.float64)
    present = weight_sum > 0
    means[present] = weighted_target[present] / weight_sum[present]
    labeled = observed.sum(axis=0).astype(np.int64)
    return means, weight_sum, labeled, weights


def apply_known_means(means, ids):
    """Map a per-column mean to a forecast. Unknown ids and missing means are 0."""
    means = np.asarray(means, dtype=np.float64)
    ids = np.asarray(ids)
    forecast = np.zeros(means.shape, dtype=np.float64)
    known = (ids > 0) & np.isfinite(means)
    forecast[known] = means[known]
    return forecast


def leave_one_day_forecasts(target, observed, ids):
    """Train forecast with each stock-day removed from its own weighted mean."""
    _, weight_sum, _, weights = weighted_stock_means(target, observed)
    weighted_target_sum = (weights * target).sum(axis=0)
    numerator = weighted_target_sum[None, :] - weights * target
    denominator = weight_sum[None, :] - weights
    forecast = np.zeros(target.shape, dtype=np.float64)
    usable = denominator > 0
    forecast[usable] = numerator[usable] / denominator[usable]
    forecast[:, np.asarray(ids) <= 0] = 0.0
    return forecast


def equal_date_mse_rows(prediction, target, observed, day_index):
    """Equal-date MSE on the supplied day positions. Prediction may be [N] or [T, N]."""
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    day_index = np.asarray(day_index, dtype=np.int64)
    y = target[day_index]
    mask = observed[day_index]
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim == 1:
        pred = np.broadcast_to(pred, y.shape)
    else:
        pred = pred[day_index]
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite prediction in equal-date MSE")
    count = mask.sum(axis=1)
    if np.any(count <= 0):
        raise ValueError("equal-date MSE received a day without labels")
    daily = ((pred - y) ** 2 * mask).sum(axis=1) / count
    return float(daily.mean()), daily


def float32_trace_mse(prediction, target, observed, day_index):
    """Match the saved validation trace arithmetic as closely as numpy allows."""
    pred = np.asarray(prediction, dtype=np.float32)[day_index]
    y = np.asarray(target, dtype=np.float32)[day_index]
    mask = np.asarray(observed, dtype=bool)[day_index]
    daily = ((pred - y) ** 2 * mask).sum(axis=1) / mask.sum(axis=1)
    return float(daily.mean())


def mean_daily_rank_ic(prediction, future, observed, day_index):
    """Mean daily Spearman correlation of a forecast with future returns.

    Days with fewer than two distinct forecasts or outcomes are omitted, matching
    the study's daily_metrics filter. A constant forecast therefore has no Rank IC.
    """
    from scipy.stats import rankdata

    future = np.asarray(future, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    day_index = np.asarray(day_index, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.float64)
    values = []
    for day in day_index:
        mask = observed[day]
        outcome = future[day, mask]
        if pred.ndim == 1:
            score = np.broadcast_to(pred, future.shape[1])[mask]
        else:
            score = pred[day, mask]
        if score.size < 2 or np.unique(score).size < 2 or np.unique(outcome).size < 2:
            continue
        left = rankdata(score, method="average")
        right = rankdata(outcome, method="average")
        left = left - left.mean()
        right = right - right.mean()
        denom = np.sqrt((left * left).sum() * (right * right).sum())
        if denom == 0:
            continue
        values.append(float((left * right).sum() / denom))
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def stock_mean(values, observed, day_mask):
    observed = np.asarray(observed, dtype=bool)
    day_mask = np.asarray(day_mask, dtype=bool)
    mask = observed & day_mask[:, None]
    count = mask.sum(axis=0).astype(np.int64)
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim == 1:
        total = grid * count
    else:
        total = np.where(mask, grid, 0.0).sum(axis=0)
    mean = np.full(count.shape, np.nan, dtype=np.float64)
    present = count > 0
    mean[present] = total[present] / count[present]
    return mean, count


def correlation_pair(left, left_days, right, right_days, min_days):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = (np.asarray(left_days) >= min_days) & (np.asarray(right_days) >= min_days)
    keep &= np.isfinite(left) & np.isfinite(right)
    n = int(keep.sum())
    pair = {
        "n_stocks": n,
        "min_days": int(min_days),
        "pearson": None,
        "spearman": None,
        "left_std": None,
        "right_std": None,
    }
    if n < 3:
        return pair
    a = pd.Series(left[keep])
    b = pd.Series(right[keep])
    pair["left_std"] = float(a.std(ddof=1))
    pair["right_std"] = float(b.std(ddof=1))
    if a.nunique(dropna=True) > 1 and b.nunique(dropna=True) > 1:
        pair["pearson"] = float(a.corr(b, method="pearson"))
        pair["spearman"] = float(a.corr(b, method="spearman"))
    return pair


def finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return None
    return number


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def predict_intervention(model, tensors, permutation, indexes, intervention, batch):
    zero_direct, uniform = intervention_flags(intervention)
    pieces = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(indexes), batch):
            chosen = indexes[offset : offset + batch]
            parts = forward_parts(
                model,
                tensors["x"][chosen],
                tensors["market"][chosen],
                tensors["ids"],
                tensors["active"][chosen],
                permutation[chosen],
                zero_direct_identity=zero_direct,
                uniform_routing=uniform,
            )
            pieces.append(parts["prediction"].detach().cpu().numpy())
    return np.concatenate(pieces, axis=0)


def check_forward_matches_module(model, tensors, permutation, indexes):
    model.eval()
    with torch.no_grad():
        module_pred, _ = model(
            tensors["x"][indexes],
            tensors["market"][indexes],
            tensors["ids"],
            tensors["active"][indexes],
            permutation[indexes],
        )
        parts = forward_parts(
            model,
            tensors["x"][indexes],
            tensors["market"][indexes],
            tensors["ids"],
            tensors["active"][indexes],
            permutation[indexes],
        )
    return float(
        np.max(
            np.abs(
                module_pred.detach().cpu().numpy()
                - parts["prediction"].detach().cpu().numpy()
            )
        )
    )


def check_path_isolation(model, tensors, permutation, indexes):
    """Fail closed if an intervention edits a tensor it is not supposed to edit."""
    model.eval()
    with torch.no_grad():
        chosen = indexes
        original = forward_parts(
            model,
            tensors["x"][chosen],
            tensors["market"][chosen],
            tensors["ids"],
            tensors["active"][chosen],
            permutation[chosen],
        )
        zeroed = forward_parts(
            model,
            tensors["x"][chosen],
            tensors["market"][chosen],
            tensors["ids"],
            tensors["active"][chosen],
            permutation[chosen],
            zero_direct_identity=True,
        )
        uniform = forward_parts(
            model,
            tensors["x"][chosen],
            tensors["market"][chosen],
            tensors["ids"],
            tensors["active"][chosen],
            permutation[chosen],
            uniform_routing=True,
        )
        both = forward_parts(
            model,
            tensors["x"][chosen],
            tensors["market"][chosen],
            tensors["ids"],
            tensors["active"][chosen],
            permutation[chosen],
            zero_direct_identity=True,
            uniform_routing=True,
        )

    def same(left, right):
        return torch.allclose(left, right, rtol=1e-6, atol=1e-6)

    if not same(zeroed["context"], original["context"]):
        raise RuntimeError("Zeroing direct identity changed peer context")
    if not same(zeroed["encoded"], original["encoded"]) or not same(
        zeroed["market_to_head"], original["market_to_head"]
    ):
        raise RuntimeError("Zeroing direct identity changed encoder or market inputs")
    if model.variant in ROUTING_VARIANTS and not same(
        zeroed["query_assignment"], original["query_assignment"]
    ):
        raise RuntimeError("Zeroing direct identity changed routing assignments")
    if not same(
        zeroed["identity_to_head"], torch.zeros_like(zeroed["identity_to_head"])
    ):
        raise RuntimeError("Direct identity input was not zero")
    if not same(uniform["identity_to_head"], original["identity_to_head"]):
        raise RuntimeError("Uniform routing changed the direct identity input")
    if not same(uniform["encoded"], original["encoded"]) or not same(
        uniform["market_to_head"], original["market_to_head"]
    ):
        raise RuntimeError("Uniform routing changed encoder or market inputs")
    if model.variant in ROUTING_VARIANTS:
        if uniform["query_assignment"] is None or not torch.allclose(
            uniform["query_assignment"],
            torch.full_like(
                uniform["query_assignment"], 1.0 / uniform["query_assignment"].shape[-1]
            ),
        ):
            raise RuntimeError("Uniform routing did not replace assignment weights")
        if not same(both["context"], uniform["context"]) or not same(
            both["query_assignment"], uniform["query_assignment"]
        ):
            raise RuntimeError(
                "Combined intervention did not keep the uniform routing path"
            )
    else:
        if not same(uniform["prediction"], original["prediction"]):
            raise RuntimeError(
                "Uniform routing changed a model that has no routing path"
            )
    if not same(both["identity_to_head"], torch.zeros_like(both["identity_to_head"])):
        raise RuntimeError("Combined intervention did not zero direct identity")
    unknown = tensors["ids"].eq(0).detach().cpu().numpy()
    if unknown.any():
        gap = np.max(
            np.abs(
                original["prediction"].detach().cpu().numpy()[:, unknown]
                - zeroed["prediction"].detach().cpu().numpy()[:, unknown]
            )
        )
        if gap > 1e-6:
            raise RuntimeError(
                "Unknown-id predictions changed when the direct identity input was zeroed"
            )


def assert_prep_match(prep, expected):
    if prep == expected:
        return {"preprocessing_exact_match": True}
    mismatches = []
    for key in sorted(set(prep) | set(expected)):
        if prep.get(key) != expected.get(key):
            mismatches.append(key)
    detail = []
    market_new, market_old = prep.get("market") or {}, expected.get("market") or {}
    for field in ("mean", "std"):
        left, right = market_new.get(field) or {}, market_old.get(field) or {}
        keys = sorted(set(left) | set(right))
        if keys and all(key in left and key in right for key in keys):
            gap = max(abs(float(left[key]) - float(right[key])) for key in keys)
            detail.append(f"market.{field} max abs {gap:.3g}")
    raise ValueError(
        "Reconstructed preprocessing does not match the parent run: "
        + ", ".join(mismatches[:12])
        + ("; " + "; ".join(detail) if detail else "")
    )


def method_test_report():
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(Path(__file__).parent),
            "-p",
            "test_identity_audit.py",
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    return tests.returncode, tests.stdout + tests.stderr


def load_parent_checkpoints(parent, prep, device):
    from factor_research import digest

    checks = json.loads((parent / "model-verification.json").read_text())
    models = Path(
        json.loads((parent / "provenance.json").read_text())["model_directory"]
    )
    loaded = []
    for name, expected in checks.items():
        path = models / f"{name}.pt"
        actual = digest(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Weight hash changed for {name}")
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["variant"] not in name or int(saved["seed"]) != int(
            name.rsplit("s", 1)[1]
        ):
            raise ValueError(f"Checkpoint metadata does not match {name}")
        if int(saved["known_stocks"]) != int(prep["known_stocks"]):
            raise ValueError(
                f"Checkpoint vocabulary does not match preprocessing for {name}"
            )
        if int(saved["input_dim"]) != int(prep["input_dim"]) or int(
            saved["market_dim"]
        ) != int(prep["market_dim"]):
            raise ValueError(
                f"Checkpoint input shape does not match preprocessing for {name}"
            )
        if float(saved["state_dict"]["identity.weight"][0].abs().max()) != 0.0:
            raise ValueError(f"Unknown embedding is not zero in {name}")
        loaded.append(
            {
                "name": name,
                "path": path,
                "sha256": actual,
                "saved": saved,
                "parameters": expected["parameters"],
            }
        )
    return loaded, models


def scatter_frame(table, min_days):
    shared = table.loc[
        table.train_labeled_days.ge(min_days)
        & table.valid_labeled_days.ge(min_days)
        & table.test_labeled_days.ge(min_days),
        [
            "symbol",
            "known_identity",
            "train_labeled_days",
            "valid_labeled_days",
            "test_labeled_days",
            "train_weighted_mean",
            "valid_weighted_mean",
            "test_weighted_mean",
        ],
    ].copy()
    pairwise = {
        "train_valid": table.loc[
            table.train_labeled_days.ge(min_days)
            & table.valid_labeled_days.ge(min_days),
            [
                "symbol",
                "train_weighted_mean",
                "valid_weighted_mean",
                "train_labeled_days",
                "valid_labeled_days",
            ],
        ],
        "train_test": table.loc[
            table.train_labeled_days.ge(min_days)
            & table.test_labeled_days.ge(min_days),
            [
                "symbol",
                "train_weighted_mean",
                "test_weighted_mean",
                "train_labeled_days",
                "test_labeled_days",
            ],
        ],
    }
    return shared, pairwise


def make_figures(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = run / "figures"
    figure_data = run / "figure-data"
    figures.mkdir(exist_ok=True)
    figure_data.mkdir(exist_ok=True)
    means = pd.read_csv(run / "period-target-means.csv")
    _, pairwise = scatter_frame(means, MIN_DAYS)
    summary = json.loads((run / "period-shift-summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = (
        (
            "train_valid",
            "train_weighted_mean",
            "valid_weighted_mean",
            "2023 weighted target mean",
            summary["train_vs_valid"],
        ),
        (
            "train_test",
            "train_weighted_mean",
            "test_weighted_mean",
            "2024-2025 weighted target mean",
            summary["train_vs_test"],
        ),
    )
    for ax, (key, xname, yname, ylabel, stats) in zip(axes, panels, strict=False):
        frame = pairwise[key]
        frame.to_csv(figure_data / f"01-period-shift-{key}.csv", index=False)
        ax.scatter(frame[xname], frame[yname], s=12, alpha=0.75, linewidths=0)
        limits = [-0.5, 0.5]
        ax.plot(limits, limits, color="0.4", linewidth=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("2018-2022 weighted target mean")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"n={stats['n_stocks']}, Pearson {fmt(stats['pearson'], 3)}, Spearman {fmt(stats['spearman'], 3)}"
        )
    fig.suptitle("Same-stock target means, at least 50 labeled days in each period")
    fig.tight_layout()
    fig.savefig(figures / "01-period-target-shift.png", dpi=160)
    fig.savefig(figures / "01-period-target-shift.svg")
    plt.close(fig)

    contribution = pd.read_parquet(run / "identity-contribution-by-stock.parquet")
    train_means = means[
        [
            "symbol",
            "train_weighted_mean",
            "test_weighted_mean",
            "train_labeled_days",
            "test_labeled_days",
        ]
    ]
    joined = contribution.loc[contribution.split.eq("train")].merge(
        train_means, on="symbol", validate="many_to_one"
    )
    plot_rows = joined.loc[
        joined.labeled_days.ge(MIN_DAYS) & joined.train_labeled_days.ge(MIN_DAYS)
    ]
    plot_rows.to_csv(
        figure_data / "02-direct-contribution-vs-train-mean.csv", index=False
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    for ax, variant in zip(axes, IDENTITY_VARIANTS, strict=False):
        part = plot_rows.loc[plot_rows.variant.eq(variant)]
        for seed, group in part.groupby("seed"):
            ax.scatter(
                group.train_weighted_mean,
                group.mean_contribution,
                s=10,
                alpha=0.65,
                linewidths=0,
                label=f"seed {seed}",
            )
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_title(variant)
        ax.set_xlabel("2018-2022 weighted target mean")
    axes[0].set_ylabel("Train-period mean direct-identity contribution")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Original forecast minus direct-identity-zeroed forecast")
    fig.tight_layout()
    fig.savefig(figures / "02-direct-contribution.png", dpi=160)
    fig.savefig(figures / "02-direct-contribution.svg")
    plt.close(fig)

    metrics = pd.read_csv(run / "checkpoint-summary.csv")
    metrics.to_csv(figure_data / "03-checkpoint-mse.csv", index=False)
    order = ["original", "zero_direct_identity", "uniform_routing", "both"]
    variants = ["local", "market", "identity", "relation", "shuffled"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=True)
    for ax, split in zip(axes, ("train", "valid", "test"), strict=False):
        part = metrics.loc[metrics.split.eq(split)]
        width = 0.18
        for index, intervention in enumerate(order):
            rows = (
                part.loc[part.intervention.eq(intervention)]
                .set_index("variant")
                .reindex(variants)
            )
            positions = np.arange(len(variants)) + (index - 1.5) * width
            present = rows["mse_mean"].notna()
            ax.bar(
                positions[present.to_numpy()],
                rows.loc[present, "mse_mean"],
                width=width,
                label=intervention,
            )
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, rotation=30, ha="right")
        ax.set_title(split)
    axes[0].set_ylabel("Equal-date MSE at epoch-12 checkpoint")
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "03-checkpoint-mse.png", dpi=160)
    fig.savefig(figures / "03-checkpoint-mse.svg")
    plt.close(fig)


def fmt(value, digits=6):
    value = finite_or_none(value)
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def markdown_table(frame):
    def cell(value):
        if isinstance(value, float | np.floating):
            return fmt(value)
        if value is None or (isinstance(value, str) and value == ""):
            return "—"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def write_paper(run, record):
    baseline = pd.read_csv(run / "baseline-metrics.csv")
    shift = json.loads((run / "period-shift-summary.json").read_text())
    summary = pd.read_csv(run / "checkpoint-summary.csv")
    trace = pd.read_csv(run / "checkpoint-vs-online-trace.csv")
    alignment = pd.read_csv(run / "contribution-alignment.csv")
    contrasts = pd.read_csv(run / "intervention-contrast-summary.csv")
    audit = json.loads((run / "sample-audit.json").read_text())
    split_rows = {row["split"]: row["rows"] for row in audit["splits"]}
    focus = alignment.loc[
        alignment.quantity.eq("direct_contribution")
        & alignment.left_period.eq("train")
        & alignment.right_name.eq("weighted_target_mean")
    ]
    prediction_focus = alignment.loc[
        alignment.quantity.eq("original_prediction")
        & alignment.left_period.eq("train")
        & alignment.right_name.eq("weighted_target_mean")
    ]
    text = f"""# 股票身份是否记住训练期排名：冻结诊断

实验编号：{run.name}
父实验：{PARENT_RUN_ID}
实验时间与记录整理时间：2026-11-22（Asia/Shanghai）。机器日志另存 UTC。
状态：诊断完成。没有重新训练。

本记录区分四件事。观察是训练期、2023 和 2024–2025 的标签均值以及已保存 checkpoint 的损失。诊断性干预是在冻结权重上只关掉一条路径。假设是“16 维身份向量记住了训练期收益排名，随后发生漂移”。再训练因果证据本轮没有产生。身份向量的某个坐标不被解释为排名。

## 摘要

父实验把身份向量同时送进预测头和静态路由。市场状态版本的验证损失已经差于纯个股版本。这里用同一批净化标签，先计算等日期平方损失下的股票常数最优预测，再对 3 个 seed、5 种结构的第 12 轮 checkpoint 做冻结推理。身份、关系和打乱关系另外各做三种干预：只把送入预测头的身份置零、只把路由权重换成均匀分布、两者同时进行。均匀路由仍剔除股票自身。这些干预改变了预测头在训练时见过的输入分布，所以损失变化不能单独当成“去掉该路径再训练”的效果。

## 1. 背景与研究问题

父实验的集成测试 Rank IC 没有支持关系结构带来可交易增量，训练损失下降而验证损失上升，和身份记忆、时段变化或容量过大相容。本轮只检验一个更窄的机制问题：直接进入预测头的身份项，是否在数值上贴近训练期的股票排名均值，以及这种贴近是否延续到 2023 和 2024–2025。

事前不把验证或测试结果用于选择模型。全部 seed 都报告。

## 2. 数据与样本

输入、标签、词表和切分重建自父实验的 `prepare`。重建后的预处理字典与父实验保存的预处理字典完全一致，权重 SHA256 与父实验 `model-verification.json` 一致。原始行情、因子矩阵和权重不复制到本目录。

切分仍是 2018–2022 训练、2023 验证、2024–2025 测试。标签仍是未来五日复权收益的当日百分位减 0.5。跨出本切分终点的标签已在父流程中剔除。身份词表只含训练期 feature-ready 股票。未知 ID 的预测基线和嵌入都按 0 处理。

样本审计给出的有标签行数：训练 {split_rows["train"]}，验证 {split_rows["valid"]}，测试 {
        split_rows["test"]
    }。未知身份的 feature-ready 行数：训练 {
        audit["unknown_identity_rows_by_split"]["train"]
    }，验证 {audit["unknown_identity_rows_by_split"]["valid"]}，测试 {
        audit["unknown_identity_rows_by_split"]["test"]
    }。

## 3. 方法与公式

设日期 t 有 n_t 只有标签股票，股票 i 的标签为 z_{{i,t}}，出现指示为 o_{{i,t}}。等日期损失是

$$L=\\frac{{1}}{{|D|}}\\sum_{{t\\in D}}\\frac{{1}}{{n_t}}\\sum_i o_{{i,t}}(\\hat y_{{i,t}}-z_{{i,t}})^2.$$

只允许每只股票一个常数时，最优常数是

$$\\mu_i=\\frac{{\\sum_t (o_{{i,t}}/n_t) z_{{i,t}}}}{{\\sum_t o_{{i,t}}/n_t}}.$$

它一般不等于该股票标签的简单平均。训练期没有标签的股票，以及未知 ID，预测为 0。零预测基线对所有股票输出 0。

训练期内用 μ_i 回代训练日，会把当天标签算进被评价的预测，因此训练指标是样本内拟合。另报告去掉当天后的 μ_i^{{(-t)}}，用来看这一机械影响有多大。验证和测试只使用训练期 μ_i。

Rank IC 沿用父实验：每个交易日计算预测与未来五日收益的 Spearman 相关，再对日等权。预测在截面上没有变异时不计算 Rank IC。平方损失的对象仍是排名标签，不是原始收益。

冻结干预只改推理输入。记原预测为 s(h,m,e,c)。只关闭直接身份时计算 s(h,m,0,c)，路由使用的 e 保持原值。只替换路由时，查询和信息源的组权重都改为 1/8，c 仍按剔除自身的池化计算，送入预测头的 e 保持原值。打乱版本先把均匀权重放到原分配上，再做原来的来源置换；各股票权重相同，置换不改变来源权重。两种干预同时进行时，头输入为 s(h,m,0,c_{{uniform}})。

股票 i 在某个切分上的直接身份贡献，是该切分有标签日期上 s(h,m,e,c)-s(h,m,0,c) 的平均值。因为预测头含 GELU，这个差不是 e_i 的线性坐标，也不能从坐标大小读出排名。

路由正则不进入本轮 MSE。本轮没有估计熵惩罚或均衡惩罚本身的作用。

## 4. 实验设计

对照是零预测、训练期加权股票均值、以及 15 个已保存 checkpoint 的原始推理。身份、关系、打乱关系各有三种冻结干预。纯个股和市场版本只作为不使用身份的参照，不额外改写路径。

Checkpoint 是固定的第 12 轮结束权重。父实验训练日志里的训练 MSE 是该轮更新过程中的批次损失，不是这批最终权重在全体训练日上的回放。验证日志才是每轮结束时的全体验证日回放。本轮用验证回放差异检查加载是否一致。

相关表只保留左右两侧都至少有 50 个有标签交易日的股票。这个阈值是报告规则，不是显著性检验，也没有做多重比较校正。

## 5. 评价方法

平方损失按父实验的等日期口径。Rank IC 按日 Spearman 后简单平均。时段相关比较的是同一股票在两个时段的加权标签均值。贡献相关比较的是贡献与对应时段的加权标签均值。另把纯个股和市场模型的股票平均预测与标签均值做同样的相关，作为没有身份通道时的参照。

三个 seed 都保留。汇总表报告 seed 间均值和样本标准差。

## 6. 结果与图

### 6.1 标签均值是否跨期保持

这是标签本身的观察，不是模型结果。加权均值使用上面的 1/n_t 权重。

{
        markdown_table(
            pd.DataFrame(
                [
                    {"comparison": "2018-2022 vs 2023", **shift["train_vs_valid"]},
                    {"comparison": "2018-2022 vs 2024-2025", **shift["train_vs_test"]},
                    {"comparison": "2023 vs 2024-2025", **shift["valid_vs_test"]},
                ]
            )
        )
    }

简单平均和加权平均的差异见 `results/period-target-means.csv` 的成对列。图 1 只画两侧都不少于 50 个有标签日的股票。图中斜线是相等线，不是拟合线。

![01-period-target-shift](figures/01-period-target-shift.png)

### 6.2 股票均值基线与零预测

训练行中的 in-sample 用含当天的 μ_i。loo 用去掉当天的 μ_i。验证和测试是冻结的训练期 μ_i。零预测没有截面差异，Rank IC 为空。

{markdown_table(baseline)}

### 6.3 第 12 轮 checkpoint 的等日期 MSE

下表是三个 seed 的均值。个体 seed、每日 MSE 和 Rank IC 见结果目录。在线训练日志与 checkpoint 验证 MSE 的最大绝对差为 {
        fmt(trace.valid_abs_diff.max(), 8)
    }。

{markdown_table(summary)}

![03-checkpoint-mse](figures/03-checkpoint-mse.png)

### 6.4 冻结路径干预

差值是干预后的 MSE 减去原始 checkpoint MSE。正值表示该次离分布干预增大了等日期损失。它不是重新训练后的增量。

{markdown_table(contrasts)}

### 6.5 直接身份贡献与训练期排名均值

贡献是原预测减去只关闭直接身份后的预测，再按股票在该切分的有标签日平均。图 2 的横轴是训练期加权标签均值，纵轴是训练期贡献。三个 seed 都画，没有挑选。

{markdown_table(focus)}

没有身份通道的模型，用原始预测的股票均值代替贡献，回答特征本身是否已经带有稳定的股票水平：

{markdown_table(prediction_focus)}

![02-direct-contribution](figures/02-direct-contribution.png)

完整的按股票贡献、预测均值和相关表都在 `results/`。未知 ID 上，关闭直接身份前后的预测最大绝对差见 `evidence/intervention-checks.json`，应接近 0，因为未知嵌入本来就是 0，而且这条干预不改上下文。

## 7. 验证与局限

方法测试覆盖加权均值的闭式解、它与简单平均的差别、未知 ID 回退为 0，以及两条干预路径的隔离。运行时另外检查诊断前向与原模块前向一致、未知 ID 贡献不泄漏、身份版本的均匀路由干预不改变预测。权重哈希在读取前后各检查一次。

局限：

- 冻结置零和均匀路由都是训练分布外的输入。预测头可能只是对没见过的输入不稳定，而不代表这条路径保存了排名。
- 预测头非线性，贡献不等于嵌入坐标，也不等于 μ_i。
- 训练期基线回代含有当天标签；loo 行只说明机械影响的量级；相邻五日标签重叠，这不是时间外验证。
- 路由正则、权重衰减和学习率没有被本轮分离。要分离它们必须重新训练。
- 2024–2025 已在父实验和更早实验中查看过，这里仍是探索性测试。
- 50 日阈值和多张相关表没有校正多重比较。
- 供应商时点仍未认证。本轮不产生收益或可交易 alpha 结论。

## 8. 结论与后续实验

本轮支持的只是描述性和诊断性陈述，具体数字以上表为准。标签均值跨期相关描述排名是否保持。贡献相关描述冻结的直接身份项是否与这些均值同向。干预前后的 MSE 描述当前权重对这条输入的依赖，不能外推为重新训练的结果。

尚未回答、需要小规模再训练才能区分的对照：

1. 训练时就去掉直接身份、保留路由，对比父实验的 identity 与 relation。这把本轮的离分布置零变成训练内干预。
2. 用冻结的训练期 μ_i 作为一维特征，替代 16 维可学习身份，看预测头是否只需要这个常数。
3. 从已保存的 market checkpoint 继续，只训练身份和预测头，分离“特征已经拟合的股票水平”和“身份通道新增的水平”。
4. 关系模型去掉熵惩罚和均衡惩罚后重训一个 seed，分离正则对路由的影响。本轮没有估计这项。
5. 固定打乱股票与 ID 的一一映射只是查表参数重命名，保留了全部身份信息，不能作为去身份对照。优先比较从训练开始删除直接身份通道、以及完全不用 ID 的共享状态路由。

这些对照都应使用新的 run_id，并保留本轮负结果和全部 seed。

## 9. 复现

```bash
./scripts/quant-own-exec python /app/research/stock_relations/diagnose_stock_identity.py --workspace /home/lyy/workspace/Quant --run-id NEW_RUN_ID
```

代码见 `code/diagnose_stock_identity.py` 和 `code/test_identity_audit.py`。图由同一脚本根据结果表绘制。输入与权重引用见 `references.json`。父实验论文和原始运行目录没有修改。
"""
    (record / "paper.md").write_text(text, encoding="utf-8")
    (record / "README.md").write_text(
        "# 股票身份冻结诊断\n\n[正文](paper.md) · [来源](references.json)\n\n"
        "results 只含指标、聚合预测差和标签均值。不包含原始行情或输入特征矩阵。权重仍在父实验 models 目录。\n",
        encoding="utf-8",
    )


def archive_record(run, references):
    from paper_records import update_index, update_manifest

    record = ROOT / "experiments/paper-records" / run.name
    if record.exists():
        raise FileExistsError(record)
    for folder in ("results", "code", "evidence", "verification", "figures"):
        (record / folder).mkdir(parents=True)
    result_names = (
        "baseline-metrics.csv",
        "period-target-means.csv",
        "period-shift-summary.json",
        "checkpoint-metrics.csv",
        "checkpoint-summary.csv",
        "checkpoint-vs-online-trace.csv",
        "daily-mse.parquet",
        "identity-contribution-by-stock.parquet",
        "stock-prediction-means.parquet",
        "contribution-alignment.csv",
        "intervention-contrasts.csv",
        "intervention-contrast-summary.csv",
        "summary.json",
    )
    for name in result_names:
        shutil.copy2(run / name, record / "results" / name)
    if (run / "figure-data").is_dir():
        shutil.copytree(run / "figure-data", record / "results" / "figure-data")
    if (run / "figures").is_dir():
        for path in (run / "figures").iterdir():
            shutil.copy2(path, record / "figures" / path.name)
    for name in (
        "status.json",
        "run.log",
        "provenance.json",
        "sample-audit.json",
        "method-tests.log",
        "intervention-checks.json",
        "model-hash-check.json",
        "preprocessing.json",
    ):
        if (run / name).exists():
            shutil.copy2(run / name, record / "evidence" / name)
    shutil.copy2(Path(__file__), record / "code" / "diagnose_stock_identity.py")
    shutil.copy2(
        Path(__file__).with_name("test_identity_audit.py"),
        record / "code" / "test_identity_audit.py",
    )
    for name in ("model.py", "reference.py", "config-v2.json"):
        shutil.copy2(Path(__file__).with_name(name), record / "code" / name)
    shutil.copy2(run / "method-tests.log", record / "verification" / "method-tests.log")
    write_json(record / "references.json", references)
    write_json(
        record / "record-status.json",
        {
            "status": "complete",
            "experiment_status": "complete",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "note": "Frozen diagnostic. No retraining and no alpha claim.",
        },
    )
    write_paper(run, record)
    update_manifest(record)
    update_index(record.parent)
    return record


def evaluate_baselines(
    run, target, observed, future, ids, symbols, split_codes, day_masks
):
    rows = []
    daily_rows = []
    per_period = []
    for split_name, _code in split_codes.items():
        mask = day_masks[split_name]
        means, weight_sum, labeled, _ = weighted_stock_means(
            target[mask], observed[mask]
        )
        per_period.append(
            {
                "split": split_name,
                "means": means,
                "weight_sum": weight_sum,
                "labeled": labeled,
            }
        )
    by_split = {item["split"]: item for item in per_period}
    train_forecast = apply_known_means(by_split["train"]["means"], ids)
    loo = np.zeros(target.shape, dtype=np.float64)
    train_rows = np.flatnonzero(day_masks["train"])
    loo[train_rows] = leave_one_day_forecasts(
        target[train_rows], observed[train_rows], ids
    )
    specs = [
        ("zero", "constant", "train", np.zeros(len(ids)), day_masks["train"]),
        ("zero", "constant", "valid", np.zeros(len(ids)), day_masks["valid"]),
        ("zero", "constant", "test", np.zeros(len(ids)), day_masks["test"]),
        ("stock_mean", "insample", "train", train_forecast, day_masks["train"]),
        ("stock_mean", "loo", "train", loo, day_masks["train"]),
        (
            "stock_mean",
            "frozen_train_mean",
            "valid",
            train_forecast,
            day_masks["valid"],
        ),
        ("stock_mean", "frozen_train_mean", "test", train_forecast, day_masks["test"]),
    ]
    for model, evaluation, split_name, forecast, mask in specs:
        indexes = np.flatnonzero(mask & observed.any(axis=1))
        mse, daily = equal_date_mse_rows(forecast, target, observed, indexes)
        rank_ic, rank_days = mean_daily_rank_ic(forecast, future, observed, indexes)
        rows.append(
            {
                "model": model,
                "evaluation": evaluation,
                "split": split_name,
                "equal_date_mse": mse,
                "rank_ic": rank_ic,
                "rank_ic_days": rank_days,
                "labeled_rows": int(observed[indexes].sum()),
            }
        )
        for day, value, count in zip(
            indexes, daily, observed[indexes].sum(axis=1), strict=False
        ):
            daily_rows.append(
                {
                    "model": model,
                    "seed": None,
                    "variant": model,
                    "intervention": evaluation,
                    "split": split_name,
                    "date_index": int(day),
                    "mse": float(value),
                    "n_labeled": int(count),
                }
            )
    table = pd.DataFrame(
        {
            "symbol": list(symbols),
            "identity_id": ids,
            "known_identity": ids > 0,
        }
    )
    for item in per_period:
        name = item["split"]
        table[f"{name}_weighted_mean"] = item["means"]
        table[f"{name}_weight_sum"] = item["weight_sum"]
        table[f"{name}_labeled_days"] = item["labeled"]
        unweighted_total = np.where(
            observed[day_masks[name]], target[day_masks[name]], 0.0
        ).sum(axis=0)
        unweighted = np.full(len(ids), np.nan)
        present = item["labeled"] > 0
        unweighted[present] = unweighted_total[present] / item["labeled"][present]
        table[f"{name}_unweighted_mean"] = unweighted
    table.to_csv(run / "period-target-means.csv", index=False)
    shift = {
        "train_vs_valid": {
            "left": "train",
            "right": "valid",
            **correlation_pair(
                table.train_weighted_mean,
                table.train_labeled_days,
                table.valid_weighted_mean,
                table.valid_labeled_days,
                MIN_DAYS,
            ),
        },
        "train_vs_test": {
            "left": "train",
            "right": "test",
            **correlation_pair(
                table.train_weighted_mean,
                table.train_labeled_days,
                table.test_weighted_mean,
                table.test_labeled_days,
                MIN_DAYS,
            ),
        },
        "valid_vs_test": {
            "left": "valid",
            "right": "test",
            **correlation_pair(
                table.valid_weighted_mean,
                table.valid_labeled_days,
                table.test_weighted_mean,
                table.test_labeled_days,
                MIN_DAYS,
            ),
        },
        "counts": {
            "train_at_least_min_days": int(
                (table.train_labeled_days >= MIN_DAYS).sum()
            ),
            "valid_at_least_min_days": int(
                (table.valid_labeled_days >= MIN_DAYS).sum()
            ),
            "test_at_least_min_days": int((table.test_labeled_days >= MIN_DAYS).sum()),
            "train_and_valid": int(
                (
                    (table.train_labeled_days >= MIN_DAYS)
                    & (table.valid_labeled_days >= MIN_DAYS)
                ).sum()
            ),
            "train_and_test": int(
                (
                    (table.train_labeled_days >= MIN_DAYS)
                    & (table.test_labeled_days >= MIN_DAYS)
                ).sum()
            ),
            "all_three_periods": int(
                (
                    (table.train_labeled_days >= MIN_DAYS)
                    & (table.valid_labeled_days >= MIN_DAYS)
                    & (table.test_labeled_days >= MIN_DAYS)
                ).sum()
            ),
            "min_days": MIN_DAYS,
        },
        "unweighted_sensitivity": {
            "train_vs_valid": correlation_pair(
                table.train_unweighted_mean,
                table.train_labeled_days,
                table.valid_unweighted_mean,
                table.valid_labeled_days,
                MIN_DAYS,
            ),
            "train_vs_test": correlation_pair(
                table.train_unweighted_mean,
                table.train_labeled_days,
                table.test_unweighted_mean,
                table.test_labeled_days,
                MIN_DAYS,
            ),
        },
    }
    write_json(run / "period-shift-summary.json", shift)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "baseline-metrics.csv", index=False)
    return metrics, table, daily_rows, by_split["train"]


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--parent-run", default=PARENT_RUN_ID)
    parser.add_argument("--batch-dates", type=int, default=32)
    args = parser.parse_args()
    ROOT = args.workspace.resolve()
    reference_cfg = json.loads(Path(__file__).with_name("config-v2.json").read_text())
    load_reference(ROOT, reference_cfg)
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unsafe run id")
    run = ROOT / "experiments/runs" / args.run_id
    parent = ROOT / "experiments/runs" / args.parent_run
    if run.exists() or (ROOT / "experiments/paper-records" / args.run_id).exists():
        raise FileExistsError(
            "Refusing to overwrite an existing diagnostic run or paper record"
        )
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    print("Running stock-identity diagnostic tests", flush=True)
    test_code, test_log = method_test_report()
    if test_code:
        sys.stderr.write(test_log)
        raise RuntimeError(
            "Diagnostic method tests failed; run directory was not created"
        )
    run.mkdir(parents=True)
    (run / "method-tests.log").write_text(test_log, encoding="utf-8")
    logfile = (run / "run.log").open("x", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, logfile)
    sys.stderr = TeeStream(sys.stderr, logfile)
    write_json(
        run / "status.json",
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "parent_run": args.parent_run,
        },
    )
    try:
        execute(run, parent, args)
    except BaseException:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "parent_run": args.parent_run,
            },
        )
        raise


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(run, parent, args):
    from factor_research import digest
    from stock_relation_research import permutation_tensor, prepare, tensor_data

    cfg = json.loads((parent / "research-config.json").read_text())
    shutil.copy2(parent / "research-config.json", run / "source-research-config.json")
    log("Reconstructing the parent sample with the existing prepare loader")
    data, prep, provenance = prepare(cfg, run)
    expected_prep = json.loads((parent / "preprocessing.json").read_text())
    assert_prep_match(prep, expected_prep)
    parent_prov = json.loads((parent / "provenance.json").read_text())
    if (
        provenance["vendor_logical_hash"] != parent_prov["vendor_logical_hash"]
        or provenance["market_index_hash"] != parent_prov["market_index_hash"]
    ):
        raise ValueError("Vendor or market-index vintage does not match the parent run")
    write_json(run / "preprocessing.json", prep)
    write_json(run / "provenance.json", provenance)
    split_codes = {
        name: code for code, name in enumerate(prep["parent_split_config"]["splits"], 1)
    }
    if list(split_codes) != ["train", "valid", "test"]:
        raise ValueError("Unexpected split order")
    expected_windows = {
        "train": ["2018-01-01", "2022-12-31"],
        "valid": ["2023-01-01", "2023-12-31"],
        "test": ["2024-01-01", "2025-12-31"],
    }
    if prep["parent_split_config"]["splits"] != expected_windows:
        raise ValueError("Split windows are not the parent study windows")
    observed = np.asarray(data["observed"], dtype=bool)
    target = np.asarray(data["target"], dtype=np.float64)
    ids = np.asarray(data["ids"], dtype=np.int64)
    split = np.asarray(data["split"])
    if np.any(observed[(split == split_codes["train"])] & (ids[None, :] == 0)):
        raise ValueError("Training labels include an unknown identity")
    samples = data["samples"]
    if samples.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("Duplicate labeled keys")
    future = np.full(observed.shape, np.nan, dtype=np.float64)
    date_index = data["dates"].get_indexer(pd.to_datetime(samples.trade_date))
    symbol_index = data["symbols"].get_indexer(samples.symbol)
    if np.any(date_index < 0) or np.any(symbol_index < 0):
        raise ValueError("Labeled sample is outside the model grid")
    future[date_index, symbol_index] = samples.future_return.to_numpy(dtype=np.float64)
    if not np.isfinite(future[observed]).all() or np.isfinite(future[~observed]).any():
        raise ValueError("Future-return grid does not match the labeled mask")
    labels = pd.read_parquet(
        provenance["parent_features"], columns=["trade_date", "symbol", "label_end"]
    )
    purged = samples.merge(
        labels, on=["trade_date", "symbol"], how="left", validate="one_to_one"
    )
    for name, (_, end) in expected_windows.items():
        part = purged.loc[purged.split.eq(name), "label_end"]
        if part.isna().any() or not part.le(pd.Timestamp(end)).all():
            raise ValueError("Unpurged crossing label")
    max_rank_gap = 0.0
    for day in np.flatnonzero(observed.any(axis=1)):
        mask = observed[day]
        expected = pd.Series(future[day, mask]).rank(pct=True).to_numpy() - 0.5
        max_rank_gap = max(
            max_rank_gap, float(np.max(np.abs(expected - target[day, mask])))
        )
    if max_rank_gap > 1e-5:
        raise ValueError(
            "Grid target is not the purged daily rank percentile minus 0.5"
        )
    log(f"Label grid matches daily rank targets; max absolute gap {max_rank_gap:.3g}")
    day_masks = {name: split == code for name, code in split_codes.items()}
    symbols = data["symbols"]
    dates = pd.DatetimeIndex(data["dates"])
    baseline, means_table, daily_rows, train_stats = evaluate_baselines(
        run, target, observed, future, ids, symbols, split_codes, day_masks
    )
    log("Baseline and period-shift tables written")
    data.pop("raw_rank", None)
    data.pop("raw_keys", None)
    data.pop("keys", None)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading checkpoints on {device}")
    checkpoints, model_dir = load_parent_checkpoints(parent, prep, device)
    tensors = tensor_data(data, device)
    trace = pd.read_csv(parent / "training-trace.csv")
    trace_final = trace.loc[trace.epoch.eq(cfg["epochs"])].set_index("model")
    metric_rows = []
    contribution_frames = []
    prediction_frames = []
    isolation_reports = []
    score_dates = np.flatnonzero(
        np.isin(split, list(split_codes.values())) & observed.any(axis=1)
    )
    probe_dates = score_dates[np.linspace(0, len(score_dates) - 1, 8, dtype=int)]
    permutations = {}
    for seed in cfg["seeds"]:
        permutations[seed] = permutation_tensor(data, seed, device)
    for item in checkpoints:
        saved = item["saved"]
        variant, seed = saved["variant"], int(saved["seed"])
        model = StockRelationNet(
            prep["input_dim"], prep["market_dim"], prep["known_stocks"], cfg, variant
        ).to(device)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if missing != ["state_projection.weight"] or unexpected:
            raise ValueError("Unexpected checkpoint architecture mismatch")
        model.eval()
        permutation = permutations[seed]
        forward_gap = check_forward_matches_module(
            model, tensors, permutation, probe_dates
        )
        if forward_gap > 1e-6:
            raise RuntimeError(
                f"Diagnostic forward differs from the module for {item['name']}: {forward_gap}"
            )
        check_path_isolation(model, tensors, permutation, probe_dates)
        interventions = INTERVENTIONS if variant in IDENTITY_VARIANTS else ("original",)
        predictions = {}
        for intervention in interventions:
            log(f"Inference {item['name']} {intervention}")
            narrow = predict_intervention(
                model, tensors, permutation, score_dates, intervention, args.batch_dates
            )
            full = np.full(observed.shape, np.nan, dtype=np.float32)
            full[score_dates] = narrow
            predictions[intervention] = full
        if variant in IDENTITY_VARIANTS:
            unknown = ids == 0
            if unknown.any():
                compared = predictions["original"][score_dates][:, unknown]
                unknown_gap = float(
                    np.max(
                        np.abs(
                            compared
                            - predictions["zero_direct_identity"][score_dates][
                                :, unknown
                            ]
                        )
                    )
                )
            else:
                unknown_gap = 0.0
            identity_uniform_gap = None
            if variant == "identity":
                identity_uniform_gap = float(
                    np.max(
                        np.abs(
                            predictions["original"][score_dates]
                            - predictions["uniform_routing"][score_dates]
                        )
                    )
                )
                if identity_uniform_gap > 1e-6:
                    raise RuntimeError(
                        "Uniform routing changed the identity model, which has no routing path"
                    )
            if unknown_gap > 1e-6:
                raise RuntimeError(
                    "Direct-identity ablation changed unknown-id forecasts"
                )
        else:
            unknown_gap = None
            identity_uniform_gap = None
        isolation_reports.append(
            {
                "model": item["name"],
                "forward_match_max_abs": forward_gap,
                "unknown_direct_identity_max_abs": unknown_gap,
                "identity_uniform_routing_max_abs": identity_uniform_gap,
            }
        )
        per_split_pred = {}
        for intervention, grid in predictions.items():
            for split_name in split_codes:
                indexes = np.flatnonzero(day_masks[split_name] & observed.any(axis=1))
                mse, daily = equal_date_mse_rows(grid, target, observed, indexes)
                rank_ic, rank_days = mean_daily_rank_ic(grid, future, observed, indexes)
                trace_mse = (
                    float32_trace_mse(grid, data["target"], observed, indexes)
                    if intervention == "original"
                    else None
                )
                metric_rows.append(
                    {
                        "model": item["name"],
                        "variant": variant,
                        "seed": seed,
                        "intervention": intervention,
                        "split": split_name,
                        "equal_date_mse": mse,
                        "float32_trace_style_mse": trace_mse,
                        "rank_ic": rank_ic,
                        "rank_ic_days": rank_days,
                        "labeled_rows": int(observed[indexes].sum()),
                    }
                )
                for day, value, count in zip(
                    indexes, daily, observed[indexes].sum(axis=1), strict=False
                ):
                    daily_rows.append(
                        {
                            "model": item["name"],
                            "seed": seed,
                            "variant": variant,
                            "intervention": intervention,
                            "split": split_name,
                            "date_index": int(day),
                            "mse": float(value),
                            "n_labeled": int(count),
                        }
                    )
                mean_prediction, counts = stock_mean(
                    grid, observed, day_masks[split_name]
                )
                per_split_pred[(intervention, split_name)] = mean_prediction
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "model": item["name"],
                            "variant": variant,
                            "seed": seed,
                            "intervention": intervention,
                            "split": split_name,
                            "symbol": list(symbols),
                            "known_identity": ids > 0,
                            "labeled_days": counts,
                            "mean_prediction": mean_prediction,
                        }
                    )
                )
        if variant in IDENTITY_VARIANTS:
            for split_name in split_codes:
                contribution, counts = stock_mean(
                    predictions["original"] - predictions["zero_direct_identity"],
                    observed,
                    day_masks[split_name],
                )
                contribution_frames.append(
                    pd.DataFrame(
                        {
                            "variant": variant,
                            "seed": seed,
                            "split": split_name,
                            "symbol": list(symbols),
                            "known_identity": ids > 0,
                            "labeled_days": counts,
                            "mean_contribution": contribution,
                            "mean_prediction_original": per_split_pred[
                                ("original", split_name)
                            ],
                            "mean_prediction_zero_direct": per_split_pred[
                                ("zero_direct_identity", split_name)
                            ],
                            "mean_prediction_uniform": per_split_pred[
                                ("uniform_routing", split_name)
                            ],
                            "mean_prediction_both": per_split_pred[
                                ("both", split_name)
                            ],
                            "weighted_target_mean": means_table[
                                f"{split_name}_weighted_mean"
                            ].to_numpy(),
                        }
                    )
                )
        online = trace_final.loc[item["name"]]
        valid_indexes = np.flatnonzero(day_masks["valid"] & observed.any(axis=1))
        replay_valid = float32_trace_mse(
            predictions["original"], data["target"], observed, valid_indexes
        )
        if abs(replay_valid - float(online.valid_mse)) > 1e-4:
            raise RuntimeError(
                f"Checkpoint validation MSE does not match epoch-12 trace for {item['name']}: "
                f"replay {replay_valid:.8f}, trace {float(online.valid_mse):.8f}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run / "checkpoint-metrics.csv", index=False)
    summary = metrics.groupby(["variant", "intervention", "split"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mse_mean=("equal_date_mse", "mean"),
        mse_std=("equal_date_mse", "std"),
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
    )
    summary.to_csv(run / "checkpoint-summary.csv", index=False)
    trace_rows = []
    for item in checkpoints:
        online = trace_final.loc[item["name"]]
        current = metrics.loc[
            metrics.model.eq(item["name"]) & metrics.intervention.eq("original")
        ]
        train_row = current.loc[current.split.eq("train")].iloc[0]
        valid_row = current.loc[current.split.eq("valid")].iloc[0]
        trace_rows.append(
            {
                "model": item["name"],
                "variant": item["saved"]["variant"],
                "seed": int(item["saved"]["seed"]),
                "online_epoch12_train_mse": float(online.train_mse),
                "online_epoch12_valid_mse": float(online.valid_mse),
                "checkpoint_train_mse": float(train_row.equal_date_mse),
                "checkpoint_valid_mse": float(valid_row.equal_date_mse),
                "valid_abs_diff": abs(
                    float(valid_row.float32_trace_style_mse) - float(online.valid_mse)
                ),
            }
        )
    pd.DataFrame(trace_rows).to_csv(run / "checkpoint-vs-online-trace.csv", index=False)
    daily = pd.DataFrame(daily_rows)
    daily["seed"] = daily["seed"].astype("Int64")
    daily["trade_date"] = (
        pd.DatetimeIndex(dates)
        .take(daily["date_index"].to_numpy())
        .strftime("%Y-%m-%d")
    )
    daily.drop(columns="date_index").to_parquet(run / "daily-mse.parquet", index=False)
    predictions_table = pd.concat(prediction_frames, ignore_index=True)
    predictions_table.to_parquet(run / "stock-prediction-means.parquet", index=False)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    contributions.to_parquet(
        run / "identity-contribution-by-stock.parquet", index=False
    )
    alignment_rows = []
    period_means = {
        name: means_table[f"{name}_weighted_mean"].to_numpy() for name in split_codes
    }
    period_days = {
        name: means_table[f"{name}_labeled_days"].to_numpy() for name in split_codes
    }
    for variant in IDENTITY_VARIANTS:
        for seed in cfg["seeds"]:
            part = contributions.loc[
                contributions.variant.eq(variant) & contributions.seed.eq(seed)
            ]
            for left_split in split_codes:
                left = (
                    part.loc[part.split.eq(left_split)]
                    .set_index("symbol")
                    .reindex(means_table.symbol)
                )
                for right_split in split_codes:
                    stats = correlation_pair(
                        left.mean_contribution.to_numpy(),
                        left.labeled_days.to_numpy(),
                        period_means[right_split],
                        period_days[right_split],
                        MIN_DAYS,
                    )
                    alignment_rows.append(
                        {
                            "quantity": "direct_contribution",
                            "variant": variant,
                            "seed": seed,
                            "left_period": left_split,
                            "right_period": right_split,
                            "left_name": "mean_direct_contribution",
                            "right_name": "weighted_target_mean",
                            **stats,
                        }
                    )
    for quantity, intervention in (
        ("original_prediction", "original"),
        ("zero_direct_prediction", "zero_direct_identity"),
        ("uniform_prediction", "uniform_routing"),
        ("both_prediction", "both"),
    ):
        available = predictions_table.loc[
            predictions_table.intervention.eq(intervention)
        ]
        for variant in sorted(available.variant.unique()):
            for seed in cfg["seeds"]:
                part = available.loc[
                    available.variant.eq(variant) & available.seed.eq(seed)
                ]
                if part.empty:
                    continue
                left = (
                    part.loc[part.split.eq("train")]
                    .set_index("symbol")
                    .reindex(means_table.symbol)
                )
                for right_split in split_codes:
                    stats = correlation_pair(
                        left.mean_prediction.to_numpy(),
                        left.labeled_days.to_numpy(),
                        period_means[right_split],
                        period_days[right_split],
                        MIN_DAYS,
                    )
                    alignment_rows.append(
                        {
                            "quantity": quantity,
                            "variant": variant,
                            "seed": int(seed),
                            "left_period": "train",
                            "right_period": right_split,
                            "left_name": "mean_prediction",
                            "right_name": "weighted_target_mean",
                            **stats,
                        }
                    )
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(run / "contribution-alignment.csv", index=False)
    contrast_source = []
    for (variant, seed, split_name), group in metrics.groupby(
        ["variant", "seed", "split"]
    ):
        base = group.loc[group.intervention.eq("original")]
        if base.empty:
            continue
        row = {
            "variant": variant,
            "seed": int(seed),
            "split": split_name,
            "mse_original": float(base.equal_date_mse.iloc[0]),
            "rank_ic_original": finite_or_none(base.rank_ic.iloc[0]),
        }
        for intervention in INTERVENTIONS:
            current = group.loc[group.intervention.eq(intervention)]
            if current.empty:
                row[f"mse_{intervention}"] = None
                row[f"delta_mse_{intervention}"] = None
                row[f"rank_ic_{intervention}"] = None
                row[f"delta_rank_ic_{intervention}"] = None
                continue
            mse = float(current.equal_date_mse.iloc[0])
            rank_ic = finite_or_none(current.rank_ic.iloc[0])
            row[f"mse_{intervention}"] = mse
            row[f"delta_mse_{intervention}"] = mse - row["mse_original"]
            row[f"rank_ic_{intervention}"] = rank_ic
            row[f"delta_rank_ic_{intervention}"] = (
                None
                if rank_ic is None or row["rank_ic_original"] is None
                else rank_ic - row["rank_ic_original"]
            )
        contrast_source.append(row)
    contrasts = pd.DataFrame(contrast_source)
    contrasts.to_csv(run / "intervention-contrasts.csv", index=False)
    contrast_summary = contrasts.groupby(["variant", "split"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mse_original=("mse_original", "mean"),
        delta_mse_zero_direct_identity=("delta_mse_zero_direct_identity", "mean"),
        delta_mse_uniform_routing=("delta_mse_uniform_routing", "mean"),
        delta_mse_both=("delta_mse_both", "mean"),
        delta_rank_ic_zero_direct_identity=(
            "delta_rank_ic_zero_direct_identity",
            "mean",
        ),
        delta_rank_ic_uniform_routing=("delta_rank_ic_uniform_routing", "mean"),
        delta_rank_ic_both=("delta_rank_ic_both", "mean"),
    )
    contrast_summary.to_csv(run / "intervention-contrast-summary.csv", index=False)
    end_hashes = {item["name"]: digest(item["path"]) for item in checkpoints}
    if end_hashes != {item["name"]: item["sha256"] for item in checkpoints}:
        raise RuntimeError("A checkpoint hash changed during the diagnostic")
    write_json(
        run / "intervention-checks.json",
        {
            "probe_dates": int(len(probe_dates)),
            "models": isolation_reports,
            "max_target_rank_gap": max_rank_gap,
        },
    )
    write_json(
        run / "model-hash-check.json",
        {"model_directory": str(model_dir), "unchanged": True, "sha256": end_hashes},
    )
    write_json(
        run / "summary.json",
        {
            "run_id": run.name,
            "parent_run": parent.name,
            "device": str(device),
            "min_labeled_days": MIN_DAYS,
            "baseline": baseline.to_dict(orient="records"),
            "period_shift": json.loads((run / "period-shift-summary.json").read_text()),
            "checkpoint_summary": summary.to_dict(orient="records"),
            "intervention_summary": contrast_summary.to_dict(orient="records"),
            "direct_contribution_alignment": alignment.loc[
                alignment.quantity.eq("direct_contribution")
            ].to_dict(orient="records"),
            "claim_boundary": "Frozen off-distribution interventions and descriptive correlations. No retraining and no alpha claim.",
            "train_stock_weight_definition": "1/n_t on each purged labeled training row; unknown ids forecast 0",
        },
    )
    make_figures(run)
    references = {
        "parent_run": str(parent),
        "parent_provenance": str(parent / "provenance.json"),
        "parent_provenance_sha256": digest(parent / "provenance.json"),
        "model_directory": str(model_dir),
        "weights_read_only": True,
        "weights_sha256": end_hashes,
        "vendor_logical_hash": provenance["vendor_logical_hash"],
        "market_index_hash": provenance["market_index_hash"],
        "parent_features": provenance["parent_features"],
        "parent_features_sha256": provenance["parent_features_sha256"],
        "input_snapshot": provenance["input_snapshot"],
        "note": "No raw price panel, feature matrix, or weight file was copied into this diagnostic.",
    }
    write_json(run / "references.json", references)
    write_json(
        run / "status.json",
        {
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "paper_record": str(ROOT / "experiments/paper-records" / run.name),
            "device": str(device),
        },
    )
    record = archive_record(run, references)
    log(f"Diagnostic complete: {record}")


if __name__ == "__main__":
    main()
