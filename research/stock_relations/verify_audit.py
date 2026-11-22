"""Recompute baseline and original checkpoint MSE from archived predictions.

Independent of the diagnostic tensor grid and forward replay. This reads only
result data; raw features and checkpoints are not needed.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def verify(workspace, audit_id):
    parent = workspace / "experiments/runs/stock-relation-20261120-001"
    audit = workspace / "experiments/runs" / audit_id
    frame = pd.read_parquet(parent / "predictions.parquet")
    frame["target"] = frame.groupby("trade_date").future_return.rank(pct=True) - 0.5
    frame["weight"] = 1 / frame.groupby("trade_date").symbol.transform("size")
    train = frame.loc[frame.split.eq("train")].copy()
    train["weighted_target"] = train.target * train.weight
    sums = train.groupby("symbol")[["weighted_target", "weight"]].sum()
    means = sums.weighted_target / sums.weight
    frame["stock_mean"] = frame.symbol.map(means).fillna(0)
    baseline = pd.read_csv(audit / "baseline-metrics.csv")
    checkpoints = pd.read_csv(audit / "checkpoint-metrics.csv")
    errors = []
    for split, part in frame.groupby("split"):
        for model in ("zero", "stock_mean"):
            prediction = 0 if model == "zero" else part.stock_mean
            mse = (
                ((prediction - part.target) ** 2).groupby(part.trade_date).mean().mean()
            )
            selected = baseline.loc[
                baseline.model.eq(model)
                & baseline.split.eq(split)
                & ~baseline.evaluation.eq("loo")
            ]
            assert len(selected) == 1
            errors.append(abs(mse - selected.equal_date_mse.iloc[0]))
        for name in [c for c in part if re.fullmatch(r"nn_.+_s\d+", c)]:
            mse = (
                ((part[name] - part.target) ** 2).groupby(part.trade_date).mean().mean()
            )
            selected = checkpoints.loc[
                checkpoints.model.eq(name)
                & checkpoints.split.eq(split)
                & checkpoints.intervention.eq("original")
            ]
            assert len(selected) == 1
            errors.append(abs(mse - selected.equal_date_mse.iloc[0]))
    assert len(errors) == 51, len(errors)
    assert np.isfinite(errors).all() and max(errors) < 1e-7, max(errors)
    return {
        "status": "passed",
        "comparisons": len(errors),
        "max_mse_error": max(errors),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit-id", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace, args.audit_id), indent=2))
