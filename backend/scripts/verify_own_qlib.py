"""Verify native Qlib reads and execute a small research backtest offline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.backtest import backtest
from qlib.contrib.strategy import TopkDropoutStrategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("Use a new validation output")
    qlib.init(
        provider_uri=str(a.dataset / "qlib"),
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
    )
    codes = D.instruments("all")
    native = D.features(
        codes,
        ["$close", "$factor", "$volume", "$amount"],
        start_time="2025-07-01",
        end_time="2025-09-30",
        freq="day",
    )
    if len(native) != 264:
        raise AssertionError("Unexpected native Qlib rows")
    original = pd.read_parquet(a.dataset / "model_features_2025.parquet")
    from backend.shared.stock_utils import StockCodeUtil

    original["instrument"] = original.symbol.map(StockCodeUtil.to_qlib)
    original = original.set_index(["instrument", "trade_date"]).sort_index()
    actual = native.sort_index()
    errors = {
        "raw_close_max_error": float(
            np.max(
                abs(
                    actual["$close"].to_numpy() / actual["$factor"].to_numpy()
                    - original.raw_close.to_numpy()
                )
            )
        ),
        "raw_volume_max_relative_error": float(
            np.max(
                abs(
                    actual["$volume"].to_numpy()
                    * actual["$factor"].to_numpy()
                    / original.volume.to_numpy()
                    - 1
                )
            )
        ),
        "amount_max_relative_error": float(
            np.max(abs(actual["$amount"].to_numpy() / original.amount.to_numpy() - 1))
        ),
    }
    if (
        errors["raw_close_max_error"] > 0.001
        or errors["raw_volume_max_relative_error"] > 1e-5
    ):
        raise AssertionError("Qlib canonical recovery failed")
    predictions = pd.read_pickle(a.model_run / "pred.pkl")
    signal = predictions["score"] if "score" in predictions else predictions.iloc[:, 0]
    if isinstance(signal.index, pd.MultiIndex):
        table = signal.rename("score").reset_index()
        table["instrument"] = table["instrument"].map(StockCodeUtil.to_qlib)
        signal = table.set_index(["datetime", "instrument"])["score"]
    strategy = TopkDropoutStrategy(signal=signal, topk=2, n_drop=1, only_tradable=True)
    days = original.index.get_level_values(1).unique()
    # Validation labels ending Sep 5 use closes through Sep 15 (T+1,H=5).
    # Start after that information is available; this run only checks execution.
    cash_benchmark = pd.Series(
        0.0, index=days[(days >= "2025-09-17") & (days <= "2025-09-19")]
    )
    portfolio, indicators = backtest(
        start_time="2025-09-17",
        end_time="2025-09-19",
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        account=1_000_000,
        benchmark=cash_benchmark,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.1,
            "deal_price": "close",
            "open_cost": 0.0003,
            "close_cost": 0.0013,
            "min_cost": 5,
        },
    )
    report = next(iter(portfolio.values()))[0]
    report.to_csv(a.output.with_suffix(".csv"))
    if report["total_turnover"].max() <= 0:
        raise AssertionError(
            "Backtest produced no executed trades; bridge is incomplete"
        )
    result = {
        "status": "passed",
        "research_only": True,
        "native_qlib_rows": len(native),
        "instruments": sorted(native.index.get_level_values(0).unique().tolist()),
        "canonical_recovery": errors,
        "backtest_executed": True,
        "backtest_days": len(report),
        "execution_window": ["2025-09-17", "2025-09-19"],
        "executed_turnover_cny": float(report["total_turnover"].iloc[-1]),
        "benchmark": "explicit zero-return cash baseline; no missing index fabricated",
        "warning": "Four-stock interface validation. No strategy performance claim; corporate actions and exchange constraints not independently certified.",
    }
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
