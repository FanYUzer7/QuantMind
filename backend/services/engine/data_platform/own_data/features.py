"""Small, explicit research snapshots; no source fetching or production routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DataContractError

FEATURES = [
    "mom_ret_1d",
    "mom_ret_5d",
    "mom_ret_20d",
    "liq_volume",
    "liq_amount",
    "fun_turnover_1",
    "mom_ret_60d",
    "vol_std_20d",
    "mom_ma_gap_20",
]


def calendar_sessions(spec, start, end):
    if not spec["valid_from"] <= start <= end <= spec["valid_to"]:
        raise DataContractError("Calendar request outside reviewed window")
    if spec.get("venues") != ["SH", "SZ"] or not spec.get("weekends_closed"):
        raise DataContractError(
            "This research calendar is only for reviewed SH/SZ sessions"
        )
    dates = pd.bdate_range(start, end).strftime("%Y-%m-%d")
    return [day for day in dates if day not in spec["weekday_closures"]]


def align_panel(frame, symbols, sessions):
    if len(symbols) != len(set(symbols)) or sessions != sorted(set(sessions)):
        raise DataContractError("Symbols/sessions must be unique and sessions sorted")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise DataContractError("Duplicate symbol/date in continuous panel")
    expected = {(s, d) for s in symbols for d in sessions}
    actual = set(zip(frame["symbol"], frame["trade_date"], strict=True))
    if actual != expected:
        raise DataContractError(
            f"Calendar/panel mismatch: missing={sorted(expected - actual)[:8]}, extra={sorted(actual - expected)[:8]}"
        )
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def check_prices(frame):
    columns = ["open", "high", "low", "close"]
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise DataContractError("Nonfinite/nonpositive prices")
    if (values["high"] < values[["open", "close", "low"]].max(axis=1)).any() or (
        values["low"] > values[["open", "close"]].min(axis=1)
    ).any():
        raise DataContractError("Invalid OHLC ordering")


def check_activity_and_state(raw):
    values = raw[["volume", "amount", "float_shares"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise DataContractError(
            "This first research window requires positive finite volume, amount and float shares"
        )
    if not set(raw["is_st"].unique()).issubset({0, 1}) or raw["is_st"].isna().any():
        raise DataContractError("ST must be copied from a recognised source value")


def validate_continuous(raw, qfq, hfq, factors, symbols, sessions):
    frames = [align_panel(f, symbols, sessions) for f in (raw, qfq, hfq, factors)]
    raw, qfq, hfq, factors = frames
    for frame in frames[:3]:
        check_prices(frame)
    check_activity_and_state(raw)
    for frame in (qfq, hfq):
        if not np.allclose(
            frame[["volume", "amount"]], raw[["volume", "amount"]], rtol=0, atol=1e-6
        ):
            raise DataContractError("Adjustment unexpectedly changed volume/amount")
    if not np.allclose(
        factors[["open", "high", "low", "close"]],
        raw[["open", "high", "low", "close"]],
        rtol=0,
        atol=1e-6,
    ):
        raise DataContractError("Cross-file factor CSV raw price mismatch")
    if (
        not np.isfinite(factors["adj_factor"]).all()
        or (factors["adj_factor"] <= 0).any()
    ):
        raise DataContractError("Invalid source adjustment factor")
    reports, events = [], []
    for symbol in symbols:
        mask = raw["symbol"] == symbol
        r, q, h, f = [x.loc[mask].reset_index(drop=True) for x in frames]
        ratio = q["close"] / r["close"]
        ratio_change = ratio / ratio.shift(1) - 1
        factor_change = f["adj_factor"] / f["adj_factor"].shift(1) - 1
        # Rounded prices and rounded factor fields limit cross-source precision.
        relative_price_tolerance = 0.002
        expected = r["close"] * f["adj_factor"]
        expected *= q["close"].iloc[0] / expected.iloc[0]
        relative_error = abs(expected / q["close"] - 1)
        if relative_error.max() > relative_price_tolerance:
            raise DataContractError(f"Source factor and qfq path disagree for {symbol}")
        candidates = factor_change.abs() > 0.001
        for index in candidates[candidates].index:
            if index == 0:
                continue
            if abs(ratio_change.iloc[index] - factor_change.iloc[index]) > 0.002:
                raise DataContractError(
                    f"Unexplained adjustment discontinuity for {symbol}"
                )
            events.append(
                {
                    "symbol": symbol,
                    "trade_date": r["trade_date"].iloc[index],
                    "raw_return": r["close"].iloc[index] / r["close"].iloc[index - 1]
                    - 1,
                    "adjusted_return": q["close"].iloc[index]
                    / q["close"].iloc[index - 1]
                    - 1,
                    "qfq_raw_ratio_change": float(ratio_change.iloc[index]),
                    "crossfile_factor_change": float(factor_change.iloc[index]),
                    "status": "source_factor_crosscheck_supported",
                    "official_corporate_action_verified": False,
                }
            )
        reports.append(
            {
                "symbol": symbol,
                "sessions": len(r),
                "factor_reconstruction_max_relative_error": float(relative_error.max()),
                "crosscheck_relative_tolerance": relative_price_tolerance,
                "source_factor_anchor": "normalised_to_first_window_observation",
                "qfq_hfq_ratio_range": [
                    float((q["close"] / h["close"]).min()),
                    float((q["close"] / h["close"]).max()),
                ],
            }
        )
    return {
        "calendar_complete_for_selected_symbols": True,
        "zero_volume_rows": 0,
        "per_symbol": reports,
        "adjustment_events": events,
        "official_corporate_actions_verified": False,
        "warning": "Rounded source-factor crosscheck, not an independent reconstruction of all cash/split actions",
    }


def build_research_features(
    raw, adjusted, symbols, sessions, start, end, *, research_preview=False
):
    if not research_preview:
        raise DataContractError(
            "Explicit research_preview=True is required; production use is not authorised by these checks"
        )
    raw, adjusted = [align_panel(f, symbols, sessions) for f in (raw, adjusted)]
    check_prices(raw)
    check_prices(adjusted)
    check_activity_and_state(raw)
    result = []
    for symbol in symbols:
        r = raw.loc[raw["symbol"] == symbol].reset_index(drop=True)
        a = adjusted.loc[adjusted["symbol"] == symbol].reset_index(drop=True)
        out = a[["symbol", "trade_date", "open", "high", "low", "close"]].copy()
        for col in ("open", "high", "low", "close"):
            out["raw_" + col] = r[col]
        out["volume"], out["amount"] = r["volume"], r["amount"]
        out["is_st"] = r["is_st"]
        for horizon in (1, 5, 20, 60):
            out[f"mom_ret_{horizon}d"] = a["close"].pct_change(
                horizon, fill_method=None
            )
        out["liq_volume"] = np.log1p(r["volume"])
        out["liq_amount"] = np.log1p(r["amount"])
        out["fun_turnover_1"] = r["volume"] / r["float_shares"]
        out["vol_std_20d"] = out["mom_ret_1d"].rolling(20, min_periods=20).std(ddof=1)
        out["mom_ma_gap_20"] = (
            a["close"] / a["close"].rolling(20, min_periods=20).mean() - 1
        )
        out = out.loc[out["trade_date"].between(start, end)].copy()
        if not np.isfinite(out[FEATURES].to_numpy()).all():
            raise DataContractError(
                "Missing features: extend actual warm-up history, do not fill values"
            )
        result.append(out)
    frame = (
        pd.concat(result, ignore_index=True)
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )
    if frame.empty:
        raise DataContractError("Empty feature snapshot")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


FEATURE_DEFINITIONS = {
    "mom_ret_1d/5d/20d/60d": "qfq_close[t]/qfq_close[t-h]-1, per instrument, backward looking",
    "liq_volume": "log1p(raw volume in shares)",
    "liq_amount": "log1p(raw amount in CNY)",
    "fun_turnover_1": "raw volume / source float shares, fraction rather than percent",
    "vol_std_20d": "20-session std of 1d adjusted returns, ddof=1",
    "mom_ma_gap_20": "qfq_close / 20-session mean(qfq_close)-1",
}
