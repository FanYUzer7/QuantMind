"""Causality, calendar and discontinuity checks using isolated synthetic fixtures."""

import numpy as np
import pandas as pd
import pytest

from backend.services.engine.data_platform.own_data.contracts import DataContractError
from backend.services.engine.data_platform.own_data.features import (
    FEATURES,
    align_panel,
    calendar_sessions,
    validate_continuous,
    build_research_features,
)


def panel():
    days = pd.bdate_range("2025-01-01", periods=100).strftime("%Y-%m-%d").tolist()
    frames = []
    for symbol in ["000001.SZ", "600519.SH"]:
        close = np.arange(100, dtype=float) + 100
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "trade_date": days,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000.0,
                    "amount": 100000.0,
                    "float_shares": 1000000.0,
                    "is_st": 0,
                }
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    factors = raw.copy()
    factors["adj_factor"] = 1.0
    return raw, factors, ["000001.SZ", "600519.SH"], days


def test_calendar_is_bounded_and_uses_official_closures():
    spec = {
        "valid_from": "2025-05-01",
        "valid_to": "2025-06-03",
        "venues": ["SH", "SZ"],
        "weekends_closed": True,
        "weekday_closures": ["2025-05-01", "2025-05-02", "2025-05-05", "2025-06-02"],
    }
    days = calendar_sessions(spec, "2025-05-01", "2025-06-03")
    assert (
        days[0] == "2025-05-06"
        and days[-1] == "2025-06-03"
        and "2025-06-02" not in days
    )
    with pytest.raises(DataContractError):
        calendar_sessions(spec, "2024-01-01", "2025-06-03")


def test_missing_session_and_duplicates_are_not_filled():
    raw, _, symbols, days = panel()
    with pytest.raises(DataContractError, match="mismatch"):
        align_panel(raw.iloc[1:], symbols, days)
    with pytest.raises(DataContractError, match="Duplicate"):
        align_panel(pd.concat([raw, raw.iloc[:1]]), symbols, days)
    with pytest.raises(DataContractError):
        align_panel(raw, symbols + [symbols[0]], days)


def test_observed_dates_do_not_define_the_calendar():
    raw, _, symbols, days = panel()
    weekend = {**raw.iloc[0].to_dict(), "trade_date": "2025-01-04"}
    with pytest.raises(DataContractError, match="extra"):
        align_panel(pd.concat([raw, pd.DataFrame([weekend])]), symbols, days)


def test_explicit_research_mode_and_real_warmup_required():
    raw, _, symbols, days = panel()
    with pytest.raises(DataContractError, match="research_preview"):
        build_research_features(raw, raw, symbols, days, days[70], days[-1])
    with pytest.raises(DataContractError, match="warm-up"):
        build_research_features(
            raw, raw, symbols, days, days[5], days[-1], research_preview=True
        )


def test_features_are_causal_and_backward_return_is_correct():
    raw, _, symbols, days = panel()
    first = build_research_features(
        raw, raw, symbols, days, days[65], days[-1], research_preview=True
    )
    changed = raw.copy()
    changed.loc[changed.trade_date > days[80], ["open", "high", "low", "close"]] *= 3
    second = build_research_features(
        changed, changed, symbols, days, days[65], days[-1], research_preview=True
    )
    pd.testing.assert_frame_equal(
        first.loc[first.trade_date <= days[80]],
        second.loc[second.trade_date <= days[80]],
    )
    assert np.isclose(first.iloc[0].mom_ret_5d, 165 / 160 - 1)
    assert not any("future" in col or "label" == col for col in first.columns)


def test_relative_features_do_not_depend_on_a_constant_price_anchor():
    raw, _, symbols, days = panel()
    scaled = raw.copy()
    scaled[["open", "high", "low", "close"]] *= 7
    a = build_research_features(
        raw, raw, symbols, days, days[65], days[-1], research_preview=True
    )
    b = build_research_features(
        raw, scaled, symbols, days, days[65], days[-1], research_preview=True
    )
    assert np.allclose(a[FEATURES], b[FEATURES], rtol=1e-10, atol=1e-10)
    assert np.allclose(a.raw_close, b.raw_close)


def test_unknown_st_and_nonpositive_activity_are_blocked():
    raw, factors, symbols, days = panel()
    bad = raw.copy()
    bad.loc[0, "is_st"] = np.nan
    with pytest.raises(DataContractError, match="ST"):
        validate_continuous(bad, raw, raw, factors, symbols, days)
    bad = raw.copy()
    bad.loc[0, "float_shares"] = 0
    with pytest.raises(DataContractError, match="positive"):
        build_research_features(
            bad, raw, symbols, days, days[65], days[-1], research_preview=True
        )


def test_split_discontinuity_needs_crossfile_factor_support():
    raw, factors, symbols, days = panel()
    adjusted = raw.copy()
    # An intentionally artificial split at a known row: raw halved, adjusted continuous.
    mask = raw.trade_date >= days[50]
    raw.loc[mask, ["open", "high", "low", "close"]] /= 2
    factors[["open", "high", "low", "close"]] = raw[["open", "high", "low", "close"]]
    factors.loc[mask, "adj_factor"] = 2
    report = validate_continuous(raw, adjusted, adjusted, factors, symbols, days)
    assert len(report["adjustment_events"]) == 2
    assert not report["official_corporate_actions_verified"]
    factors["adj_factor"] = 1
    with pytest.raises(DataContractError, match="disagree"):
        validate_continuous(raw, adjusted, adjusted, factors, symbols, days)


def test_adjustment_cannot_change_share_or_cash_activity():
    raw, factors, symbols, days = panel()
    bad = raw.copy()
    bad.loc[0, "volume"] *= 2
    with pytest.raises(DataContractError, match="volume/amount"):
        validate_continuous(raw, bad, raw, factors, symbols, days)


def test_source_raw_prices_must_agree_before_factor_crosscheck():
    raw, factors, symbols, days = panel()
    factors.loc[0, "close"] += 0.1
    with pytest.raises(DataContractError, match="raw price mismatch"):
        validate_continuous(raw, raw, raw, factors, symbols, days)
