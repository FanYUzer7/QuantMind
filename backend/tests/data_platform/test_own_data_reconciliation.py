"""Tests for evidence thresholds and conservative admission, no data downloads."""

from copy import deepcopy

from backend.services.engine.data_platform.own_data import DailyBatch
from backend.services.engine.data_platform.own_data.reconciliation import (
    scale_hypothesis,
    diagnose_panel,
    assess_feature_input,
)


def panel(name="reference", scale=1, count=12):
    rows = []
    for i in range(count):
        amount = 1000 + i * 50
        volume = 100 + i * 10
        rows.append(
            {
                "symbol": "000001.SZ" if i % 2 else "600519.SH",
                "trade_date": f"2026-01-{i // 2 + 1:02d}",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "volume_shares": volume,
                "amount_cny": amount,
                "volume_raw": volume / scale,
                "amount_raw": amount / scale,
                "price_adjustment": "raw",
                "data_kind": "historical_daily",
                "asset_type": "stock",
                "quality_flags": [],
            }
        )
    return DailyBatch(name, "test-v1", rows=rows)


def test_lot_scale_requires_multiple_days_and_instruments():
    result = scale_hypothesis(
        panel(), panel("lots", 100), "volume_shares", "volume_raw"
    )
    assert result["status"] == "sample_supported" and result["supported_scale"] == 100
    assert not result["automatic_application"]


def test_small_sample_does_not_authorise_scale_even_if_exact():
    result = scale_hypothesis(
        panel(count=8), panel("lots", 100, count=8), "volume_shares", "volume_raw"
    )
    assert result["status"] == "insufficient_evidence"
    assert result["supported_scale"] is None


def test_single_instrument_cannot_pass_multi_instrument_gate():
    a, b = panel(), panel("lots", 100)
    for i, r in enumerate(a.rows):
        r["symbol"] = "600519.SH"
        r["trade_date"] = f"2026-01-{i + 1:02d}"
        b.rows[i]["symbol"] = r["symbol"]
        b.rows[i]["trade_date"] = r["trade_date"]
    assert (
        scale_hypothesis(a, b, "volume_shares", "volume_raw")["supported_scale"] is None
    )


def test_zero_rows_provide_no_unit_evidence():
    a, b = panel(), panel("candidate")
    for x, y in zip(a.rows, b.rows, strict=True):
        x["volume_shares"] = y["volume_raw"] = 0
    result = scale_hypothesis(a, b, "volume_shares", "volume_raw")
    assert result["usable_pairs"] == 0 and result["supported_scale"] is None


def test_missing_and_negative_values_do_not_create_scale_hints():
    a, b = panel(), panel("candidate")
    for i, y in enumerate(b.rows):
        y["volume_raw"] = None if i % 2 else -1
    assert (
        scale_hypothesis(a, b, "volume_shares", "volume_raw")["status"]
        == "insufficient_evidence"
    )


def test_mixed_units_are_flagged_inconsistent_not_majority_merged():
    a, b = panel(), panel("candidate", 100)
    for y in b.rows[:3]:
        y["volume_raw"] *= 100
    result = scale_hypothesis(a, b, "volume_shares", "volume_raw")
    assert (
        result["status"] == "inconsistent_or_ambiguous"
        and result["supported_scale"] is None
    )


def test_duplicate_keys_are_excluded_from_scale_evidence():
    a, b = panel(), panel("candidate", 100)
    b.rows.append(deepcopy(b.rows[0]))
    result = scale_hypothesis(a, b, "volume_shares", "volume_raw")
    assert result["usable_pairs"] == 11 and result["duplicate_keys_excluded"] == 1


def test_unknown_price_basis_is_not_certified_by_equal_values():
    a, b = panel(), panel("candidate")
    for y in b.rows:
        y["price_adjustment"] = "unknown"
    result = diagnose_panel(a, b)
    assert result["fields"]["close"]["equal_within_1e_6"] == 12
    assert not result["declared_price_comparable"] and not result["selection_performed"]


def test_invalid_prices_cannot_be_declared_comparable():
    a, b = panel(), panel("candidate")
    b.rows[0]["quality_flags"] = ["invalid_ohlc"]
    assert not diagnose_panel(a, b)["declared_price_comparable"]


def test_observed_panel_does_not_change_production_route():
    result = assess_feature_input(panel())
    assert result["ready"] and not result["production_route_changed"]


def test_sample_evidence_is_blocked_even_if_batch_metadata_was_lost():
    batch = panel()
    batch.rows[0]["quality_flags"] = ["sample_evidence_only"]
    assert not assess_feature_input(batch)["ready"]
    batch.rows[0]["quality_flags"] = []
    batch.evidence_only = True
    assert not assess_feature_input(batch)["ready"]


def test_unknown_units_snapshots_empty_and_truncation_are_blocked():
    batch = panel()
    batch.rows[0]["quality_flags"] = [
        "unknown_units",
        "snapshot_not_confirmed_daily_bar",
    ]
    batch.rows[0]["volume_shares"] = None
    batch.truncated = True
    result = assess_feature_input(batch)
    assert (
        not result["ready"] and "canonical_quantity_unavailable" in result["blockers"]
    )
    assert not assess_feature_input(DailyBatch("empty", "v1"))["ready"]
