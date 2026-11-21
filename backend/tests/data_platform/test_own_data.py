"""Meaningful contract regressions; all files are isolated test fixtures."""

import csv
import json
from dataclasses import replace

import pandas as pd
import pytest

from backend.services.engine.data_platform.own_data import (
    DailyQuery,
    SourceSpec,
    DataContractError,
    TabularDailyReader,
    QuantDBPreviewReader,
    EastmoneySavedReader,
    OwnDataRepository,
    compare_daily,
)
from backend.services.engine.data_platform.own_data.contracts import cn_symbol


def spec(**kwargs):
    return SourceSpec(
        "test",
        "v1",
        "raw",
        "lots",
        "thousand_yuan",
        unit_evidence="explicit test headers",
        **kwargs,
    )


def query(**kwargs):
    return DailyQuery(
        ("600519.SH", "000001.SH", "000001.SZ"), "2026-10-01", "2026-10-31", **kwargs
    )


def row(symbol="600519.SH", day="20261030", **kwargs):
    return {
        "symbol": symbol,
        "trade_date": day,
        "open": 10,
        "high": 12,
        "low": 9,
        "close": 11,
        "volume": 2,
        "amount": 3,
        **kwargs,
    }


def csv_reader(tmp_path, rows, source=None, **kwargs):
    p = tmp_path / "bars.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return TabularDailyReader(source or spec(), [p], **kwargs)


def test_explicit_exchange_separates_index_and_stock():
    assert cn_symbol("000001", exchange="SH") == "000001.SH"
    assert cn_symbol("000001", exchange="SZ") == "000001.SZ"
    assert cn_symbol("sh600519") == "600519.SH"
    with pytest.raises(DataContractError):
        cn_symbol("000001")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbols": ("000001",)},
        {"symbols": ()},
        {"start": "2020-01-01"},
        {"max_rows": 5001},
        {"adjustment": "auto"},
        {"end": "2026-09-01"},
    ],
)
def test_bad_query_rejected(kwargs):
    options = {"symbols": ("600519.SH",), "start": "2026-10-01", "end": "2026-10-31"}
    options.update(kwargs)
    with pytest.raises((DataContractError, ValueError)):
        DailyQuery(**options)


def test_units_are_converted_without_changing_raw_values(tmp_path):
    reader = csv_reader(tmp_path, [row()])
    before = reader.files[0].read_bytes()
    batch = reader.read_daily(query())
    bar = batch.rows[0]
    assert (bar["volume_shares"], bar["amount_cny"]) == (200, 3000)
    assert (bar["volume_raw"], bar["amount_raw"]) == (2, 3)
    assert reader.files[0].read_bytes() == before
    assert not batch.requested_window_complete
    assert batch.observed_calendar()["kind"] == "observed_not_exchange_calendar"


def test_unknown_units_are_not_assumed_or_filled(tmp_path):
    source = SourceSpec("unknown", "v1")
    bar = (
        csv_reader(tmp_path, [row(amount="", volume="-")], source)
        .read_daily(query())
        .rows[0]
    )
    assert bar["volume_shares"] is None and bar["amount_cny"] is None
    assert "unknown_units" in bar["quality_flags"]
    assert "unknown_adjustment" in bar["quality_flags"]


def test_units_require_evidence():
    with pytest.raises(DataContractError):
        SourceSpec("x", "v1", volume_unit="lots")


def test_duplicates_and_invalid_prices_are_retained_for_audit(tmp_path):
    batch = csv_reader(tmp_path, [row(), row(close=99)]).read_daily(query())
    assert len(batch.rows) == 2
    assert "duplicate_key" in batch.rows[1]["quality_flags"]
    assert "invalid_ohlc" in batch.rows[1]["quality_flags"]
    assert batch.issues[0]["code"] == "duplicate_key"
    assert not compare_daily(batch, batch)["pairs"][0]["comparable_prices"]


def test_cap_and_scan_failure_are_explicit(tmp_path):
    reader = csv_reader(tmp_path, [row(), row("000001.SZ")])
    batch = reader.read_daily(query(max_rows=1))
    assert len(batch.rows) == 1 and batch.truncated
    reader = TabularDailyReader(spec(), reader.files, scan_limit=1)
    with pytest.raises(DataContractError, match="scan cap"):
        reader.read_daily(query())


def test_parquet_pushdown_and_prefix_boundary(tmp_path):
    p = tmp_path / "bars.parquet"
    pd.DataFrame([row("SH600519"), row("000001.SZ"), row(day="20250101")]).to_parquet(p)
    batch = TabularDailyReader(spec(), [p]).read_daily(query())
    assert {r["symbol"] for r in batch.rows} == {"600519.SH", "000001.SZ"}
    assert len(batch.rows) == 2


def test_comparison_converts_quantity_but_never_selects_a_source(tmp_path):
    left = csv_reader(tmp_path, [row()]).read_daily(query())
    right = csv_reader(
        tmp_path,
        [row(volume=200, amount=3000)],
        replace(spec(), source_id="other", volume_unit="shares", amount_unit="yuan"),
    ).read_daily(query())
    report = compare_daily(left, right)
    assert report["overlap_keys"] == 1
    assert report["pairs"][0]["right_minus_left"]["volume_shares"] == 0
    assert not report["selection_performed"]
    incompatible = csv_reader(
        tmp_path, [row()], replace(spec(), adjustment="qfq")
    ).read_daily(query())
    assert not compare_daily(left, incompatible)["pairs"][0]["comparable_prices"]


def saved_reader(tmp_path, group, *, snapshot=False):
    p = tmp_path / "saved.qm.json"
    p.write_text(
        json.dumps(
            {
                "format": "quant-eastmoney-demo",
                "schema_version": 1,
                "saved_at": "2026-11-01T00:00:00Z",
                "groups": [group],
            }
        )
    )
    return EastmoneySavedReader(
        replace(spec(), snapshot=snapshot), [p], [group["job_id"]]
    )


def test_saved_raw_csv_decodes_offline(tmp_path):
    group = {
        "job_id": "daily_test",
        "dataset": "daily",
        "secid": "1.600519",
        "price_adjustment": "unadjusted",
        "data_kind": "historical_kline",
        "raw_columns": [
            "time",
            "open",
            "close",
            "high",
            "low",
            "volume_raw",
            "amount_raw",
        ],
        "rows": [{"record": "2026-10-30,10,11,12,9,2,3"}],
    }
    batch = saved_reader(tmp_path, group).read_daily(query())
    assert batch.rows[0]["close"] == 11
    assert batch.rows[0]["available_at"] is None
    assert batch.rows[0]["collected_at"] == "2026-11-01T00:00:00Z"


def test_snapshot_is_not_silently_accepted_as_daily(tmp_path):
    group = {
        "job_id": "a_market",
        "price_adjustment": "unadjusted",
        "scope": "CN",
        "data_kind": "quote_snapshot_with_day_ohlc_and_optional_flow",
        "rows": [
            {
                "secid": "1.600519",
                "source_date": "2026-10-30",
                "last": 11,
                "open": 10,
                "high": 12,
                "low": 9,
                "volume_raw": 2,
                "amount_raw": 3,
            }
        ],
    }
    reader = saved_reader(tmp_path, group, snapshot=True)
    with pytest.raises(DataContractError, match="allow_snapshots"):
        reader.read_daily(query())
    batch = reader.read_daily(query(allow_snapshots=True))
    assert "snapshot_not_confirmed_daily_bar" in batch.rows[0]["quality_flags"]


def test_minute_data_and_mismatched_adjustment_are_rejected(tmp_path):
    base = {
        "job_id": "min5",
        "dataset": "min5",
        "data_kind": "historical_kline",
        "price_adjustment": "unadjusted",
        "rows": [],
    }
    with pytest.raises(DataContractError, match="Intraday"):
        saved_reader(tmp_path, base).read_daily(query())
    with pytest.raises(DataContractError, match="adjustment"):
        saved_reader(tmp_path, {**base, "price_adjustment": "qfq"}).read_daily(query())


def test_preview_is_explicit_sample_and_never_downloads(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden")
    )
    p = tmp_path / "preview.json"
    p.write_text(
        json.dumps(
            [
                {
                    "dataset": "daily_unadjusted",
                    "status": 200,
                    "body": {"data": [{**row(), "time": "2026-10-30"}]},
                }
            ]
        )
    )
    with pytest.raises(DataContractError):
        QuantDBPreviewReader(spec(), p)
    r = QuantDBPreviewReader(replace(spec(), evidence_only=True), p)
    batch = r.read_daily(query())
    assert batch.evidence_only and len(batch.rows) == 1


def test_repository_needs_explicit_source_and_disallows_duplicates(tmp_path):
    reader = csv_reader(tmp_path, [row()])
    with pytest.raises(DataContractError):
        OwnDataRepository([reader, reader])
    repo = OwnDataRepository([reader])
    assert repo.list_sources()[0]["source_id"] == "test"
    with pytest.raises(KeyError):
        repo.read_daily("auto", query())


def test_explicit_bare_symbol_mapping_and_ambiguous_rejection(tmp_path):
    reader = csv_reader(tmp_path, [row("000001")])
    batch = reader.read_daily(query())
    assert not batch.rows and batch.issues[0]["code"] == "unparseable_record"
    reader = TabularDailyReader(
        spec(), reader.files, symbol_map={"000001": "000001.SZ"}
    )
    assert reader.read_daily(query()).rows[0]["symbol"] == "000001.SZ"


def test_intraday_timestamp_is_quarantined(tmp_path):
    batch = csv_reader(tmp_path, [row(day="2026-10-30 09:31:00")]).read_daily(query())
    assert not batch.rows
    assert "Intraday" in batch.issues[0]["message"]


def test_no_overlap_is_insufficient_evidence(tmp_path):
    left = csv_reader(tmp_path, [row()]).read_daily(query())
    right = csv_reader(tmp_path, [row("000001.SH")]).read_daily(query())
    report = compare_daily(left, right)
    assert report["conclusion"] == "insufficient_overlap"
    assert not report["selection_performed"]


def test_minute_request_metadata_cannot_masquerade_as_daily(tmp_path):
    group = {
        "job_id": "bad",
        "dataset": "daily",
        "data_kind": "historical_kline",
        "price_adjustment": "unadjusted",
        "rows": [],
        "request_params": {"klt": 5},
    }
    with pytest.raises(DataContractError, match="Intraday"):
        saved_reader(tmp_path, group).read_daily(query())
