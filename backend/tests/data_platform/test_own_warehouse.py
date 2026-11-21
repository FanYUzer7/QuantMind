import io
from pathlib import Path
import zipfile

import pytest

from backend.scripts.build_own_warehouse import Builder
from backend.services.engine.data_platform.own_data.warehouse import Warehouse


HEADER = "日期,代码,名称,开盘价,最高价,最低价,收盘价,成交量（股）,成交额（元）,流通股本（股）,是否ST\n"
GOOD = "2025-01-02,000001,平安银行,10,11,9,10,100,1000,10000,否\n"


def source_tree(tmp_path):
    source = tmp_path / "source"
    folder = source / "A股分钟线+日频量价/股票更新至4.30/每只股票一个文件"
    folder.mkdir(parents=True)
    for name in ["不复权", "前复权", "后复权"]:
        with zipfile.ZipFile(folder / (name + ".zip"), "w") as z:
            z.writestr(name + "/000001.csv", HEADER + GOOD)
    return source


def test_import_preserves_source_and_explicit_units(tmp_path):
    source = source_tree(tmp_path)
    root = tmp_path / "warehouse"
    b = Builder(root, source, tmp_path / "eastmoney")
    b.daily()
    b.finish()
    w = Warehouse(root)
    frame = w.daily(["000001.SZ"], "2025-01-02", "2025-01-02")
    assert frame.iloc[0]["代码"] == "000001"
    assert frame.iloc[0]["amount"] == 1000
    assert frame.iloc[0]["volume"] == 100
    assert frame.iloc[0]["is_st"] == 0
    assert frame.iloc[0]["quality_flags"] == ""
    assert frame.iloc[0]["price_adjustment"] == "raw"
    assert (
        len(w.daily(["000001.SZ"], "2025-01-02", "2025-01-02", adjustment="qfq")) == 1
    )
    w.close()
    # Resuming verifies immutable objects and does not double rows.
    b = Builder(root, source, tmp_path / "eastmoney")
    b.daily()
    b.finish()
    w = Warehouse(root)
    assert len(w.daily(["000001.SZ"], "2025-01-02", "2025-01-02")) == 1
    w.close()


def test_conflicts_and_invalid_rows_are_retained(tmp_path):
    source = source_tree(tmp_path)
    archive = source / "A股分钟线+日频量价/股票更新至4.30/每只股票一个文件/不复权.zip"
    bad = GOOD.replace(",10,11,9,10,", ",10,8,9,10,").replace(",否", ",未知")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000001.csv", HEADER + GOOD + bad)
    b = Builder(tmp_path / "db", source, tmp_path / "em")
    b.daily()
    assert b.con.execute("SELECT copies FROM daily_conflicts").fetchone()[0] == 2
    assert (
        b.con.execute(
            "SELECT count(*) FROM daily_bars WHERE quality_flags LIKE '%invalid_ohlc%' AND is_st IS NULL"
        ).fetchone()[0]
        == 1
    )
    b.finish()
    w = Warehouse(tmp_path / "db")
    with pytest.raises(ValueError, match="Conflicting"):
        w.daily(["000001.SZ"], "2025-01-02", "2025-01-02")
    w.close()


def test_source_revision_retained_and_updates_have_explicit_priority(tmp_path):
    source = source_tree(tmp_path)
    update = source / "A股分钟线+日频量价/每天一个文件/不复权/2025"
    update.mkdir(parents=True)
    (update / "2025-01-02.csv").write_text(HEADER + GOOD.replace(",1000,", ",1100,"))
    b = Builder(tmp_path / "db", source, tmp_path / "em")
    b.daily()
    assert (
        b.con.execute(
            "SELECT count(*) FROM daily_all_versions WHERE price_adjustment='raw'"
        ).fetchone()[0]
        == 2
    )
    assert (
        b.con.execute(
            "SELECT amount FROM daily_bars WHERE price_adjustment='raw'"
        ).fetchone()[0]
        == 1100
    )
    b.finish()


def test_eastmoney_preserves_source_date_and_failed_groups(tmp_path):
    import json
    import duckdb

    source = source_tree(tmp_path)
    em = tmp_path / "em"
    em.mkdir()
    x = {
        "format": "quant-eastmoney-demo",
        "saved_at": "2026-11-09T13:00:00Z",
        "groups": [
            {
                "job_id": "margin_stock",
                "status": "complete",
                "collected_at": "2026-11-09T12:59:00Z",
                "rows": [
                    {"code": "000001", "source_date": "2026-11-06", "amount": None}
                ],
            },
            {"job_id": "empty", "status": "failed", "error": "timeout", "rows": []},
        ],
    }
    (em / "snapshot.json").write_text(json.dumps(x))
    b = Builder(tmp_path / "db", source, em)
    b.eastmoney_import()
    b.finish()
    con = duckdb.connect(str(tmp_path / "db/warehouse.duckdb"), read_only=True)
    row = con.execute(
        "SELECT code,_qm_source_date,_qm_available_at,amount FROM eastmoney_margin_stock"
    ).fetchone()
    assert row == ("000001", "2026-11-06", "2026-11-09T12:59:00Z", None)
    assert (
        con.execute(
            "SELECT count(*) FROM ingest_events WHERE status='empty_group'"
        ).fetchone()[0]
        == 1
    )
    con.close()


def test_owned_hub_and_simulation_keep_canonical_units_and_historical_st(tmp_path):
    from datetime import date
    from backend.services.engine.data_platform.own_data.hub import OwnWarehouseHub
    from backend.services.simulation.services.local_market_data import LocalMarketData

    source = source_tree(tmp_path)
    b = Builder(tmp_path / "db", source, tmp_path / "em")
    b.daily()
    b.finish()
    hub = OwnWarehouseHub(str(tmp_path / "db"))
    result = hub.fetch_daily_kline(
        "SZ000001", date(2025, 1, 2), date(2025, 1, 2), adjust="raw"
    )
    assert result.iloc[0].symbol == "000001.SZ"
    assert result.iloc[0].amount == 1000
    market = LocalMarketData(hub=hub, market="CN")
    bar = market.get_bar("000001.SZ", date(2025, 1, 2))
    assert bar.amount == 1000
    assert bar.volume == 100
    assert bar.vwap == 10
    assert bar.is_st is False


def test_qlib_export_recovers_raw_prices_and_shares(tmp_path):
    import numpy as np
    import pandas as pd
    from backend.scripts.export_own_warehouse import write_qlib

    days = ["2025-01-02", "2025-01-03"]
    raw = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 2,
            "trade_date": days,
            "open": [10, 11],
            "high": [11, 12],
            "low": [9, 10],
            "close": [10, 11],
            "volume": [100, 200],
            "amount": [1000, 2200],
            "is_st": [0, 1],
        }
    )
    adj = raw.copy()
    for k in ["open", "high", "low", "close"]:
        adj[k] *= 0.5
    write_qlib(tmp_path / "qlib", raw, adj, ["000001.SZ"], days)
    folder = tmp_path / "qlib/features/sz000001"
    values = {
        k: np.fromfile(folder / (k + ".day.bin"), dtype="<f4")
        for k in ["close", "factor", "volume"]
    }
    assert values["close"][0] == 0  # starting calendar index
    assert np.allclose(values["close"][1:] / values["factor"][1:], raw.close)
    assert np.allclose(values["volume"][1:] * values["factor"][1:], raw.volume)
    change = np.fromfile(folder / "change.day.bin", dtype="<f4")
    assert np.isnan(change[1])
    assert change[2] == pytest.approx(0.1)


def downloader_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "quantdb_warehouse_downloader", "/quant/scripts/quantdb-warehouse-download.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_over_budget_never_opens_business_request(tmp_path, monkeypatch):
    import json

    module = downloader_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "download-plan-complement.json").write_text(
        json.dumps({"total_bytes": 10_000_000_001, "entries": []})
    )
    monkeypatch.setattr(module, "get_secret", lambda _: "dummy-not-a-real-key")
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download", "download", "--run", str(run), "--root", str(tmp_path / "data")],
    )
    monkeypatch.setattr(
        module, "transfer", lambda *args: pytest.fail("Billable request must not start")
    )
    with pytest.raises(SystemExit, match="exceeds 10 GB"):
        module.main()


def test_download_redirect_never_sends_key_to_cdn(tmp_path):
    import hashlib

    module = downloader_module()
    calls = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"Location": "https://cdn.example.invalid/object"}

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield b"abc"

    class Session:
        def __init__(self, status):
            self.status = status

        def get(self, url, **kwargs):
            calls.append((self.status, url, kwargs))
            return Response(self.status)

    module.LOCAL.api = Session(302)
    module.LOCAL.cdn = Session(200)
    entry = (
        "balance",
        "3",
        {
            "key": "3_financial_data/balance/test.parquet",
            "size": 3,
            "etag": hashlib.md5(b"abc").hexdigest(),
        },
    )
    _, result, error = module.transfer(entry, tmp_path, "dummy-not-a-real-key")
    assert error is None and result
    assert calls[0][2]["headers"]["X-API-Key"] == "dummy-not-a-real-key"
    assert "headers" not in calls[1][2]


def test_financial_filter_uses_announcement_instead_of_report_period(tmp_path):
    import pandas as pd
    from backend.services.engine.data_platform.own_data.hub import OwnWarehouseHub

    source = source_tree(tmp_path)
    path = tmp_path / "balance.parquet"
    pd.DataFrame(
        {
            "Symbol": ["000001.SZ"] * 3,
            "m_timetag": ["20181231"] * 3,
            "m_anntime": ["20251231", "20260101", None],
            "tot_assets": [1.0, 2.0, 3.0],
        }
    ).to_parquet(path, index=False)
    b = Builder(tmp_path / "db", source, tmp_path / "em")
    b.publish("quantdb_balance", [path], "quantdb", "test-only")
    b.finish()
    hub = OwnWarehouseHub(str(tmp_path / "db"))
    result = hub.fetch_financial("000001.SZ", "balance", "2025-01-01", "2025-12-31")
    assert result.tot_assets.tolist() == [1.0]


def test_nested_plugin_archive_keeps_all_partitions_and_relative_paths(tmp_path):
    import json

    source = source_tree(tmp_path)
    em = tmp_path / "em"
    archive = em / "qm-eastmoney-archive"
    archive.mkdir(parents=True)
    for i in range(2):
        value = {
            "format": "quant-eastmoney-demo",
            "saved_at": "2026-11-09T13:00:00Z",
            "groups": [
                {
                    "job_id": "finance_balance",
                    "status": "complete",
                    "rows": [
                        {"SECUCODE": f"00000{i + 1}.SZ", "REPORT_DATE": "2026-09-30"}
                    ],
                }
            ],
        }
        (archive / f"part-{i}.qm.json").write_text(json.dumps(value))
    (archive / "manifest.json").write_text(
        json.dumps({"format": "quant-eastmoney-archive", "files": []})
    )
    b = Builder(tmp_path / "db", source, em)
    b.eastmoney_import()
    count = b.con.execute("SELECT count(*) FROM eastmoney_finance_balance").fetchone()[
        0
    ]
    paths = b.con.execute(
        "SELECT _qm_source_file FROM eastmoney_finance_balance ORDER BY 1"
    ).fetchall()
    assert count == 2
    assert paths == [
        ("qm-eastmoney-archive/part-0.qm.json",),
        ("qm-eastmoney-archive/part-1.qm.json",),
    ]
    b.finish()
