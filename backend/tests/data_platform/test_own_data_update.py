import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from backend.scripts.build_own_warehouse import Builder
from backend.scripts.update_own_data import enforce_budget
from backend.services.engine.data_platform.own_data.publication import (
    current_release,
    publish,
)
from backend.services.engine.data_platform.own_data.sync import (
    SyncState,
    atomic_json,
    fingerprint,
    safe_relative,
)
from backend.services.engine.data_platform.own_data.warehouse import Warehouse, sha256


def parquet(tmp_path, name, frame):
    p = tmp_path / (name + ".parquet")
    frame.to_parquet(p, index=False)
    return p


def item(path, key="6_ml_datasets/l1_factors/dt=20261110/data.parquet"):
    return {
        "key": key,
        "relative_path": key,
        "size": path.stat().st_size,
        "etag": hashlib.md5(path.read_bytes()).hexdigest(),
        "trade_date": "2026-11-10",
    }


def base_catalog(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "quantdb-compatible").mkdir()
    source = tmp_path / "sources"
    source.mkdir()
    b = Builder(base, source, source)
    dates = pd.to_datetime(["2026-11-09", "2026-11-10"])
    prices = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 2,
            "time": dates,
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [100.0, 100.0],
            "amount": [0.1, 0.1],
        }
    )
    for dataset, scale in [
        ("daily_unadjusted", 1),
        ("daily_forward", 0.5),
        ("daily_backward", 2),
        ("etf_kline", 0.1),
    ]:
        frame = prices.copy()
        for col in ["open", "high", "low", "close"]:
            frame[col] *= scale
        if dataset == "etf_kline":
            frame["symbol"] = "510300.SH"
        p = parquet(tmp_path, dataset, frame)
        b.publish("quantdb_" + dataset, [p], "quantdb", "fixture")
    detail = pd.DataFrame(
        {
            "Symbol": ["000001.SZ"],
            "Name": ["fixture"],
            "HqDate": ["2026-11-10"],
            "IsSTGP": ["0"],
        }
    )
    b.publish(
        "quantdb_instrument_detail",
        [parquet(tmp_path, "detail", detail)],
        "quantdb",
        "fixture",
    )
    value = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 2,
            "time": dates,
            "circulating_capital": [1000.0, 1000.0],
        }
    )
    b.publish(
        "quantdb_valuation",
        [parquet(tmp_path, "valuation", value)],
        "quantdb",
        "fixture",
    )
    b.con.execute(
        "CREATE VIEW daily_bars AS SELECT symbol,time::DATE AS trade_date,open,high,low,close,volume,amount*10000 AS amount,'raw' AS price_adjustment,'fixture' AS source_id,1::TINYINT AS is_st,1000.0 AS float_shares FROM quantdb_daily_unadjusted WHERE time<DATE '2026-11-10'"
    )
    b.finish()
    return base, source


def factor_plan(tmp_path, state, duplicate=False):
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": pd.to_datetime(["2026-11-10"]),
            "mom_ret_1d": [0.01],
        }
    )
    if duplicate:
        frame = pd.concat([frame, frame], ignore_index=True)
    p = parquet(tmp_path, "factors-duplicate" if duplicate else "factors", frame)
    obj = item(p)
    state.complete("l1_factors", obj, p, sha256(p))
    return {
        "entries": [("l1_factors", "6", obj)],
        "manifests": {"l1_factors": {"files": [obj]}},
        "categories": {"l1_factors": "6"},
    }


def test_plan_reuses_identical_versions_and_detects_equal_size_revisions(tmp_path):
    state = SyncState(tmp_path / "root")
    p = parquet(tmp_path, "one", pd.DataFrame({"symbol": ["000001.SZ"], "x": [1.0]}))
    obj = item(p)
    state.complete("l1_factors", obj, p, sha256(p))
    manifest = {"l1_factors": {"files": [obj]}}
    assert state.plan(manifest, {"l1_factors": "6"})["download_bytes"] == 0
    changed = {**obj, "etag": "0" * 32}
    plan = state.plan({"l1_factors": {"files": [changed]}}, {"l1_factors": "6"})
    assert plan["download_bytes"] == obj["size"] and len(plan["todo"]) == 1
    assert fingerprint(changed) != fingerprint(obj)
    state.close()


def test_missing_size_or_fingerprint_and_path_traversal_rejected():
    with pytest.raises(ValueError):
        fingerprint({"key": "x", "size": 10})
    for key in [
        "../evil.parquet",
        "/evil.parquet",
        "x/../../evil.parquet",
        "x\\evil.parquet",
    ]:
        with pytest.raises(ValueError):
            safe_relative({"key": key})


def test_budget_checked_before_billable_dispatch():
    config = {"monthly_budget_bytes": 50_000_000_000, "max_run_bytes": 45_000_000_000}
    enforce_budget(
        {"download_bytes": 36_998_296_468},
        {"traffic_limit": 53_687_091_200, "used_traffic": 0},
        config,
    )
    with pytest.raises(ValueError):
        enforce_budget(
            {"download_bytes": 10}, {"traffic_limit": 50, "used_traffic": 45}, config
        )


def test_payload_proof_recovers_after_crash_without_request(tmp_path):
    state = SyncState(tmp_path / "root")
    p = parquet(tmp_path, "payload", pd.DataFrame({"x": [1]}))
    obj = item(p)
    target = state.target("l1_factors", obj)
    target.parent.mkdir(parents=True)
    target.write_bytes(p.read_bytes())
    atomic_json(
        target.with_suffix(".proof.json"),
        {"fingerprint": fingerprint(obj), "size": obj["size"], "sha256": sha256(p)},
    )
    assert state.recover("l1_factors", obj)
    assert (
        state.plan({"l1_factors": {"files": [obj]}}, {"l1_factors": "6"})["todo"] == []
    )
    state.close()


def test_equal_size_local_corruption_never_counts_as_cached(tmp_path):
    state = SyncState(tmp_path / "root")
    p = parquet(tmp_path, "payload", pd.DataFrame({"x": [1]}))
    obj = item(p)
    state.complete("l1_factors", obj, p, sha256(p))
    body = bytearray(p.read_bytes())
    body[10] ^= 1
    p.write_bytes(body)
    with pytest.raises(ValueError, match="checksum"):
        state.plan({"l1_factors": {"files": [obj]}}, {"l1_factors": "6"})
    state.close()


def test_cdn_disconnect_retries_same_ticket_without_another_paid_get(
    tmp_path, monkeypatch
):
    import requests
    from backend.services.engine.data_platform.own_data import sync

    p = parquet(tmp_path, "wire", pd.DataFrame({"symbol": ["000001.SZ"], "x": [1.0]}))
    obj = item(p)
    body = p.read_bytes()
    calls = []

    class Response:
        def __init__(self, status=200, fail=False):
            self.status_code = status
            self.fail = fail
            self.headers = {"Location": "https://cdn.example.invalid/ticket"}

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            if self.fail:
                yield body[:10]
                raise requests.exceptions.ChunkedEncodingError("fixture")
            yield body

    class API:
        def get(self, url, **kwargs):
            calls.append(("api", kwargs))
            return Response(302)

    class CDN:
        count = 0

        def get(self, url, **kwargs):
            self.count += 1
            calls.append(("cdn", kwargs))
            return Response(fail=self.count == 1)

    monkeypatch.setattr(sync, "sessions", lambda: (API(), cdn))
    monkeypatch.setattr(sync.time, "sleep", lambda _: None)
    cdn = CDN()
    target = tmp_path / "stored.parquet"
    proof = sync.Remote("dummy-not-a-real-key").download(
        ("l1_factors", "6", obj), target
    )
    assert proof["sha256"] == sha256(p)
    assert [kind for kind, _ in calls] == ["api", "cdn", "cdn"]
    assert all("headers" not in options for kind, options in calls if kind == "cdn")


def test_failed_download_stops_queue_and_requires_explicit_retry(tmp_path):
    state = SyncState(tmp_path / "root")
    p = parquet(tmp_path, "payload", pd.DataFrame({"x": [1]}))
    obj = item(p)
    manifest = {
        "l1_factors": {
            "files": [obj, {**obj, "key": obj["key"].replace("20261110", "20261109")}]
        }
    }
    plan = state.plan(manifest, {"l1_factors": "6"})

    class Failed:
        calls = 0

        def download(self, *args):
            self.calls += 1
            raise RuntimeError("fixture failure")

    remote = Failed()
    with pytest.raises(RuntimeError):
        state.transfer(plan, remote, 1)
    assert remote.calls == 1
    with pytest.raises(RuntimeError):
        state.plan(manifest, {"l1_factors": "6"})
    assert (
        len(state.plan(manifest, {"l1_factors": "6"}, retry_failed=True)["todo"]) == 2
    )
    assert state.db.execute("SELECT status FROM attempts").fetchone()[0] == "failed"
    state.close()


def test_publish_pins_one_release_and_keeps_older_reader_valid(tmp_path):
    base, em = base_catalog(tmp_path)
    root = tmp_path / "root"
    state = SyncState(root)
    plan = factor_plan(tmp_path, state)
    run = root / "runs/release1"
    run.mkdir(parents=True)
    publish(root, base, run, plan, state, em)
    old = Warehouse(root)
    raw = old.daily(["000001.SZ"], "2026-11-09", "2026-11-10", source="integrated")
    assert raw.close.tolist() == [10.0, 11.0] and raw.amount.tolist() == [
        1000.0,
        1000.0,
    ]
    assert raw.is_st.tolist() == [1, 0]
    assert old.daily(
        ["000001.SZ"], "2026-11-10", "2026-11-10", adjustment="qfq", source="integrated"
    ).close.tolist() == [5.5]
    assert old.daily(
        ["510300.SH"], "2026-11-10", "2026-11-10", source="integrated"
    ).asset_type.tolist() == ["etf"]
    frame = old.factors(
        ["000001.SZ"], "2026-11-10", "2026-11-10", ["mom_ret_1d"], source="l1_factors"
    )
    assert frame.mom_ret_1d.tolist() == [0.01]
    run2 = root / "runs/release2"
    run2.mkdir()
    publish(root, base, run2, plan, state, em)
    assert old.release["release_id"] == "release1"
    assert current_release(root)["release_id"] == "release2"
    assert (
        old.daily(
            ["000001.SZ"], "2026-11-10", "2026-11-10", source="integrated"
        ).close.iloc[0]
        == 11
    )
    old.close()
    state.close()


def test_bad_partition_does_not_replace_current_pointer(tmp_path):
    base, em = base_catalog(tmp_path)
    root = tmp_path / "root"
    state = SyncState(root)
    plan = factor_plan(tmp_path, state)
    run = root / "runs/good"
    run.mkdir(parents=True)
    publish(root, base, run, plan, state, em)
    before = (root / "CURRENT.json").read_bytes()
    broken = factor_plan(tmp_path, state, duplicate=True)
    run2 = root / "runs/bad"
    run2.mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        publish(root, base, run2, broken, state, em)
    assert (root / "CURRENT.json").read_bytes() == before
    state.close()


def test_eastmoney_incremental_keeps_existing_rows_when_source_removed(tmp_path):
    root = tmp_path / "root"
    em = tmp_path / "em"
    em.mkdir()

    def save(name, symbol):
        (em / name).write_text(
            json.dumps(
                {
                    "format": "quant-eastmoney-demo",
                    "saved_at": "2026-11-10T13:00:00Z",
                    "groups": [
                        {
                            "job_id": "a_market",
                            "rows": [{"code": symbol, "source_date": "2026-11-10"}],
                        }
                    ],
                }
            )
        )

    save("first.json", "000001")
    b = Builder(root, tmp_path / "unused", em)
    b.eastmoney_import(incremental=True)
    b.finish()
    (em / "first.json").unlink()
    save("second.json", "000002")
    b = Builder(root, tmp_path / "unused", em)
    b.eastmoney_import(incremental=True)
    assert b.con.execute("SELECT count(*) FROM eastmoney_a_market").fetchone()[0] == 2
    b.eastmoney_import(incremental=True)
    assert b.con.execute("SELECT count(*) FROM eastmoney_a_market").fetchone()[0] == 2
    b.finish()


def test_legacy_publication_rejects_persistent_qlib(tmp_path):
    with pytest.raises(ValueError, match="temporary_provider"):
        publish(tmp_path, tmp_path, tmp_path, {}, None, tmp_path, build_qlib=True)


def test_daily_refuses_legacy_warehouse_before_network_or_writes(tmp_path, monkeypatch):
    import sys
    from backend.scripts import update_own_data

    root = tmp_path / "database"
    root.mkdir()
    atomic_json(root / "CURRENT.json", {"source_list": "sources.json"})
    config = tmp_path / "config.json"
    atomic_json(config, {"warehouse": "database"})
    monkeypatch.setattr(sys, "argv", ["update", "--workspace", str(tmp_path), "--config", str(config)])
    with pytest.raises(SystemExit, match="Current warehouse required"):
        update_own_data.main()
    assert not (root / "_ops").exists()
