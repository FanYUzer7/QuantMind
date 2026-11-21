import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.scripts.migrate_own_current import migrate
from backend.services.engine.data_platform.own_data.current import access, sources, update_current
from backend.services.engine.data_platform.own_data.publication import publish
from backend.services.engine.data_platform.own_data.sync import SyncState, atomic_json
from backend.services.engine.data_platform.own_data.temporary_qlib import cleanup_abandoned, temporary_provider
from backend.services.engine.data_platform.own_data.warehouse import Warehouse, sha256
from backend.tests.data_platform.test_own_data_update import base_catalog, factor_plan, item, parquet


@pytest.fixture
def current(tmp_path):
    base, em = base_catalog(tmp_path)
    old = tmp_path / "old"
    state = SyncState(old)
    plan = factor_plan(tmp_path, state)
    run = old / "runs/initial"
    run.mkdir(parents=True)
    publish(old, base, run, plan, state, em)
    state.close()
    root = tmp_path / "current"
    migrate(old, root)
    return root, em, old


def test_migration_has_no_old_view_references_and_reads_data(current):
    root, _, old = current
    with Warehouse(root) as w:
        assert w.daily(["000001.SZ"], "2026-11-10", "2026-11-10").close.tolist() == [11]
        assert sources(w.connection)["entries"]
        for (sql,) in w.connection.execute("SELECT sql FROM duckdb_views() WHERE internal=false").fetchall():
            assert str(old) not in sql
        paths = [Path(p) for (j,) in w.connection.execute("SELECT files_json FROM catalog_tables").fetchall() for p in json.loads(j)]
        assert all(p.is_relative_to(root) for p in paths)
    assert not (root / "releases").exists()


def test_current_revision_removal_empty_and_repeat(current, tmp_path):
    root, em, _ = current
    state = SyncState(root)
    p = parquet(tmp_path, "revised", pd.DataFrame({"symbol": ["000001.SZ"], "date": pd.to_datetime(["2026-11-10"]), "mom_ret_1d": [0.25]}))
    obj = item(p)
    state.complete("l1_factors", obj, p, sha256(p))
    plan = {"entries": [("l1_factors", "6", obj)], "manifests": {"l1_factors": {"files": [obj]}}}
    report = update_current(root, root / "_ops/runs/test", plan, state, em)
    assert report["revised"] == 1 and not report["qlib_generated"]
    with Warehouse(root) as w:
        assert w.factors(["000001.SZ"], "2026-11-10", "2026-11-10", ["mom_ret_1d"], source="l1_factors").mom_ret_1d.tolist() == [0.25]
    assert update_current(root, root / "_ops/runs/repeat", plan, state, em)["revised"] == 0
    empty = {"entries": [], "manifests": {"l1_factors": {"files": []}}}
    assert update_current(root, root / "_ops/runs/empty", empty, state, em)["removed"] == 1
    with Warehouse(root) as w:
        assert w.connection.execute("SELECT count(*) FROM quantdb_l1_factors").fetchone()[0] == 0
    assert not (root / "releases").exists()
    assert not (root / "qlib").exists()
    state.close()


def test_readers_exclude_update_and_failure_rolls_back(current, tmp_path):
    root, em, _ = current
    state = SyncState(root)
    with Warehouse(root):
        with pytest.raises(RuntimeError, match="busy"):
            with access(root, write=True):
                pass
    broken = factor_plan(tmp_path, state, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        update_current(root, root / "_ops/runs/broken", broken, state, em)
    with Warehouse(root) as w:
        assert w.factors(["000001.SZ"], "2026-11-10", "2026-11-10", ["mom_ret_1d"], source="l1_factors").mom_ret_1d.tolist() == [0.01]
    state.close()


def test_temporary_qlib_is_bounded_and_cleaned_even_on_failure(current):
    root, _, _ = current
    with Warehouse(root, read_only=False) as w:
        w.connection.execute("CREATE TABLE quantdb_trading_calendar AS SELECT DATE '2026-11-11' AS TradingDate,1 AS IsTradingDay")
    with pytest.raises(RuntimeError, match="task failed"):
        with temporary_provider(root, symbols=["000001.SZ"], start="2026-11-10", end="2026-11-10", history_days=0) as path:
            assert (path / "calendars/day.txt").read_text().splitlines() == ["2026-11-10", "2026-11-11"]
            assert [p.name for p in (path / "features").iterdir()] == ["sz000001"]
            values = np.fromfile(path / "features/sz000001/close.day.bin", dtype="<f4")
            assert values[1] == 22
            cleanup_abandoned(root)
            assert path.exists()
            with pytest.raises(RuntimeError, match="busy"):
                with access(root, write=True):
                    pass
            raise RuntimeError("task failed")
    assert not path.exists()
    dead = root / "_ops/tmp/qlib-task-abandoned"
    dead.mkdir()
    cleanup_abandoned(root)
    assert not dead.exists()


def test_factor_constructor_uses_catalog(current, monkeypatch):
    from backend.services.engine.data_platform.quantdb_factor_reader import QuantDBFactorReader
    root, _, _ = current
    monkeypatch.setenv("QM_OWN_WAREHOUSE_ROOT", str(root))
    reader = QuantDBFactorReader()
    assert type(reader).__name__ == "CurrentFactorReader"
    assert reader.available_dates("l1_factors") == ["2026-11-10"]
    data = reader.read_range("l1_factors", features=["mom_ret_1d"], start="2026-11-10", end="2026-11-10")
    assert data.close.tolist() == [22]
    assert data.mom_ret_1d.tolist() == [0.01]


def test_hub_daily_alias_uses_canonical_units(current):
    from backend.services.engine.data_platform.own_data.hub import OwnWarehouseHub
    root, _, _ = current
    data = OwnWarehouseHub(root).query("SELECT close,amount FROM qdb_daily_unadjusted WHERE dt=20261110")
    assert data.close.tolist() == [11]
    assert data.amount.tolist() == [1000]


def test_prediction_uses_owned_calendar_instead_of_image_calendar(current, monkeypatch):
    from backend.services.engine.inference.script_runner import InferenceScriptRunner
    root, _, _ = current
    with Warehouse(root, read_only=False) as w:
        w.connection.execute("CREATE TABLE quantdb_trading_calendar AS SELECT DATE '2026-11-11' AS TradingDate,1 AS IsTradingDay")
    monkeypatch.setenv("QM_OWN_WAREHOUSE_ROOT", str(root))
    assert InferenceScriptRunner._resolve_prediction_trade_date("2026-11-10") == "2026-11-11"
    with pytest.raises(RuntimeError, match="no next session"):
        InferenceScriptRunner._resolve_prediction_trade_date("2026-11-11")


def test_offline_daily_uses_current_manifest_and_has_no_plan_history(current, tmp_path, monkeypatch):
    import sys
    from backend.scripts.update_own_data import main
    root, em, _ = current
    config = tmp_path / "update.json"
    atomic_json(config, {"warehouse": str(root), "eastmoney_root": str(em), "datasets": {"l1_factors": "6"}, "monthly_budget_bytes": 100, "max_run_bytes": 100})
    monkeypatch.setattr(sys, "argv", ["update", "--config", str(config), "--workspace", str(tmp_path), "--offline", "--run-id", "offline-test"])
    main()
    report = root / "_ops/runs/offline-test/daily-report.json"
    assert json.loads(report.read_text())["status"] == "complete"
    assert not (report.parent / "plan.json").exists()
