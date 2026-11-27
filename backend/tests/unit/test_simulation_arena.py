"""
Offline unit tests for Champion-Challenger multi-account simulation arena.

Runs fully offline using SQLite (aiosqlite / sqlite3) + in-memory fake Redis,
requiring no Docker, PostgreSQL, or market data.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.services.engine.inference.gap_backfill import list_default_models
from backend.services.simulation.models import Base as SimBase
from backend.services.simulation.models.account import SimulationAccount
from backend.services.simulation.models.account_daily import SimulationAccountDaily
from backend.services.simulation.models.fill import SimulationFill
from backend.services.simulation.models.position_daily import SimulationPositionDaily
from backend.services.simulation.models.position_lot import SimulationPositionLot
from backend.services.simulation.routers.simulation_arena import router as arena_router
from backend.services.simulation.services.arena_service import SimulationArenaService
from backend.services.simulation.services.daily_snapshot_service import (
    SimulationDailySnapshotService,
)
from backend.services.simulation.services.projection_service import (
    SimulationProjectionService,
)
from backend.services.simulation.services.simulation_hosted_scheduler import (
    SimulationHostedScheduler,
    _lock_key,
    _task_id,
)
from backend.services.trade_shared.deps import AuthContext, get_auth_context, get_db, get_redis
from backend.services.trade_shared.simulation_manager import SimulationAccountManager
from backend.shared.simulation_account_keys import (
    account_key,
    active_strategy_key,
    build_default_account_id,
    is_default_account_id,
    parse_active_strategy_account_id,
    parse_active_strategy_key,
)


class FakeRedisBackend:
    """Minimal dict-backed Redis client implementing the commands used by simulation."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: Any, ex: int | None = None, nx: bool = False) -> bool:
        if nx and key in self._store:
            return False
        self._store[key] = value if isinstance(value, str) else str(value)
        return True

    def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                removed += 1
        return removed

    def scan_iter(self, match: str = "*", count: int | None = None):
        import fnmatch

        for k in list(self._store.keys()):
            if fnmatch.fnmatch(k, match):
                yield k


class FakeRedisClient:
    def __init__(self) -> None:
        self.client = FakeRedisBackend()


@pytest_asyncio.fixture
async def async_db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(
            SimBase.metadata.create_all,
            tables=[
                SimulationAccount.__table__,
                SimulationAccountDaily.__table__,
                SimulationPositionLot.__table__,
                SimulationPositionDaily.__table__,
                SimulationFill.__table__,
            ],
        )
        await conn.execute(
            text(
                """
                CREATE TABLE qm_model_inference_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    signals_count INTEGER,
                    coverage REAL,
                    ic REAL,
                    rank_ic REAL
                )
                """
            )
        )
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


def test_account_and_active_strategy_key_isolation():
    default_aid = build_default_account_id("default", "10000001")
    challenger_aid = "sim:default:10000001:challenger_abc123"

    assert default_aid == "sim:default:10000001"
    assert is_default_account_id(default_aid, tenant_id="default", user_id="10000001")
    assert not is_default_account_id(challenger_aid, tenant_id="default", user_id="10000001")

    # Default account preserves legacy Redis key format
    assert account_key("default", "10000001") == "simulation:account:default:10000001"
    assert account_key("default", "10000001", account_id=default_aid) == "simulation:account:default:10000001"
    assert active_strategy_key("default", "10000001") == "trade:active_strategy:default:10000001"
    assert active_strategy_key("default", "10000001", account_id=default_aid) == "trade:active_strategy:default:10000001"

    # Non-default challenger account uses isolated v2 account key + :acc:{aid} strategy key
    assert account_key("default", "10000001", account_id=challenger_aid) == f"simulation:account_v2:{challenger_aid}"
    assert (
        account_key("default", "10000001", market="HK", account_id=challenger_aid)
        == f"simulation:account_v2:{challenger_aid}:HK"
    )
    ckey = active_strategy_key("default", "10000001", account_id=challenger_aid)
    assert ckey == f"trade:active_strategy:default:10000001:acc:{challenger_aid}"
    assert parse_active_strategy_key(ckey) == ("default", "10000001")
    assert parse_active_strategy_account_id(ckey) == challenger_aid
    assert parse_active_strategy_account_id("trade:active_strategy:default:10000001") is None


@pytest.mark.asyncio
async def test_simulation_manager_multi_account_redis_isolation():
    redis = FakeRedisClient()
    mgr = SimulationAccountManager(redis)  # type: ignore[arg-type]

    default_aid = "sim:default:10000001"
    challenger_aid = "sim:default:10000001:challenger_m2"

    acc_champ = await mgr.init_account(
        10000001, initial_cash=1_000_000.0, tenant_id="default", account_id=default_aid
    )
    acc_chal = await mgr.init_account(
        10000001, initial_cash=500_000.0, tenant_id="default", account_id=challenger_aid
    )

    assert acc_champ["cash"] == 1_000_000.0
    assert acc_chal["cash"] == 500_000.0

    # Verify both accounts live in completely separate Redis keys and read back independently
    champ_key = account_key("default", "10000001", account_id=default_aid)
    chal_key = account_key("default", "10000001", account_id=challenger_aid)
    assert champ_key != chal_key

    # Mutate challenger cache directly and confirm champion stays untouched
    chal_payload = json.loads(redis.client.get(chal_key) or "{}")
    chal_payload["cash"] = 380_000.0
    chal_payload["available_cash"] = 380_000.0
    redis.client.set(chal_key, json.dumps(chal_payload))

    fetched_champ = await mgr.get_account(
        10000001, tenant_id="default", account_id=default_aid
    )
    fetched_chal = await mgr.get_account(
        10000001, tenant_id="default", account_id=challenger_aid
    )
    assert fetched_champ is not None and fetched_champ["cash"] == 1_000_000.0
    assert fetched_chal is not None and fetched_chal["cash"] == 380_000.0


@pytest.mark.asyncio
async def test_daily_snapshot_and_projection_isolated_by_account_id(async_db_session: AsyncSession):
    db = async_db_session
    redis = FakeRedisClient()

    champ = await SimulationArenaService.create_account(
        db,
        tenant_id="default",
        user_id="10000001",
        name="Champion v1",
        role="champion",
        initial_balance=1_000_000.0,
        bound_model_id="model_v1",
        bound_strategy_id="strat_top20",
        is_default=True,
        redis=redis,  # type: ignore[arg-type]
    )
    chal = await SimulationArenaService.create_account(
        db,
        tenant_id="default",
        user_id="10000001",
        name="Challenger v2",
        role="challenger",
        initial_balance=1_000_000.0,
        bound_model_id="model_v2",
        bound_strategy_id="strat_top20",
        is_default=False,
        redis=redis,  # type: ignore[arg-type]
    )

    # Write distinct position lots for each account on the same user
    db.add(
        SimulationPositionLot(
            account_id=champ["account_id"],
            tenant_id="default",
            user_id="10000001",
            symbol="600519.SH",
            position_side="long",
            quantity_open=100.0,
            quantity_remaining=100.0,
            cost_price=1500.0,
            cost_amount=150_000.0,
            status="open",
        )
    )
    db.add(
        SimulationPositionLot(
            account_id=chal["account_id"],
            tenant_id="default",
            user_id="10000001",
            symbol="000858.SZ",
            position_side="long",
            quantity_open=500.0,
            quantity_remaining=500.0,
            cost_price=140.0,
            cost_amount=70_000.0,
            status="open",
        )
    )
    await db.commit()

    # Projection must only return lots belonging to the requested account_id
    async def _price_loader(symbol: str) -> float:
        return 1550.0 if symbol == "600519.SH" else 150.0

    proj_svc = SimulationProjectionService(db)
    proj_champ = await proj_svc.load_projection(
        tenant_id="default",
        user_id="10000001",
        account_id=champ["account_id"],
        latest_price_loader=_price_loader,
    )
    proj_chal = await proj_svc.load_projection(
        tenant_id="default",
        user_id="10000001",
        account_id=chal["account_id"],
        latest_price_loader=_price_loader,
    )
    assert list(proj_champ.positions.keys()) == ["600519.SH"]
    assert list(proj_chal.positions.keys()) == ["000858.SZ"]

    # Daily snapshot replace for challenger on the same date must NOT delete champion's snapshot
    snap_day = date(2026, 11, 20)
    snap_svc = SimulationDailySnapshotService(db)
    await snap_svc.replace_daily_snapshot(
        tenant_id="default",
        user_id="10000001",
        account_id=champ["account_id"],
        snapshot_date=snap_day,
        account_payload={
            "cash": 845_000.0,
            "available_cash": 845_000.0,
            "equity": 1_005_000.0,
            "total_asset": 1_005_000.0,
            "initial_equity": 1_000_000.0,
        },
        positions={"600519.SH": {"volume": 100.0, "price": 1550.0, "cost": 1500.0, "market_value": 155_000.0}},
    )
    await snap_svc.replace_daily_snapshot(
        tenant_id="default",
        user_id="10000001",
        account_id=chal["account_id"],
        snapshot_date=snap_day,
        account_payload={
            "cash": 925_000.0,
            "available_cash": 925_000.0,
            "equity": 1_020_000.0,
            "total_asset": 1_020_000.0,
            "initial_equity": 1_000_000.0,
        },
        positions={"000858.SZ": {"volume": 500.0, "price": 150.0, "cost": 140.0, "market_value": 75_000.0}},
    )
    await db.commit()

    # Compare scoreboard: both accounts must have their own snapshot + Rank IC + excess return
    await db.execute(
        text(
            """
            INSERT INTO qm_model_inference_quality (model_id, trade_date, signals_count, coverage, ic, rank_ic)
            VALUES
              ('model_v1', '2026-11-18', 1000, 0.95, 0.04, 0.052),
              ('model_v2', '2026-11-18', 1000, 0.96, 0.06, 0.078)
            """
        )
    )
    await db.commit()

    report = await SimulationArenaService.compare_accounts(
        db, tenant_id="default", user_id="10000001", redis=redis  # type: ignore[arg-type]
    )
    assert report["champion_account_id"] == champ["account_id"]
    by_id = {a["account_id"]: a for a in report["accounts"]}
    assert by_id[champ["account_id"]]["days"] == 1
    assert by_id[champ["account_id"]]["total_return"] == pytest.approx(0.005, rel=1e-4)
    assert by_id[champ["account_id"]]["live_rank_ic"] == pytest.approx(0.052, rel=1e-4)
    assert by_id[chal["account_id"]]["days"] == 1
    assert by_id[chal["account_id"]]["total_return"] == pytest.approx(0.020, rel=1e-4)
    assert by_id[chal["account_id"]]["excess_return_vs_champion"] == pytest.approx(0.015, rel=1e-4)
    assert by_id[chal["account_id"]]["live_rank_ic"] == pytest.approx(0.078, rel=1e-4)


@pytest.mark.asyncio
async def test_hosted_scheduler_dispatches_parallel_accounts_with_distinct_models():
    redis = FakeRedisClient()
    champ_aid = "sim:default:10000001"
    chal_aid = "sim:default:10000001:challenger_cosine"

    assert _lock_key(
        tenant_id="default", user_id="10000001", strategy_id="s1", trade_date="2026-11-27", phase="BUY", account_id=champ_aid
    ) != _lock_key(
        tenant_id="default", user_id="10000001", strategy_id="s1", trade_date="2026-11-27", phase="BUY", account_id=chal_aid
    )
    assert _task_id(
        tenant_id="default", user_id="10000001", strategy_id="s1", trade_date="2026-11-27", phase="BUY", account_id=champ_aid
    ) != _task_id(
        tenant_id="default", user_id="10000001", strategy_id="s1", trade_date="2026-11-27", phase="BUY", account_id=chal_aid
    )

    k1 = active_strategy_key("default", "10000001", account_id=champ_aid)
    k2 = active_strategy_key("default", "10000001", account_id=chal_aid)
    base_cfg = {
        "enabled": True,
        "schedule_type": "interval",
        "rebalance_days": 1,
        "buy_time": "09:35",
        "sell_time": "09:35",
        "enabled_sessions": ["AM", "PM"],
    }
    redis.client.set(
        k1,
        json.dumps(
            {
                "mode": "SIMULATION",
                "tenant_id": "default",
                "user_id": "10000001",
                "account_id": champ_aid,
                "strategy_id": "strat_top20",
                "model_id": "model_champion",
                "hosted_trade_enabled": True,
                "live_trade_config": {**base_cfg, "model_id": "model_champion"},
            }
        ),
    )
    redis.client.set(
        k2,
        json.dumps(
            {
                "mode": "SIMULATION",
                "tenant_id": "default",
                "user_id": "10000001",
                "account_id": chal_aid,
                "strategy_id": "strat_top20",
                "model_id": "model_challenger",
                "hosted_trade_enabled": True,
                "live_trade_config": {**base_cfg, "model_id": "model_challenger"},
            }
        ),
    )

    scheduler = SimulationHostedScheduler(redis=redis)  # type: ignore[arg-type]
    captured_calls: list[dict[str, Any]] = []

    async def fake_run_cycle(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(kwargs)
        return {"status": "succeeded", "filled_count": 3, "orders": [], "run_id": kwargs["run_id"]}

    from zoneinfo import ZoneInfo

    trigger_now = datetime(2026, 11, 27, 9, 35, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    with (
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.run_simulation_cycle_for_active",
            side_effect=fake_run_cycle,
        ),
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.SimulationRebalanceJobService.expire_outdated_jobs",
            new=AsyncMock(),
        ),
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.SimulationRebalanceJobService.ensure_job",
            new=AsyncMock(),
        ),
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.SimulationRebalanceJobService.mark_ready",
            new=AsyncMock(),
        ),
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.SimulationRebalanceJobService.mark_started",
            new=AsyncMock(),
        ),
        patch(
            "backend.services.simulation.services.simulation_hosted_scheduler.SimulationRebalanceJobService.mark_finished",
            new=AsyncMock(),
        ),
    ):
        dispatched = await scheduler.run_once(now=trigger_now)

    assert dispatched == 2
    assert len(captured_calls) == 2
    pairs = {(c["account_id"], c["model_id"]) for c in captured_calls}
    assert pairs == {
        (champ_aid, "model_champion"),
        (chal_aid, "model_challenger"),
    }


def test_list_default_models_includes_arena_bound_models(tmp_path: Path):
    db_file = tmp_path / "gap_backfill_test.db"
    db_url = f"sqlite:///{db_file}"
    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE qm_user_models (
                    tenant_id TEXT,
                    user_id TEXT,
                    model_id TEXT,
                    storage_path TEXT,
                    metadata_json TEXT,
                    is_default BOOLEAN,
                    status TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE simulation_accounts (
                    account_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    user_id TEXT,
                    bound_model_id TEXT,
                    is_archived BOOLEAN
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO qm_user_models VALUES
                  ('default', '10000001', 'model_default', '/m/default.pkl', '{}', 1, 'ready'),
                  ('default', '10000001', 'model_challenger', '/m/chal.pkl', '{}', 0, 'ready'),
                  ('default', '10000001', 'model_archived', '/m/arch.pkl', '{}', 0, 'ready'),
                  ('default', '10000001', 'model_unused', '/m/unused.pkl', '{}', 0, 'ready')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO simulation_accounts VALUES
                  ('sim:default:10000001', 'default', '10000001', 'model_default', 0),
                  ('sim:default:10000001:c1', 'default', '10000001', 'model_challenger', 0),
                  ('sim:default:10000001:old', 'default', '10000001', 'model_archived', 1)
                """
            )
        )
    eng.dispose()

    models = list_default_models(db_url=db_url)
    model_ids = [m["model_id"] for m in models]
    assert model_ids == ["model_default", "model_challenger"]


@pytest.mark.asyncio
async def test_arena_router_end_to_end(async_db_session: AsyncSession):
    redis = FakeRedisClient()
    app = FastAPI()
    app.include_router(arena_router, prefix="/api/v1/simulation")

    async def _override_db():
        yield async_db_session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_redis] = lambda: redis
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user_id="10000001",
        tenant_id="default",
        raw_sub="10000001",
        roles=["admin"],
        is_admin=True,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(
            "/api/v1/simulation/arena/accounts",
            json={
                "name": "Champion LightGBM",
                "role": "champion",
                "initial_balance": 1_000_000,
                "bound_model_id": "m_lgb",
                "bound_strategy_id": "s_top20",
                "is_default": True,
            },
        )
        assert r1.status_code == 201
        champ_id = r1.json()["account"]["account_id"]

        r2 = await client.post(
            "/api/v1/simulation/arena/accounts",
            json={
                "name": "Challenger Cosine",
                "role": "challenger",
                "initial_balance": 1_000_000,
                "bound_model_id": "m_cos",
                "bound_strategy_id": "s_top20",
            },
        )
        assert r2.status_code == 201
        chal_id = r2.json()["account"]["account_id"]

        # Start hosted on challenger
        r_start = await client.post(
            f"/api/v1/simulation/arena/accounts/{chal_id}/hosted/start",
            json={"live_trade_config": {"schedule_type": "daily", "buy_time": "09:35"}},
        )
        assert r_start.status_code == 200
        assert r_start.json()["active"] is True

        # List shows both accounts with challenger hosted active
        r_list = await client.get("/api/v1/simulation/arena/accounts")
        assert r_list.status_code == 200
        assert r_list.json()["total"] == 2
        by_id = {a["account_id"]: a for a in r_list.json()["accounts"]}
        assert by_id[chal_id]["hosted"]["active"] is True
        assert by_id[champ_id]["hosted"]["active"] is False

        # Promote challenger -> champion
        r_prom = await client.post(f"/api/v1/simulation/arena/accounts/{chal_id}/promote")
        assert r_prom.status_code == 200
        assert r_prom.json()["account"]["role"] == "champion"

        # Compare endpoint works
        r_cmp = await client.get("/api/v1/simulation/arena/compare")
        assert r_cmp.status_code == 200
        assert r_cmp.json()["champion_account_id"] == chal_id

        # Stop hosted on challenger
        r_stop = await client.post(f"/api/v1/simulation/arena/accounts/{chal_id}/hosted/stop")
        assert r_stop.status_code == 200
        assert r_stop.json()["stopped"] is True
