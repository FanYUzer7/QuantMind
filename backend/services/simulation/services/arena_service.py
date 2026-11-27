"""
Multi-account Champion-Challenger simulation arena service.

Allows a single user to run multiple isolated simulation accounts in parallel,
each bound to its own model and/or strategy, with independent Redis caches,
hosted-scheduler payloads, daily equity curves, and side-by-side performance
comparison (return, max drawdown, Sharpe, cash weight, turnover/fills, and
live T+5 Rank IC from ``qm_model_inference_quality``).
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.simulation.models.account import SimulationAccount
from backend.services.simulation.models.account_daily import SimulationAccountDaily
from backend.services.simulation.models.fill import SimulationFill
from backend.services.simulation.models.position_lot import SimulationPositionLot
from backend.services.trade_shared.redis_client import RedisClient
from backend.services.trade_shared.simulation_manager import SimulationAccountManager
from backend.shared.logging_config import get_logger
from backend.shared.simulation_account_keys import (
    active_strategy_key,
    build_default_account_id,
    is_default_account_id,
)

logger = get_logger(__name__)

VALID_ROLES = {"champion", "challenger", "baseline"}
_TRADING_DAYS_PER_YEAR = 252.0


def build_arena_account_id(tenant_id: str, user_id: str, *, slug: str | None = None) -> str:
    """Build a unique account_id for a non-default arena sub-account."""
    t = str(tenant_id or "default").strip() or "default"
    u = str(user_id).strip()
    token = (slug or "").strip().lower()
    if token:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)[:24]
        suffix = f"{cleaned}_{uuid.uuid4().hex[:6]}" if cleaned else uuid.uuid4().hex[:10]
    else:
        suffix = uuid.uuid4().hex[:10]
    return f"sim:{t}:{u}:{suffix}"


def _serialize_account(
    row: SimulationAccount,
    *,
    hosted_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "account_id": row.account_id,
        "tenant_id": row.tenant_id,
        "user_id": row.user_id,
        "name": row.name or ("默认模拟账户" if row.is_default else row.account_id),
        "role": row.role or ("champion" if row.is_default else "challenger"),
        "bound_model_id": row.bound_model_id,
        "bound_strategy_id": row.bound_strategy_id,
        "is_default": bool(row.is_default),
        "is_archived": bool(row.is_archived),
        "initial_balance": float(row.initial_equity or 0.0),
        "initial_equity": float(row.initial_equity or 0.0),
        "cash": float(row.cash or 0.0),
        "available_cash": float(row.available_cash or 0.0),
        "frozen_cash": float(row.frozen_cash or 0.0),
        "equity": float(row.equity or 0.0),
        "total_asset": float(row.total_asset or 0.0),
        "long_market_value": float(row.long_market_value or 0.0),
        "short_market_value": float(row.short_market_value or 0.0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "hosted": hosted_status or {"active": False},
    }


def _compute_curve_metrics(
    snapshots: list[SimulationAccountDaily],
    *,
    initial_balance: float,
) -> dict[str, Any]:
    """Compute standardized return / risk / cash-weight metrics from daily snapshots."""
    if not snapshots:
        return {
            "days": 0,
            "start_date": None,
            "end_date": None,
            "latest_equity": round(initial_balance, 2),
            "total_return": 0.0,
            "annualized_return": None,
            "max_drawdown": 0.0,
            "sharpe": None,
            "win_rate": None,
            "avg_cash_weight": None,
            "curve": [],
        }

    ordered = sorted(snapshots, key=lambda r: r.snapshot_date)
    equities = [float(r.equity or 0.0) for r in ordered]
    base_equity = initial_balance if initial_balance > 0 else (equities[0] if equities[0] > 0 else 1.0)
    latest_equity = equities[-1]
    total_return = (latest_equity - base_equity) / base_equity if base_equity > 0 else 0.0

    # Daily simple returns from equity series (prepend initial_balance for day-1 return).
    daily_returns: list[float] = []
    prev = base_equity
    for eq in equities:
        if prev > 0:
            daily_returns.append((eq - prev) / prev)
        prev = eq

    # Max drawdown (anchored at initial_balance so day-1 losses are captured).
    peak = max(base_equity, equities[0])
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd

    # Annualized return & Sharpe (require >= 2 observations for std).
    n_days = len(ordered)
    annualized_return: float | None = None
    if n_days >= 2 and base_equity > 0 and latest_equity > 0:
        annualized_return = (latest_equity / base_equity) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1.0

    sharpe: float | None = None
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(max(var, 0.0))
        if std_r > 1e-12:
            sharpe = (mean_r / std_r) * math.sqrt(_TRADING_DAYS_PER_YEAR)

    win_rate: float | None = None
    if daily_returns:
        wins = sum(1 for r in daily_returns if r > 0)
        win_rate = wins / len(daily_returns)

    cash_weights: list[float] = []
    for r in ordered:
        ta = float(r.total_asset or r.equity or 0.0)
        if ta > 0:
            cash_weights.append(max(0.0, min(1.0, float(r.cash or 0.0) / ta)))
    avg_cash_weight = (sum(cash_weights) / len(cash_weights)) if cash_weights else None

    curve = [
        {
            "date": r.snapshot_date.isoformat(),
            "equity": round(float(r.equity or 0.0), 2),
            "cash": round(float(r.cash or 0.0), 2),
            "daily_pnl": round(float(r.daily_pnl or 0.0), 2),
            "total_pnl": round(float(r.total_pnl or 0.0), 2),
            "cum_return": round(
                ((float(r.equity or 0.0) - base_equity) / base_equity) if base_equity > 0 else 0.0,
                6,
            ),
        }
        for r in ordered
    ]

    return {
        "days": n_days,
        "start_date": ordered[0].snapshot_date.isoformat(),
        "end_date": ordered[-1].snapshot_date.isoformat(),
        "latest_equity": round(latest_equity, 2),
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6) if annualized_return is not None else None,
        "max_drawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "avg_cash_weight": round(avg_cash_weight, 4) if avg_cash_weight is not None else None,
        "curve": curve,
    }


class SimulationArenaService:
    """CRUD, hosted lifecycle, and Champion-Challenger comparison for multi-account simulation."""

    @staticmethod
    def _read_hosted_status(
        redis: RedisClient | None,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
    ) -> dict[str, Any]:
        if redis is None or getattr(redis, "client", None) is None:
            return {"active": False}
        key = active_strategy_key(tenant_id, user_id, account_id=account_id)
        try:
            raw = redis.client.get(key)
        except Exception:
            return {"active": False}
        if not raw:
            return {"active": False}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
        except Exception:
            return {"active": False}
        if not isinstance(payload, dict):
            return {"active": False}
        return {
            "active": True,
            "redis_key": key,
            "strategy_id": payload.get("strategy_id"),
            "strategy_name": payload.get("strategy_name"),
            "model_id": payload.get("model_id"),
            "started_at": payload.get("started_at"),
            "live_trade_config": payload.get("live_trade_config"),
        }

    @classmethod
    async def list_accounts(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        include_archived: bool = False,
        redis: RedisClient | None = None,
    ) -> list[dict[str, Any]]:
        stmt = select(SimulationAccount).where(
            SimulationAccount.tenant_id == str(tenant_id),
            SimulationAccount.user_id == str(user_id),
        )
        if not include_archived:
            stmt = stmt.where(SimulationAccount.is_archived.is_(False))
        stmt = stmt.order_by(
            desc(SimulationAccount.is_default),
            SimulationAccount.created_at.asc(),
        )
        rows = (await db.execute(stmt)).scalars().all()
        return [
            _serialize_account(
                row,
                hosted_status=cls._read_hosted_status(
                    redis,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                    account_id=row.account_id,
                ),
            )
            for row in rows
        ]

    @classmethod
    async def get_account(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
        redis: RedisClient | None = None,
    ) -> dict[str, Any] | None:
        stmt = select(SimulationAccount).where(
            SimulationAccount.tenant_id == str(tenant_id),
            SimulationAccount.user_id == str(user_id),
            SimulationAccount.account_id == str(account_id),
        )
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return _serialize_account(
            row,
            hosted_status=cls._read_hosted_status(
                redis,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                account_id=row.account_id,
            ),
        )

    @classmethod
    async def create_account(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        role: str = "challenger",
        initial_balance: float = 1_000_000.0,
        bound_model_id: str | None = None,
        bound_strategy_id: str | None = None,
        is_default: bool = False,
        redis: RedisClient | None = None,
    ) -> dict[str, Any]:
        norm_role = str(role or "challenger").strip().lower()
        if norm_role not in VALID_ROLES:
            raise ValueError(f"invalid role '{role}', expected one of {sorted(VALID_ROLES)}")
        balance = float(initial_balance)
        if balance <= 0:
            raise ValueError("initial_balance must be > 0")

        if is_default:
            account_id = build_default_account_id(tenant_id, user_id)
            existing = (
                await db.execute(
                    select(SimulationAccount).where(
                        SimulationAccount.account_id == account_id
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"default account already exists: {account_id}")
        else:
            account_id = build_arena_account_id(tenant_id, user_id, slug=norm_role)

        now = datetime.utcnow()
        row = SimulationAccount(
            account_id=account_id,
            tenant_id=str(tenant_id or "default"),
            user_id=str(user_id),
            name=str(name or account_id).strip()[:128],
            role=norm_role,
            bound_model_id=str(bound_model_id).strip() if bound_model_id else None,
            bound_strategy_id=str(bound_strategy_id).strip() if bound_strategy_id else None,
            is_default=bool(is_default),
            is_archived=False,
            initial_equity=balance,
            cash=balance,
            available_cash=balance,
            frozen_cash=0.0,
            long_market_value=0.0,
            short_market_value=0.0,
            total_asset=balance,
            liabilities=0.0,
            equity=balance,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

        if redis is not None:
            manager = SimulationAccountManager(redis)
            await manager.init_account(
                user_id=int(user_id) if str(user_id).isdigit() else user_id,  # type: ignore[arg-type]
                initial_cash=balance,
                tenant_id=str(tenant_id or "default"),
                account_id=account_id,
            )

        return _serialize_account(
            row,
            hosted_status=cls._read_hosted_status(
                redis,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                account_id=row.account_id,
            ),
        )

    @classmethod
    async def update_account(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
        name: str | None = None,
        role: str | None = None,
        bound_model_id: str | None = None,
        bound_strategy_id: str | None = None,
        is_archived: bool | None = None,
        redis: RedisClient | None = None,
    ) -> dict[str, Any] | None:
        stmt = select(SimulationAccount).where(
            SimulationAccount.tenant_id == str(tenant_id),
            SimulationAccount.user_id == str(user_id),
            SimulationAccount.account_id == str(account_id),
        )
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None

        if name is not None and str(name).strip():
            row.name = str(name).strip()[:128]
        if role is not None:
            norm_role = str(role).strip().lower()
            if norm_role not in VALID_ROLES:
                raise ValueError(f"invalid role '{role}', expected one of {sorted(VALID_ROLES)}")
            row.role = norm_role
        if bound_model_id is not None:
            row.bound_model_id = str(bound_model_id).strip() or None
        if bound_strategy_id is not None:
            row.bound_strategy_id = str(bound_strategy_id).strip() or None
        if is_archived is not None:
            row.is_archived = bool(is_archived)
            if row.is_archived and redis is not None:
                cls.stop_hosted(
                    redis,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                    account_id=row.account_id,
                )

        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)
        return _serialize_account(
            row,
            hosted_status=cls._read_hosted_status(
                redis,
                tenant_id=row.tenant_id,
                user_id=row.user_id,
                account_id=row.account_id,
            ),
        )

    @classmethod
    async def promote_champion(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
        redis: RedisClient | None = None,
    ) -> dict[str, Any] | None:
        """Promote target account to 'champion' and demote previous champions to 'challenger'."""
        rows = (
            await db.execute(
                select(SimulationAccount).where(
                    SimulationAccount.tenant_id == str(tenant_id),
                    SimulationAccount.user_id == str(user_id),
                    SimulationAccount.is_archived.is_(False),
                )
            )
        ).scalars().all()
        target = next((r for r in rows if r.account_id == str(account_id)), None)
        if target is None:
            return None
        now = datetime.utcnow()
        for row in rows:
            if row.account_id == target.account_id:
                row.role = "champion"
                row.updated_at = now
            elif row.role == "champion":
                row.role = "challenger"
                row.updated_at = now
        await db.commit()
        await db.refresh(target)
        return _serialize_account(
            target,
            hosted_status=cls._read_hosted_status(
                redis,
                tenant_id=target.tenant_id,
                user_id=target.user_id,
                account_id=target.account_id,
            ),
        )

    @classmethod
    async def start_hosted(
        cls,
        db: AsyncSession,
        redis: RedisClient,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
        strategy_id: str | None = None,
        strategy_name: str | None = None,
        model_id: str | None = None,
        live_trade_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enable hosted automatic paper trading for a specific simulation account."""
        row = (
            await db.execute(
                select(SimulationAccount).where(
                    SimulationAccount.tenant_id == str(tenant_id),
                    SimulationAccount.user_id == str(user_id),
                    SimulationAccount.account_id == str(account_id),
                    SimulationAccount.is_archived.is_(False),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ValueError(f"active simulation account not found: {account_id}")

        resolved_strategy_id = str(strategy_id or row.bound_strategy_id or "").strip()
        if not resolved_strategy_id:
            raise ValueError("strategy_id is required (pass in request or bind on account)")
        resolved_model_id = str(model_id or row.bound_model_id or "").strip() or None

        if resolved_strategy_id != row.bound_strategy_id or resolved_model_id != row.bound_model_id:
            row.bound_strategy_id = resolved_strategy_id
            row.bound_model_id = resolved_model_id
            row.updated_at = datetime.utcnow()
            await db.commit()

        cfg = dict(live_trade_config or {})
        cfg.setdefault("enabled", True)
        if resolved_model_id:
            cfg["model_id"] = resolved_model_id

        payload = {
            "mode": "SIMULATION",
            "tenant_id": str(tenant_id or "default"),
            "user_id": str(user_id),
            "account_id": row.account_id,
            "account_name": row.name,
            "role": row.role,
            "strategy_id": resolved_strategy_id,
            "strategy_name": str(strategy_name or row.name or resolved_strategy_id),
            "model_id": resolved_model_id,
            "hosted_trade_enabled": True,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "live_trade_config": cfg,
        }
        key = active_strategy_key(tenant_id, user_id, account_id=row.account_id)
        if getattr(redis, "client", None) is None:
            raise RuntimeError("redis client unavailable")
        redis.client.set(key, json.dumps(payload, ensure_ascii=False))

        # Ensure Redis account cache is warm for this sub-account.
        manager = SimulationAccountManager(redis)
        uid_arg: Any = int(user_id) if str(user_id).isdigit() else str(user_id)
        existing_cached = await manager.get_account(
            uid_arg, tenant_id=str(tenant_id), account_id=row.account_id
        )
        if existing_cached is None:
            await manager.init_account(
                user_id=uid_arg,
                initial_cash=float(row.initial_equity or 1_000_000.0),
                tenant_id=str(tenant_id),
                account_id=row.account_id,
            )

        return {
            "account_id": row.account_id,
            "redis_key": key,
            "active": True,
            "payload": payload,
        }

    @staticmethod
    def stop_hosted(
        redis: RedisClient,
        *,
        tenant_id: str,
        user_id: str,
        account_id: str,
    ) -> dict[str, Any]:
        key = active_strategy_key(tenant_id, user_id, account_id=account_id)
        deleted = 0
        if getattr(redis, "client", None) is not None:
            deleted = int(redis.client.delete(key) or 0)
        return {
            "account_id": account_id,
            "redis_key": key,
            "stopped": deleted > 0,
        }

    @classmethod
    async def _load_model_rank_ic(
        cls,
        db: AsyncSession,
        model_ids: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Read realized T+5 Rank IC summary from ``qm_model_inference_quality`` if available."""
        unique_ids = sorted({m for m in model_ids if m})
        if not unique_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for mid in unique_ids:
            clauses = ["model_id = :model_id", "rank_ic IS NOT NULL"]
            params: dict[str, Any] = {"model_id": mid}
            if start_date is not None:
                clauses.append("trade_date >= :start_date")
                params["start_date"] = start_date
            if end_date is not None:
                clauses.append("trade_date <= :end_date")
                params["end_date"] = end_date
            where_sql = " AND ".join(clauses)
            try:
                res = await db.execute(
                    text(
                        f"SELECT COUNT(*) AS n, AVG(rank_ic) AS mean_ic, "
                        f"AVG(ic) AS mean_pearson_ic "
                        f"FROM qm_model_inference_quality WHERE {where_sql}"
                    ),
                    params,
                )
                row = res.first()
                if row and int(row[0] or 0) > 0:
                    out[mid] = {
                        "ic_days": int(row[0]),
                        "mean_rank_ic": round(float(row[1]), 6) if row[1] is not None else None,
                        "mean_ic": round(float(row[2]), 6) if row[2] is not None else None,
                    }
            except Exception as exc:
                # Table may not exist in lightweight/offline test DBs.
                logger.debug("qm_model_inference_quality lookup skipped for %s: %s", mid, exc)
                break
        return out

    @classmethod
    async def compare_accounts(
        cls,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: str,
        account_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        include_archived: bool = False,
        redis: RedisClient | None = None,
    ) -> dict[str, Any]:
        """Build side-by-side Champion-Challenger scoreboard & equity curves."""
        stmt = select(SimulationAccount).where(
            SimulationAccount.tenant_id == str(tenant_id),
            SimulationAccount.user_id == str(user_id),
        )
        if not include_archived:
            stmt = stmt.where(SimulationAccount.is_archived.is_(False))
        if account_ids:
            stmt = stmt.where(SimulationAccount.account_id.in_(account_ids))
        stmt = stmt.order_by(
            desc(SimulationAccount.is_default),
            SimulationAccount.created_at.asc(),
        )
        accounts = (await db.execute(stmt)).scalars().all()
        if not accounts:
            return {
                "tenant_id": str(tenant_id),
                "user_id": str(user_id),
                "champion_account_id": None,
                "accounts": [],
            }

        target_ids = [a.account_id for a in accounts]

        # Daily snapshots
        snap_stmt = select(SimulationAccountDaily).where(
            SimulationAccountDaily.account_id.in_(target_ids)
        )
        if start_date is not None:
            snap_stmt = snap_stmt.where(SimulationAccountDaily.snapshot_date >= start_date)
        if end_date is not None:
            snap_stmt = snap_stmt.where(SimulationAccountDaily.snapshot_date <= end_date)
        snap_rows = (await db.execute(snap_stmt)).scalars().all()
        snaps_by_acc: dict[str, list[SimulationAccountDaily]] = {aid: [] for aid in target_ids}
        for s in snap_rows:
            snaps_by_acc.setdefault(s.account_id, []).append(s)

        # Fill counts & gross traded notional per account
        fill_stmt = (
            select(
                SimulationFill.account_id,
                func.count(SimulationFill.id),
                func.coalesce(func.sum(SimulationFill.gross_amount), 0.0),
                func.coalesce(
                    func.sum(
                        SimulationFill.commission
                        + SimulationFill.stamp_duty
                        + SimulationFill.transfer_fee
                    ),
                    0.0,
                ),
            )
            .where(SimulationFill.account_id.in_(target_ids))
            .group_by(SimulationFill.account_id)
        )
        fill_stats: dict[str, dict[str, Any]] = {}
        for aid, cnt, notional, fees in (await db.execute(fill_stmt)).all():
            fill_stats[str(aid)] = {
                "fill_count": int(cnt or 0),
                "gross_traded_amount": round(float(notional or 0.0), 2),
                "total_fees": round(float(fees or 0.0), 2),
            }

        # Open position counts per account
        pos_stmt = (
            select(SimulationPositionLot.account_id, func.count(SimulationPositionLot.id))
            .where(
                SimulationPositionLot.account_id.in_(target_ids),
                SimulationPositionLot.quantity_remaining > 0,
            )
            .group_by(SimulationPositionLot.account_id)
        )
        open_positions = {str(aid): int(cnt or 0) for aid, cnt in (await db.execute(pos_stmt)).all()}

        # Live model Rank IC
        ic_by_model = await cls._load_model_rank_ic(
            db,
            [a.bound_model_id for a in accounts if a.bound_model_id],
            start_date=start_date,
            end_date=end_date,
        )

        # Assemble entries
        entries: list[dict[str, Any]] = []
        champion_entry: dict[str, Any] | None = None
        for acc in accounts:
            base = float(acc.initial_equity or 0.0)
            metrics = _compute_curve_metrics(
                snaps_by_acc.get(acc.account_id, []),
                initial_balance=base,
            )
            fstat = fill_stats.get(
                acc.account_id,
                {"fill_count": 0, "gross_traded_amount": 0.0, "total_fees": 0.0},
            )
            turnover_ratio = (
                round(fstat["gross_traded_amount"] / base, 4) if base > 0 else 0.0
            )
            model_ic = ic_by_model.get(acc.bound_model_id or "", {})
            entry = {
                "account_id": acc.account_id,
                "name": acc.name or ("默认模拟账户" if acc.is_default else acc.account_id),
                "role": acc.role or ("champion" if acc.is_default else "challenger"),
                "is_default": bool(acc.is_default),
                "bound_model_id": acc.bound_model_id,
                "bound_strategy_id": acc.bound_strategy_id,
                "initial_balance": round(base, 2),
                "open_positions": open_positions.get(acc.account_id, 0),
                "hosted": cls._read_hosted_status(
                    redis,
                    tenant_id=acc.tenant_id,
                    user_id=acc.user_id,
                    account_id=acc.account_id,
                ),
                **metrics,
                **fstat,
                "turnover_ratio": turnover_ratio,
                "live_rank_ic": model_ic.get("mean_rank_ic"),
                "live_ic_days": model_ic.get("ic_days", 0),
            }
            entries.append(entry)
            if entry["role"] == "champion" and champion_entry is None:
                champion_entry = entry

        if champion_entry is None and entries:
            champion_entry = next(
                (e for e in entries if is_default_account_id(e["account_id"], tenant_id=tenant_id, user_id=user_id)),
                entries[0],
            )

        champ_ret = champion_entry["total_return"] if champion_entry else 0.0
        for e in entries:
            e["excess_return_vs_champion"] = round(float(e["total_return"]) - float(champ_ret), 6)

        return {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "champion_account_id": champion_entry["account_id"] if champion_entry else None,
            "accounts": entries,
        }
