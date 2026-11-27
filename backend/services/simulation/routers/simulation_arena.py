"""
Multi-account Champion-Challenger simulation arena router.

Endpoints (mounted under ``/api/v1/simulation`` in ``backend/services/trade/main.py``):
  - GET    /arena/accounts                        List user's arena simulation accounts
  - POST   /arena/accounts                        Create a new challenger/champion/baseline account
  - GET    /arena/accounts/{account_id}           Get single account details + hosted status
  - PATCH  /arena/accounts/{account_id}           Update name/role/model/strategy/archived
  - POST   /arena/accounts/{account_id}/promote   Promote account to champion
  - POST   /arena/accounts/{account_id}/hosted/start  Start automatic hosted paper trading
  - POST   /arena/accounts/{account_id}/hosted/stop   Stop automatic hosted paper trading
  - GET    /arena/compare                         Side-by-side Champion-Challenger scoreboard & curves
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.simulation.services.arena_service import SimulationArenaService
from backend.services.simulation.services.simulation_manager import require_sim_user_id
from backend.services.trade_shared.deps import AuthContext, get_auth_context, get_db, get_redis
from backend.services.trade_shared.redis_client import RedisClient

router = APIRouter()


class CreateArenaAccountRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    role: str = Field(default="challenger", description="champion | challenger | baseline")
    initial_balance: float = Field(default=1_000_000.0, gt=0)
    bound_model_id: str | None = None
    bound_strategy_id: str | None = None
    is_default: bool = False


class UpdateArenaAccountRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    role: str | None = Field(default=None, description="champion | challenger | baseline")
    bound_model_id: str | None = None
    bound_strategy_id: str | None = None
    is_archived: bool | None = None


class StartArenaHostedRequest(BaseModel):
    strategy_id: str | None = None
    strategy_name: str | None = None
    model_id: str | None = None
    live_trade_config: dict[str, Any] = Field(default_factory=dict)


def _sim_owner(auth: AuthContext) -> tuple[str, str]:
    tenant_id = str(auth.tenant_id or "default")
    user_id = str(require_sim_user_id(auth.user_id, tenant_id=tenant_id))
    return tenant_id, user_id


@router.get("/arena/accounts")
async def list_arena_accounts(
    include_archived: bool = Query(default=False),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    accounts = await SimulationArenaService.list_accounts(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        include_archived=include_archived,
        redis=redis,
    )
    return {"accounts": accounts, "total": len(accounts)}


@router.post("/arena/accounts", status_code=status.HTTP_201_CREATED)
async def create_arena_account(
    body: CreateArenaAccountRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    try:
        account = await SimulationArenaService.create_account(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            name=body.name,
            role=body.role,
            initial_balance=body.initial_balance,
            bound_model_id=body.bound_model_id,
            bound_strategy_id=body.bound_strategy_id,
            is_default=body.is_default,
            redis=redis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"account": account}


@router.get("/arena/accounts/{account_id}")
async def get_arena_account(
    account_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    account = await SimulationArenaService.get_account(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        account_id=account_id,
        redis=redis,
    )
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="simulation account not found")
    return {"account": account}


@router.patch("/arena/accounts/{account_id}")
async def update_arena_account(
    account_id: str,
    body: UpdateArenaAccountRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    try:
        account = await SimulationArenaService.update_account(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            account_id=account_id,
            name=body.name,
            role=body.role,
            bound_model_id=body.bound_model_id,
            bound_strategy_id=body.bound_strategy_id,
            is_archived=body.is_archived,
            redis=redis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="simulation account not found")
    return {"account": account}


@router.post("/arena/accounts/{account_id}/promote")
async def promote_arena_champion(
    account_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    account = await SimulationArenaService.promote_champion(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        account_id=account_id,
        redis=redis,
    )
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="simulation account not found")
    return {"account": account}


@router.post("/arena/accounts/{account_id}/hosted/start")
async def start_arena_hosted(
    account_id: str,
    body: StartArenaHostedRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    try:
        result = await SimulationArenaService.start_hosted(
            db,
            redis,
            tenant_id=tenant_id,
            user_id=user_id,
            account_id=account_id,
            strategy_id=body.strategy_id,
            strategy_name=body.strategy_name,
            model_id=body.model_id,
            live_trade_config=body.live_trade_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return result


@router.post("/arena/accounts/{account_id}/hosted/stop")
async def stop_arena_hosted(
    account_id: str,
    auth: AuthContext = Depends(get_auth_context),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    return SimulationArenaService.stop_hosted(
        redis,
        tenant_id=tenant_id,
        user_id=user_id,
        account_id=account_id,
    )


@router.get("/arena/compare")
async def compare_arena_accounts(
    account_ids: str | None = Query(
        default=None,
        description="Comma-separated account_ids to compare (default: all active accounts)",
    ),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    include_archived: bool = Query(default=False),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
):
    tenant_id, user_id = _sim_owner(auth)
    parsed_ids = (
        [x.strip() for x in account_ids.split(",") if x.strip()]
        if account_ids
        else None
    )
    return await SimulationArenaService.compare_accounts(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        account_ids=parsed_ids,
        start_date=start_date,
        end_date=end_date,
        include_archived=include_archived,
        redis=redis,
    )
