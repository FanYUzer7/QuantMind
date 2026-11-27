"""
Simulation account ledger root model.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.services.simulation.models import Base, TimestampMixin


class SimulationAccount(Base, TimestampMixin):
    __tablename__ = "simulation_accounts"

    account_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", index=True
    )
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="默认模拟账户")
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="champion", index=True
    )
    bound_model_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    bound_strategy_id: Mapped[str | None] = mapped_column(
        String(96), nullable=True, index=True
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    base_currency: Mapped[str] = mapped_column(
        String(16), nullable=False, default="CNY"
    )
    account_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="cash"
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")

    initial_equity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    available_cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frozen_cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    long_market_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    short_market_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_asset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liabilities: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    equity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    maintenance_margin_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    last_trade_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_projected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index(
            "idx_simulation_accounts_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "idx_simulation_accounts_model_active",
            "tenant_id",
            "bound_model_id",
            "is_archived",
        ),
    )
