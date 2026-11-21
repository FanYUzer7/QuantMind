"""Opt-in own-data pilot; deliberately not in the global source registry."""

from .contracts import DailyBatch, DailyQuery, SourceSpec, DataContractError
from .readers import (
    EastmoneySavedReader,
    OwnDataRepository,
    QuantDBPreviewReader,
    TabularDailyReader,
    compare_daily,
)

__all__ = [
    "DailyBatch",
    "DailyQuery",
    "SourceSpec",
    "DataContractError",
    "EastmoneySavedReader",
    "OwnDataRepository",
    "QuantDBPreviewReader",
    "TabularDailyReader",
    "compare_daily",
]
