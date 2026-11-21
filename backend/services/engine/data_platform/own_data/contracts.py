"""Experimental, read-only daily contract. No provider is registered globally."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re
from typing import Protocol

from backend.shared.stock_utils import StockCodeUtil


class DataContractError(ValueError):
    pass


def cn_symbol(value: str, *, exchange: str | None = None) -> str:
    """Reject ambiguous bare codes (000001 could mean a stock or an index)."""
    raw = str(value).strip().upper()
    if re.fullmatch(r"\d{6}", raw):
        if exchange not in {"SH", "SZ", "BJ"}:
            raise DataContractError("Bare code needs an explicit exchange")
        raw = f"{raw}.{exchange}"
    symbol = StockCodeUtil.normalize(raw)
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", symbol):
        raise DataContractError(f"Not an explicit CN instrument: {value}")
    return symbol


@dataclass(frozen=True)
class DailyQuery:
    symbols: tuple[str, ...]
    start: str
    end: str
    adjustment: str | None = None
    max_rows: int = 5000
    allow_snapshots: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.start) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", self.end
        ):
            raise DataContractError("Use ISO YYYY-MM-DD dates")
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if start > end or (end - start).days > 93:
            raise DataContractError(
                "Pilot daily query needs a range of at most 93 days"
            )
        if not 1 <= len(self.symbols) <= 20 or not 1 <= self.max_rows <= 5000:
            raise DataContractError("Pilot query: 1..20 symbols and 1..5000 rows")
        if any(cn_symbol(s) != s for s in self.symbols):
            raise DataContractError(
                "Query symbols must already be canonical suffix codes"
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise DataContractError("Repeated query symbol")
        if self.adjustment not in {None, "raw", "qfq", "hfq", "unknown"}:
            raise DataContractError("Unknown price adjustment")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    dataset_version: str
    adjustment: str = "unknown"
    volume_unit: str = "unknown"
    amount_unit: str = "unknown"
    currency: str = "CNY"
    asset_type: str = "stock"
    evidence_only: bool = False
    snapshot: bool = False
    unit_evidence: str | None = None

    def __post_init__(self):
        if self.adjustment not in {"raw", "qfq", "hfq", "unknown"}:
            raise DataContractError("Invalid adjustment declaration")
        if self.volume_unit not in {"shares", "lots", "unknown"}:
            raise DataContractError("Invalid volume unit")
        if self.amount_unit not in {
            "yuan",
            "thousand_yuan",
            "ten_thousand_yuan",
            "unknown",
        }:
            raise DataContractError("Invalid amount unit")
        if self.asset_type not in {"stock", "etf", "index"} or self.currency != "CNY":
            raise DataContractError(
                "First contract supports CN CNY stocks/ETFs/indices only"
            )
        if not self.source_id or not self.dataset_version:
            raise DataContractError("Source and version are required")
        if (
            self.volume_unit != "unknown" or self.amount_unit != "unknown"
        ) and not self.unit_evidence:
            raise DataContractError("Declared units require supporting evidence")


@dataclass
class DailyBatch:
    source_id: str
    dataset_version: str
    rows: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    truncated: bool = False
    evidence_only: bool = False
    # A bounded reader does not certify an exchange calendar or complete history.
    requested_window_complete: bool = False

    def observed_instruments(self) -> dict:
        return {
            "kind": "observed_not_security_master",
            "instruments": sorted({r["symbol"] for r in self.rows}),
        }

    def observed_calendar(self) -> dict:
        return {
            "kind": "observed_not_exchange_calendar",
            "dates": sorted({r["trade_date"] for r in self.rows}),
        }

    def coverage(self) -> dict:
        days = sorted({r["trade_date"] for r in self.rows})
        return {
            "scope": "query_return_only",
            "rows": len(self.rows),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
            "truncated": self.truncated,
            "requested_window_complete": False,
            "evidence_only": self.evidence_only,
        }


class DailyReader(Protocol):
    spec: SourceSpec

    def read_daily(self, query: DailyQuery) -> DailyBatch: ...
