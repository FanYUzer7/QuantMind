"""Explicit opt-in adapter; never fetches remote data or falls back silently."""

import os

from backend.services.engine.data_platform.base import (
    DataUnavailable,
    InvalidFieldRequest,
    OfflineDataSourceAdapter,
)
from backend.services.engine.data_platform.own_data.hub import OwnWarehouseHub


class OwnWarehouseAdapter(OfflineDataSourceAdapter):
    name = "own_warehouse"
    markets = ["A"]
    fields = {"daily_kline", "stock_list", "valuation", "financial_report", "dividend"}

    def __init__(self):
        self.hub = OwnWarehouseHub()

    def fetch_daily(self, symbol, start, end, *, adjust="qfq"):
        frame = self.hub.fetch_daily_kline(symbol, start, end, adjust=adjust)
        if frame.empty:
            raise DataUnavailable("Owned warehouse has no bars in this window")
        return frame

    def fetch_meta(self, market):
        if market != "A":
            raise InvalidFieldRequest("Owned warehouse adapter only supports A")
        return self.hub.fetch_stock_list().rename(
            columns={"Symbol": "symbol", "Name": "name"}
        )

    def fetch_field(self, field, symbol, *, start=None, end=None, **kwargs):
        if field == "daily_kline":
            return self.fetch_daily(
                symbol, start, end, adjust=kwargs.get("adjust", "qfq")
            )
        if field == "stock_list":
            return self.fetch_meta("A")
        if field == "valuation":
            return self.hub.fetch_valuation(symbol, start, end)
        if field == "dividend":
            return self.hub.fetch_dividend_factors(symbol)
        if field == "financial_report":
            return self.hub.fetch_financial(
                symbol, kwargs.get("statement_type", "balance"), start, end
            )
        raise InvalidFieldRequest("Unsupported owned field: " + field)


def register():
    if not os.getenv("QM_OWN_WAREHOUSE_ROOT"):
        return False
    from backend.services.engine.data_platform.registry import get_registry

    get_registry().register(OwnWarehouseAdapter)
    return True
