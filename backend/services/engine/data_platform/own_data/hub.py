"""QuantMind-facing read-only hub for the owned warehouse."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path

import pandas as pd

from backend.shared.stock_utils import StockCodeUtil
from .warehouse import Warehouse, identifier


class OwnWarehouseHub:
    canonical_units = True

    def __init__(self, root=None, daily_source=None):
        self.root = root or os.environ["QM_OWN_WAREHOUSE_ROOT"]
        default = (
            "integrated" if any((Path(self.root) / p).exists() for p in ("STORE.json", "CURRENT.json")) else "offline"
        )
        self.daily_source = daily_source or os.getenv("QM_OWN_DAILY_SOURCE", default)
        if self.daily_source not in {"offline", "quantdb", "integrated"}:
            raise ValueError(
                "QM_OWN_DAILY_SOURCE must be offline, quantdb or integrated"
            )

    @property
    def data_dir(self):
        # Canonical callers must not bypass the selected source using legacy files.
        return Path(self.root)

    def __getattr__(self, name):
        if (Path(self.root) / "STORE.json").is_file():
            from .catalog_hub import method

            return method(self, name)
        delegated = {
            "fetch_series",
            "fetch_latest_rows",
            "fetch_technical_indicators",
            "fetch_market_sentiment",
            "fetch_margin_trading",
            "fetch_index_weights",
            "fetch_sector_members",
            "fetch_instrument_industry",
            "fetch_universe_stocks",
            "fetch_l1_factors",
            "fetch_l2_factors",
            "fetch_features_daily",
            "fetch_alpha_factors",
            "fetch_ml_columns",
            "fetch_l1_factor_categories",
        }
        if name not in delegated:
            raise AttributeError(name)
        from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub

        from .publication import current_release

        compatible = current_release(self.root)["compatible"]
        if not compatible.is_dir():
            raise ValueError("QuantDB compatibility directory not published")
        return getattr(QuantDBDataHub(compatible), name)

    @classmethod
    def get_instance(cls):
        return cls()

    @property
    def available(self):
        from pathlib import Path

        return (Path(self.root) / "warehouse.duckdb").is_file()

    def warm_up(self):
        with Warehouse(self.root) as warehouse:
            warehouse.connection.execute("SELECT 1")

    def query(self, sql, params=None):
        w = Warehouse(self.root)
        try:
            # Compatibility views are connection-local, preserving read-only storage.
            for (raw_name,) in w.connection.execute(
                "SELECT table_name FROM catalog_tables WHERE table_name LIKE 'quantdb_%'"
            ).fetchall():
                alias = "qdb_" + raw_name.removeprefix("quantdb_")
                if alias not in sql:
                    continue
                if raw_name in {"quantdb_daily_bars", "quantdb_unit_audit"}:
                    continue
                cols = {r[0] for r in w.connection.execute("DESCRIBE " + identifier(raw_name)).fetchall()}
                date_key = next((k for k in ("time", "date", "trade_date") if k in cols), None)
                extra = (
                    ",CAST(strftime(" + identifier(date_key) + "::DATE,'%Y%m%d') AS INTEGER) AS dt"
                    if date_key and "dt" not in cols else ""
                )
                w.connection.execute("CREATE OR REPLACE TEMP VIEW " + identifier(alias) + " AS SELECT *" + extra + " FROM " + identifier(raw_name))
            table = {
                "offline": "daily_bars",
                "quantdb": "quantdb_daily_bars",
                "integrated": "research_daily_bars",
            }[self.daily_source]
            for adjust, name in [
                ("raw", "daily_unadjusted"),
                ("qfq", "daily_forward"),
                ("hfq", "daily_backward"),
            ]:
                if name not in sql:
                    continue
                extra = (
                    " AND asset_type='stock'"
                    if self.daily_source == "integrated"
                    else ""
                )
                w.connection.execute(
                    f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT *, CAST(strftime(trade_date,'%Y%m%d') AS INTEGER) AS dt FROM {table} WHERE price_adjustment='{adjust}'{extra}"
                )
                w.connection.execute(
                    f"CREATE OR REPLACE TEMP VIEW qdb_{name} AS SELECT * FROM {name}"
                )
            return w.connection.execute(sql, params or []).fetchdf()
        finally:
            w.close()

    def fetch_daily_kline(self, symbol, start, end, *, adjust="qfq"):
        adjust = {"none": "raw", "unadjusted": "raw"}.get(adjust, adjust)
        w = Warehouse(self.root)
        try:
            frame = w.daily(
                [StockCodeUtil.normalize(symbol)],
                str(start),
                str(end),
                adjustment=adjust,
                source=self.daily_source,
            )
            if (
                not frame.empty
                and frame["quality_flags"]
                .str.contains("invalid_|unit_mismatch", regex=True)
                .any()
            ):
                raise ValueError(
                    "Requested bars contain quality issues; inspect daily_bars"
                )
            frame["source"] = "own_warehouse/" + self.daily_source
            # No fabricated adjustment factor. Qlib export derives the explicit price ratio.
            frame["adj_factor"] = float("nan")
            return frame
        finally:
            w.close()

    def fetch_daily_kline_batch(self, symbols, start, end, *, adjust="qfq"):
        frames = [self.fetch_daily_kline(s, start, end, adjust=adjust) for s in symbols]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_stock_list(self):
        if self.daily_source in {"quantdb", "integrated"}:
            return self.query("SELECT * FROM quantdb_instrument_detail")
        return self.query("""SELECT symbol AS Symbol, arg_max("名称",trade_date) AS Name,
            min(try_cast("上市时间" AS DATE)) AS list_date,
            max(try_cast("退市时间" AS DATE)) AS delist_date
            FROM daily_bars WHERE price_adjustment='raw' GROUP BY symbol""")

    def fetch_calendar(self, start=None, end=None):
        sql = "SELECT * FROM quantdb_trading_calendar WHERE IsTradingDay=1"
        params = []
        for value, op in [(start, ">="), (end, "<=")]:
            if value:
                sql += " AND coalesce(try_cast(TradingDate AS DATE),try_strptime(CAST(TradingDate AS VARCHAR),'%Y%m%d')::DATE)" + op + "CAST(? AS DATE)"
                params.append(str(value))
        return self.query(sql, params)

    def fetch_index_kline(self, symbol, start, end):
        frame = self.fetch_table(
            "quantdb_index_daily", symbol=symbol, start=start, end=end
        )
        if "time" in frame:
            frame["trade_date"] = pd.to_datetime(frame["time"])
        return frame

    def fetch_table(self, name, *, symbol=None, start=None, end=None):
        w = Warehouse(self.root)
        try:
            names = {
                r[0]
                for r in w.connection.execute(
                    "SELECT table_name FROM catalog_tables"
                ).fetchall()
            }
            if name not in names:
                raise ValueError("Unknown warehouse table")
            columns = {
                r[0].lower(): r[0]
                for r in w.connection.execute("DESCRIBE " + identifier(name)).fetchall()
            }
            predicates = []
            params = []
            if symbol:
                key = columns.get("symbol")
                if not key:
                    raise ValueError("Table has no explicit symbol column")
                predicates.append(identifier(key) + "=?")
                params.append(StockCodeUtil.normalize(symbol))
            for value, op in [(start, ">="), (end, "<=")]:
                if value:
                    key = next(
                        (
                            columns[k]
                            for k in ["time", "trade_date", "date"]
                            if k in columns
                        ),
                        None,
                    )
                    if not key:
                        raise ValueError("Table has no declared observation date")
                    predicates.append(
                        "try_cast("
                        + identifier(key)
                        + " AS DATE)"
                        + op
                        + "CAST(? AS DATE)"
                    )
                    params.append(str(value))
            sql = "SELECT * FROM " + identifier(name)
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            return w.connection.execute(sql, params).fetchdf()
        finally:
            w.close()

    def fetch_valuation(self, symbol=None, start=None, end=None):
        return self.fetch_table(
            "quantdb_valuation", symbol=symbol, start=start, end=end
        )

    def fetch_financial(self, symbol, statement_type="balance", start=None, end=None):
        if statement_type not in {
            "balance",
            "income",
            "cashflow",
            "capital",
            "holder_num",
            "pershare_index",
            "per_share",
            "dividend_factors",
        }:
            raise ValueError("Unknown financial statement")
        frame = self.fetch_table("quantdb_" + statement_type, symbol=symbol)
        if start or end:
            key = next((k for k in ["m_anntime", "declareDate"] if k in frame), None)
            if key is None:
                raise ValueError("Date filtering needs an explicit announcement field")
            announced = pd.to_datetime(frame[key], format="mixed", errors="coerce")
            if start:
                frame = frame.loc[announced >= pd.Timestamp(start)]
            if end:
                frame = frame.loc[announced <= pd.Timestamp(end)]
        return frame

    def fetch_dividend_factors(self, symbol):
        return self.fetch_financial(symbol, "dividend_factors")
