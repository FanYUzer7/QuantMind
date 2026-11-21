"""Factor reader backed by the owned SQL catalog, independent of vendor paths."""

import json
from pathlib import Path

from ..quantdb_factor_reader import QuantDBFactorReader
from .current import access
from .current import sources
from .warehouse import Warehouse, literal


class CurrentFactorReader(QuantDBFactorReader):
    def __init__(self, data_dir=None, market=None):
        import os

        self.data_dir = Path(data_dir or os.environ["QM_OWN_WAREHOUSE_ROOT"]).resolve()
        self.market = "CN"

    def _files(self, source):
        self.validate_source(source)
        with Warehouse(self.data_dir) as w:
            row = w.connection.execute(
                "SELECT files_json FROM catalog_tables WHERE table_name=?",
                ["quantdb_" + source],
            ).fetchone()
        return [Path(p) for p in json.loads(row[0])] if row else []

    def source_path(self, source):
        self.validate_source(source)
        return self.data_dir / ("quantdb_" + source)

    def _partition_date_range(self, root):
        source = Path(root).name.removeprefix("quantdb_")
        with Warehouse(self.data_dir) as w:
            entries = sources(w.connection).get("entries", [])
        days = [str(obj["trade_date"])[:10] for dataset, _, obj in entries if dataset == source and obj.get("trade_date")]
        return (min(days), max(days)) if days else (None, None)

    def _relation(self, source):
        paths = self._files(source)
        if not paths:
            raise ValueError("Owned factor dataset is empty: " + source)
        rel = "read_parquet([" + ",".join(literal(p) for p in paths) + "],union_by_name=true,hive_partitioning=false)"
        return rel

    def _daily_backward_relation(self):
        with Warehouse(self.data_dir) as w:
            row = w.connection.execute(
                "SELECT files_json FROM catalog_tables WHERE table_name='quantdb_daily_backward'"
            ).fetchone()
        paths = json.loads(row[0]) if row else []
        if not paths:
            raise ValueError("Owned HFQ prices are required for factor labels")
        rel = "read_parquet([" + ",".join(literal(p) for p in paths) + "],union_by_name=true,hive_partitioning=false)"
        return f"(SELECT *,strftime(time::DATE,'%Y%m%d') AS dt FROM {rel})"

    def _ohlcv_donor_relation(self):
        return None

    def _donor_has_ohlcv(self):
        return self._daily_backward_relation() is not None

    def describe(self, source):
        with access(self.data_dir):
            return super().describe(source)

    def read_range(self, source, **kwargs):
        with access(self.data_dir):
            include = kwargs.pop("include_ohlcv", True)
            frame = super().read_range(source, include_ohlcv=False, **kwargs)
            if include and not frame.empty:
                from ..quantdb_factor_reader import OHLCV_COLUMNS
                with Warehouse(self.data_dir) as w:
                    prices = w.daily(
                        frame.symbol.unique().tolist(), str(kwargs["start"])[:10],
                        str(kwargs["end"])[:10], adjustment="hfq",
                    )
                if prices.quality_flags.str.contains("invalid_", regex=True).any():
                    raise ValueError("Owned training prices have invalid quality flags")
                frame = frame.merge(
                    prices[["symbol", "trade_date", *OHLCV_COLUMNS]],
                    on=["symbol", "trade_date"], how="left", validate="one_to_one",
                )
                if frame[list(OHLCV_COLUMNS)].isna().any().any():
                    raise ValueError("Owned factor rows have missing canonical OHLCV")
            return frame

    def available_dates(self, source, *, start=None, end=None):
        with access(self.data_dir):
            with Warehouse(self.data_dir) as w:
                entries = sources(w.connection).get("entries", [])
            days = sorted({str(obj["trade_date"])[:10] for dataset, _, obj in entries if dataset == source and obj.get("trade_date")})
            if days:
                return [d for d in days if (not start or d >= start) and (not end or d <= end)]
            return super().available_dates(source, start=start, end=end)
