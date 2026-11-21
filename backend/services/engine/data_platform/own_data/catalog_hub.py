"""Owned catalog implementations of the platform's ancillary data capabilities."""

from functools import partial
import json
from pathlib import Path

import pandas as pd

from backend.shared.stock_utils import StockCodeUtil
from .warehouse import Warehouse, identifier


DATASETS = {
    "fetch_technical_indicators": "technical_indicators",
    "fetch_market_sentiment": "market_sentiment",
    "fetch_margin_trading": "margin_trading",
    "fetch_l1_factors": "l1_factors",
    "fetch_l2_factors": "l2_factors",
    "fetch_alpha_factors": "alpha_library",
    "fetch_features_daily": "features_daily",
}


def frame(hub, dataset, *, symbols=None, start=None, end=None, columns=None):
    table = "quantdb_" + dataset
    with Warehouse(hub.root) as w:
        cols = [r[0] for r in w.connection.execute(f"DESCRIBE {identifier(table)}").fetchall()]
        lowered = {c.lower(): c for c in cols}
        date = next((lowered[k] for k in ("time", "date", "trade_date", "dt") if k in lowered), None)
        date_expr = (
            f"coalesce(try_cast({identifier(date)} AS DATE),try_strptime(CAST({identifier(date)} AS VARCHAR),'%Y%m%d')::DATE)"
            if date else None
        )
        projection = "*"
        if columns:
            names = list(dict.fromkeys([lowered.get("symbol", "symbol"), *columns]))
            if date and date not in names:
                names.append(date)
            if any(c not in cols for c in names):
                raise ValueError("Requested owned dataset fields are unavailable")
            projection = ",".join(identifier(c) for c in names)
        conditions, params = [], []
        if symbols:
            key = lowered.get("symbol")
            if not key:
                raise ValueError("Dataset has no symbol column")
            conditions.append(f"{identifier(key)} IN ({','.join('?' for _ in symbols)})")
            params.extend(StockCodeUtil.normalize(s) for s in symbols)
        for value, op in ((start, ">="), (end, "<=")):
            if value:
                if not date_expr:
                    raise ValueError("Dataset has no declared observation date")
                conditions.append(f"{date_expr}{op}CAST(? AS DATE)")
                params.append(str(value))
        sql = f"SELECT {projection} FROM {identifier(table)}"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        result = w.connection.execute(sql, params).fetchdf()
    if date:
        parsed = pd.to_datetime(result[date].astype(str), format="mixed", errors="coerce")
        result["trade_date"] = parsed
        result["dt"] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")
    return result


def standard(hub, dataset, symbol=None, start=None, end=None):
    return frame(hub, dataset, symbols=[symbol] if symbol else None, start=start, end=end)


def factors(hub, dataset, start=None, end=None):
    return frame(hub, dataset, start=start, end=end)


def industry(hub):
    data = hub.fetch_stock_list()
    sym = "Symbol" if "Symbol" in data else "symbol"
    key = next((k for k in ("rs_hyname", "industry_name", "industry") if k in data), None)
    if not key:
        raise ValueError("Owned instrument snapshot has no industry field")
    return data[[sym, key]].rename(columns={sym: "symbol", key: "industry"})


def index_weights(hub, index_symbol):
    data = frame(hub, "index_weights")
    key = next((c for c in ("IndexSymbol", "index_symbol", "IndexCode", "index_code") if c in data), None)
    if not key:
        raise ValueError("Owned index weights have no declared index identifier")
    return data[data[key].astype(str).map(StockCodeUtil.normalize) == StockCodeUtil.normalize(index_symbol)]


def sectors(hub, sector_name=None):
    data = frame(hub, "sector_concept")
    if sector_name is not None:
        if "SectorName" not in data:
            raise ValueError("Owned sector data has no SectorName field")
        data = data[data.SectorName == sector_name]
    return data


def universe(hub, name):
    if name in {"all", "all_a"}:
        data = hub.fetch_stock_list()
        return data.rename(columns={"Symbol": "symbol"})[["symbol"]].assign(weight=1.0)
    from backend.shared.stock_pool.builtins import cn_index_symbols

    symbol = dict(cn_index_symbols()).get(name)
    if not symbol:
        raise ValueError("Unknown owned universe: " + name)
    data = index_weights(hub, symbol)
    key = next((c for c in ("Symbol", "symbol", "ConstituentCode") if c in data), None)
    if not key:
        raise ValueError("Index constituents have no symbol field")
    weight = next((c for c in ("Weight", "weight", "weight_ratio") if c in data), None)
    return pd.DataFrame({"symbol": data[key], "weight": data[weight] if weight else 1.0})


def series(hub, view, symbol, start_dt, end_dt=None, columns=None):
    dataset = view.removeprefix("qdb_").removeprefix("quantdb_")
    def day(value):
        return pd.to_datetime(str(value), format="%Y%m%d").date() if value else None
    return frame(hub, dataset, symbols=[symbol], start=day(start_dt), end=day(end_dt), columns=columns)


def latest(hub, view, symbols, dt=None, lookback=100, columns=None):
    end = pd.to_datetime(str(dt), format="%Y%m%d") if dt else None
    start = end - pd.Timedelta(days=lookback) if end is not None else None
    data = frame(hub, view.removeprefix("qdb_").removeprefix("quantdb_"), symbols=symbols, start=start.date() if start is not None else None, end=end.date() if end is not None else None, columns=columns)
    return data.sort_values("trade_date").drop_duplicates("symbol", keep="last")


def ml(hub, dataset, columns, start=None, end=None, symbols=None):
    return frame(hub, dataset, symbols=symbols, start=start, end=end, columns=columns)


def categories(hub):
    path = Path(__file__).resolve().parents[5] / "config/features/model_training_feature_catalog_v1.json"
    catalog = json.loads(path.read_text())
    return {"categories": [{
        "id": c.get("id", ""), "name": c.get("name", ""),
        "feature_count": len(c.get("features", [])),
        "sample_features": [f["key"] for f in c.get("features", [])[:5] if f.get("key")],
    } for c in catalog.get("categories", [])]}


def method(hub, name):
    if name in DATASETS:
        func = factors if name in {"fetch_l1_factors", "fetch_l2_factors", "fetch_alpha_factors"} else standard
        return partial(func, hub, DATASETS[name])
    methods = {
        "fetch_instrument_industry": industry, "fetch_index_weights": index_weights,
        "fetch_universe_stocks": universe, "fetch_series": series,
        "fetch_latest_rows": latest, "fetch_ml_columns": ml,
        "fetch_sector_members": sectors,
        "fetch_l1_factor_categories": categories,
    }
    if name not in methods:
        raise AttributeError("Unsupported owned data capability: " + name)
    return partial(methods[name], hub)
