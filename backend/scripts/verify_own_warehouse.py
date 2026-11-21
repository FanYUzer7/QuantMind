"""Final warehouse inventory and integration checks, offline and reproducible."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
from datetime import date, datetime, timezone

import duckdb
import numpy as np

from backend.services.engine.data_platform.own_data.warehouse import (
    sha256,
    export_dictionary,
    identifier,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, default=Path("/quant"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("Use a new verification output")
    con = duckdb.connect(str(a.root / "warehouse.duckdb"))
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute(
        "SET temp_directory=" + "'" + str(a.root / "staging").replace("'", "''") + "'"
    )
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "checking",
        "warehouse": str(a.root),
        "research_only": True,
        "tables": [],
        "checks": {},
        "limitations": [],
    }
    try:
        tables = con.execute(
            "SELECT table_name,source_family,notes FROM catalog_tables ORDER BY table_name"
        ).fetchall()
        for name, family, notes in tables:
            count = con.execute("SELECT count(*) FROM " + identifier(name)).fetchone()[
                0
            ]
            fields = con.execute(
                "SELECT count(*) FROM catalog_fields WHERE table_name=?", [name]
            ).fetchone()[0]
            con.execute(
                "UPDATE catalog_tables SET row_count=? WHERE table_name=?",
                [count, name],
            )
            report["tables"].append(
                {
                    "table": name,
                    "family": family,
                    "rows": count,
                    "fields": fields,
                    "notes": notes,
                }
            )
            print("inventory", name, count, flush=True)
        report["checks"]["offline_coverage"] = con.execute(
            "SELECT price_adjustment,count(*),count(DISTINCT symbol),min(trade_date),max(trade_date) FROM daily_bars GROUP BY 1 ORDER BY 1"
        ).fetchall()
        report["checks"]["offline_quality"] = con.execute(
            "SELECT price_adjustment,quality_flags,count(*) FROM daily_bars GROUP BY 1,2 ORDER BY 1,2"
        ).fetchall()
        conflicts = con.execute("SELECT count(*) FROM daily_conflicts").fetchone()[0]
        report["checks"]["unresolved_daily_conflict_keys"] = conflicts
        report["checks"]["quantdb_unit_partitions_flagged"] = con.execute(
            "SELECT count(*) FROM quantdb_unit_audit WHERE median_price_volume_amount_ratio IS NULL OR median_price_volume_amount_ratio NOT BETWEEN 1000 AND 100000"
        ).fetchone()[0]
        report["checks"]["quantdb_coverage"] = con.execute(
            "SELECT price_adjustment,count(*),min(trade_date),max(trade_date) FROM quantdb_daily_bars GROUP BY 1 ORDER BY 1"
        ).fetchall()
        report["checks"]["historical_import_failures_retained"] = con.execute(
            "SELECT source_path,detail FROM ingest_events WHERE status='failed'"
        ).fetchall()
        # Compare a reviewed continuous window across independently packaged sources.
        report["checks"]["daily_cross_source_2025"] = con.execute("""WITH matched AS (
          SELECT d.symbol,d.trade_date,d.close AS offline_close,q.close AS quantdb_close,
            abs(d.close-q.close) AS close_delta,
            abs(d.volume-q.volume)/nullif(d.volume,0) AS volume_relative_delta,
            abs(d.amount-q.amount)/nullif(d.amount,0) AS amount_relative_delta
          FROM daily_bars d JOIN quantdb_daily_bars q USING(symbol,trade_date,price_adjustment)
          WHERE d.price_adjustment='raw' AND d.symbol IN ('000001.SZ','000002.SZ','600519.SH','688699.SH')
            AND d.trade_date BETWEEN DATE '2025-03-03' AND DATE '2025-09-30')
          SELECT count(*),max(close_delta),max(volume_relative_delta),max(amount_relative_delta) FROM matched""").fetchone()
        report["checks"]["eastmoney_source_date_isolated"] = con.execute(
            "SELECT count(*) FROM eastmoney_margin_stock WHERE _qm_source_date<>substr(_qm_collected_at,1,10)"
        ).fetchone()[0]
        report["checks"]["normalized_quantdb_missing_st"] = con.execute(
            "SELECT count(*) FROM quantdb_daily_bars WHERE is_st IS NULL"
        ).fetchone()[0]
        report["field_dictionary_rows"] = con.execute(
            "SELECT count(*) FROM catalog_fields"
        ).fetchone()[0]
        ledger = sqlite3.connect(a.run / "download-ledger.sqlite")
        records = ledger.execute(
            "SELECT path,sha256,size,status FROM downloads"
        ).fetchall()
        if any(status != "complete" for _, _, _, status in records):
            raise AssertionError("Incomplete download ledger")
        for i, (path, digest, size, _) in enumerate(records):
            p = Path(path)
            if not p.is_absolute():
                p = a.workspace / p
            if not p.is_file() or p.stat().st_size != size or sha256(p) != digest:
                raise AssertionError("Downloaded object no longer matches ledger")
            if i % 10000 == 0:
                print("download hashes", i, len(records), flush=True)
        before = json.loads((a.run / "quantdb-usage.json").read_text())
        after = json.loads((a.run / "quantdb-usage-after.json").read_text())
        charged = (after["used_traffic"] - before["used_traffic"]) + (
            before["package_traffic"] - after["package_traffic"]
        )
        report["checks"]["quantdb_download"] = {
            "files": len(records),
            "bytes": sum(r[2] for r in records),
            "charged_bytes": charged,
            "budget_bytes": 10_000_000_000,
            "all_size_and_sha256_verified": True,
            "recovery_budget_reserved": ledger.execute(
                "SELECT coalesce(sum(reserved_bytes),0) FROM recovery_budget"
            ).fetchone()[0],
        }
        ledger.close()
        if charged > 10_000_000_000:
            raise AssertionError("Download budget exceeded")
        if conflicts:
            report["limitations"].append(
                "Daily conflict keys retained; explicit source selection required"
            )
        for name in ["native-loading-02.json", "qlib-04.json"]:
            evidence = json.loads((a.run / name).read_text())
            if evidence["status"] != "passed":
                raise AssertionError("Integration evidence failed: " + name)
            report["checks"][name] = evidence
        signals = json.loads((a.run / "inference-01.json").read_text())
        if len(signals) != 4 or not all(np.isfinite(s["score"]) for s in signals):
            raise AssertionError("Inference output incomplete")
        report["checks"]["model_reload_inference_signals"] = len(signals)
        report["limitations"] += [
            "Raw financial timing/revisions and complete historical ST are not independently certified",
            "LASSO membership and full-sample seat profiles are retained as estimated/research data, not historical truth",
            "One-minute and redundant interval/format copies were not imported; five-minute source units/adjustment remain unverified",
            "Training/inference/Qlib execution checks use four stocks and a reviewed 2025 window; no strategy performance claim",
        ]
        export_dictionary(con, a.root / "FIELD-DICTIONARY.md")
        con.execute("CHECKPOINT")
        report["status"] = "inventory_passed"
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        con.close()
        a.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str)
        )
    os.environ["QM_OWN_WAREHOUSE_ROOT"] = str(a.root)
    os.environ["QM_OWN_DAILY_SOURCE"] = "offline"
    os.environ["QM_QUANTDB_DATA_DIR"] = str(a.root / "quantdb-compatible")
    from backend.services.engine.data_platform.adapters.own_warehouse_adapter import (
        OwnWarehouseAdapter,
    )
    from backend.services.engine.data_platform.registry import SourceRegistry
    from backend.services.engine.data_platform.aggregator import (
        FieldAggregator,
        FieldRoutingTable,
    )
    from backend.services.engine.data_platform.market_hub import get_hub_for_market
    from backend.services.simulation.services.local_market_data import LocalMarketData
    from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub

    registry = SourceRegistry()
    registry.register(OwnWarehouseAdapter)
    routing = FieldRoutingTable(Path("/app/config/data_sources/own_warehouse.yaml"))
    result = FieldAggregator(registry=registry, routing=routing).fetch(
        market="A",
        field="daily_kline",
        symbol="000001.SZ",
        start=date(2025, 9, 19),
        end=date(2025, 9, 19),
        adjust="raw",
    )
    if len(result.data) != 1 or result.source_used != "own_warehouse":
        raise AssertionError("Field routing did not use owned source")
    hub = get_hub_for_market("CN")
    local = LocalMarketData(hub=hub, market="CN")
    bar = local.get_bar("000001.SZ", date(2025, 9, 19))
    if bar is None or not np.isfinite(bar.close):
        raise AssertionError("Owned simulation quote missing")
    expected = result.data.iloc[0]
    if (
        abs(bar.amount - expected.amount) > 0.01
        or abs(bar.volume - expected.volume) > 0.01
    ):
        raise AssertionError("Owned simulation rescaled canonical units")
    compat = QuantDBDataHub(a.root / "quantdb-compatible")
    native_daily = compat.fetch_daily_kline(
        "000001.SZ", date(2025, 9, 19), date(2025, 9, 19), adjust="none"
    )
    valuation = hub.fetch_valuation("000001.SZ", date(2025, 9, 19), date(2025, 9, 19))
    finance = hub.fetch_financial("000001.SZ", "balance")
    index = hub.fetch_index_kline("000300.SH", date(2025, 9, 17), date(2025, 9, 19))
    if native_daily.empty or valuation.empty or finance.empty or index.empty:
        raise AssertionError("Native compatibility / rich hub reads incomplete")
    report["checks"]["platform_reads"] = {
        "field_route_source": result.source_used,
        "daily_rows": len(result.data),
        "simulation_amount_cny": bar.amount,
        "simulation_volume_shares": bar.volume,
        "simulation_historical_st": bar.is_st,
        "native_quantdb_compatible_daily_rows": len(native_daily),
        "valuation_rows": len(valuation),
        "financial_rows": len(finance),
        "index_rows": len(index),
        "network_used": False,
    }
    report["status"] = "passed"
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(
        json.dumps(
            {
                "status": report["status"],
                "tables": len(report["tables"]),
                "fields": report["field_dictionary_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
