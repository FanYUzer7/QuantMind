"""Build bounded local reconciliation evidence; no network or source rewriting."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from backend.services.engine.data_platform.own_data import (
    DailyBatch,
    DailyQuery,
    SourceSpec,
    TabularDailyReader,
    EastmoneySavedReader,
)
from backend.services.engine.data_platform.own_data.reconciliation import (
    diagnose_panel,
    adjustment_relations,
    assess_feature_input,
)

SYMBOLS = ("000001.SZ", "000002.SZ", "600519.SH", "688699.SH")
WINDOWS = (
    ("2019-09-27", "2019-09-27"),
    ("2020-07-14", "2020-07-14"),
    ("2025-12-22", "2026-01-09"),
)
CSV_COLUMNS = {
    "symbol": "__symbol",
    "trade_date": "日期",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "volume": "成交量（股）",
    "amount": "成交额（元）",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/quant"))
    parser.add_argument(
        "--run", default="experiments/data-integration/own-data-reconciliation-20261101-02"
    )
    parser.add_argument("--output", default="reconciliation-01.json")
    args = parser.parse_args()
    run = args.workspace / args.run
    target = run / args.output
    if target.exists():
        raise SystemExit("Use a new report filename")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise SystemExit("Extraction is not complete")
    grouped = {}
    for entry in manifest["files"]:
        p = (run / entry["sample"]).resolve()
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sample_sha256"]:
            raise ValueError("Sample hash mismatch")
        grouped.setdefault(entry["dataset"], []).append(p)
    readers = {}
    for dataset, files in grouped.items():
        adjustment = (
            dataset.removeprefix("per_stock_")
            if dataset.startswith("per_stock_")
            else "raw"
            if dataset == "per_day_raw"
            else "unknown"
        )
        volume_unit, amount_unit, proof = "shares", "yuan", "Explicit CSV header 股/元"
        columns = CSV_COLUMNS
        if dataset == "other_csv":
            volume_unit, amount_unit, proof = (
                "lots",
                "thousand_yuan",
                "Explicit CSV header 手/千元",
            )
            columns = {
                **CSV_COLUMNS,
                "trade_date": "交易日",
                "volume": "成交量（手）",
                "amount": "成交额（千元）",
            }
        if dataset == "daily_parquet":
            volume_unit, amount_unit, proof = "unknown", "unknown", None
            columns = {
                "symbol": "code",
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "vol",
                "amount": "amount",
            }
        content_hash = hashlib.sha256(
            "".join(hashlib.sha256(f.read_bytes()).hexdigest() for f in files).encode()
        ).hexdigest()
        spec = SourceSpec(
            dataset,
            content_hash,
            adjustment,
            volume_unit,
            amount_unit,
            evidence_only=True,
            unit_evidence=proof,
        )
        readers[dataset] = TabularDailyReader(spec, files, columns)
    saved = (
        args.workspace / "experiments/tools/eastmoney-browser/versions/eastmoney-browser-demo-20261101-v030/saved-data"
    )
    readers["eastmoney_history"] = EastmoneySavedReader(
        SourceSpec("eastmoney_history", "saved-v030", "raw", evidence_only=True),
        [saved / "daily_000001_raw.qm.json", saved / "daily_600519_raw.qm.json"],
        ["daily_000001_raw", "daily_600519_raw"],
    )
    batches = {}
    for name, reader in readers.items():
        combined = DailyBatch(name, reader.spec.dataset_version, evidence_only=True)
        for start, end in WINDOWS:
            batch = reader.read_daily(DailyQuery(SYMBOLS, start, end, max_rows=100))
            combined.rows.extend(batch.rows)
            combined.issues.extend(batch.issues)
            combined.truncated |= batch.truncated
        batches[name] = combined
    reference = batches["per_stock_raw"]
    comparisons = [
        diagnose_panel(reference, batches[name])
        for name in ("per_day_raw", "other_csv", "daily_parquet", "eastmoney_history")
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": "manifest.json",
        "symbols": SYMBOLS,
        "windows": WINDOWS,
        "reference": "per_stock_raw",
        "reference_reason": "Explicit raw folder and share/yuan headers; chosen for diagnostics, not production source selection",
        "batches": {k: asdict(v) for k, v in batches.items()},
        "comparisons": comparisons,
        "adjustment_relations": adjustment_relations(
            reference, batches["per_stock_qfq"], batches["per_stock_hfq"]
        ),
        "adjusted_activity": [
            diagnose_panel(reference, batches[name])
            for name in ("per_stock_qfq", "per_stock_hfq")
        ],
        "feature_admission": {k: assess_feature_input(v) for k, v in batches.items()},
        "quantdb": {
            "status": "local_full_dataset_path_unconfirmed",
            "network_requests": 0,
        },
        "automatic_merge": False,
        "source_writes": 0,
    }
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(target),
                "batch_rows": {k: len(v.rows) for k, v in batches.items()},
                "comparisons": [
                    {
                        "source": c["candidate"],
                        "overlap": c["overlap_keys"],
                        "price_equal": c["fields"]["close"]["equal_within_1e_6"],
                        "volume_scale": c["volume_scale"]["supported_scale"],
                        "amount_scale": c["amount_scale"]["supported_scale"],
                    }
                    for c in comparisons
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
