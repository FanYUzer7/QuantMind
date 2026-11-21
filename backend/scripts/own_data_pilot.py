"""Offline pilot report. Run with workspace scripts/quant-exec; no network."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from backend.services.engine.data_platform.own_data import (
    DailyQuery,
    SourceSpec,
    TabularDailyReader,
    QuantDBPreviewReader,
    EastmoneySavedReader,
    OwnDataRepository,
    compare_daily,
)


def build(workspace, config):
    readers, versions = [], {}
    from backend.services.engine.data_platform.own_data import (
        contracts,
        readers as source_readers,
    )

    implementation_hash = hashlib.sha256(
        Path(contracts.__file__).read_bytes()
        + Path(source_readers.__file__).read_bytes()
    ).hexdigest()
    for item in config["sources"]:
        files = [workspace / p for p in item["files"]]
        fingerprints = [
            {
                "path": str(p.relative_to(workspace)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ]
        version = hashlib.sha256(
            json.dumps(
                {
                    "files": fingerprints,
                    "mapping": item,
                    "implementation_hash": implementation_hash,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        spec = SourceSpec(
            source_id=item["source_id"],
            dataset_version=version,
            adjustment=item["adjustment"],
            volume_unit=item["volume_unit"],
            amount_unit=item["amount_unit"],
            evidence_only=True,
            snapshot=item.get("snapshot", False),
            unit_evidence=item.get("unit_evidence"),
        )
        if item["kind"] == "tabular":
            reader = TabularDailyReader(
                spec, files, item.get("columns"), symbol_map=item.get("symbol_map")
            )
        elif item["kind"] == "quantdb_preview":
            reader = QuantDBPreviewReader(spec, files[0], item["dataset"])
        elif item["kind"] == "eastmoney_saved":
            reader = EastmoneySavedReader(spec, files, item["group_ids"])
        else:
            raise ValueError("Unknown source kind")
        readers.append(reader)
        versions[spec.source_id] = fingerprints
    return OwnDataRepository(readers), versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("/quant"))
    ap.add_argument("--config", default="configs/own-data-sources.pilot.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = args.workspace / args.output
    if output.exists():
        raise SystemExit("Use a new report filename; do not overwrite old evidence")
    config = json.loads((args.workspace / args.config).read_text())
    repo, versions = build(args.workspace, config)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "source_versions": versions,
        "sources": repo.list_sources(),
        "batches": {},
        "parquet_profiles": [],
        "saved_demo_map": [],
        "network_requests": 0,
        "automatic_merge": False,
        "production_readiness": False,
    }
    batches = {}
    for item in config["sources"]:
        q = item["query"]
        query = DailyQuery(
            tuple(q["symbols"]),
            q["start"],
            q["end"],
            allow_snapshots=q.get("allow_snapshots", False),
            max_rows=20,
        )
        batch = repo.read_daily(item["source_id"], query)
        batches[item["source_id"]] = batch
        report["batches"][item["source_id"]] = {
            **asdict(batch),
            "coverage": batch.coverage(),
            "observed_calendar": batch.observed_calendar(),
            "observed_instruments": batch.observed_instruments(),
        }

    report["comparisons"] = [
        compare_daily(
            batches["offline_daily_csv_sample"], batches["eastmoney_daily_saved"]
        ),
        compare_daily(
            batches["quantdb_saved_preview"], batches["eastmoney_quote_sample"]
        ),
    ]
    import pyarrow.parquet as pq

    sample_root = args.workspace / "experiments/data-integration/own-data-pilot-20261101-01/samples"
    for p in sorted(sample_root.glob("*.parquet")):
        f = pq.ParquetFile(p)
        report["parquet_profiles"].append(
            {
                "file": str(p.relative_to(args.workspace)),
                "rows_from_footer": f.metadata.num_rows,
                "schema": str(f.schema_arrow),
                "sample": next(f.iter_batches(batch_size=2)).to_pylist(),
            }
        )
    saved = json.loads(
        (
            args.workspace
            / "experiments/tools/eastmoney-browser/versions/eastmoney-browser-demo-20261101-v040/saved-data/live-samples.qm.json"
        ).read_text()
    )
    for group in saved["groups"]:
        report["saved_demo_map"].append(
            {
                "job": group["job_id"],
                "kind": group.get("data_kind"),
                "mode": group.get("acquisition_mode"),
                "frequency": group.get("update_frequency"),
                "status": group["status"],
                "rows": len(group["rows"]),
                "columns": sorted({k for r in group["rows"] for k in r}),
                "source_dates": group.get("source_dates"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "batch_rows": {k: len(v.rows) for k, v in batches.items()},
                "comparisons": [
                    {
                        "left": r["left"],
                        "right": r["right"],
                        "overlap": r["overlap_keys"],
                    }
                    for r in report["comparisons"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
