"""Audit current references, then retire only the explicitly superseded layout."""

import argparse
import json
from pathlib import Path
import shutil

import duckdb

from backend.services.engine.data_platform.own_data.current import access
from backend.services.engine.data_platform.own_data.sync import atomic_json


def retire(workspace, *, apply=False):
    workspace = Path(workspace).resolve()
    root = workspace / "database/market"
    if not (root / "STORE.json").is_file():
        raise ValueError("Validated current warehouse required")
    verification = workspace / "experiments/validation/data-current-20261112-01/verification.json"
    proof = json.loads(verification.read_text())
    if proof.get("status") != "passed" or not proof.get("temporary_qlib_removed"):
        raise ValueError("Real-data verification must pass before retiring old data")
    targets = [
        workspace / "database/own-v001", workspace / "database/own-v002",
        workspace / "datasets/qlib-functional-v1",
        workspace / "datasets/own-warehouse-research-v001/qlib",
        workspace / "datasets/own-warehouse-research-v002/qlib",
        workspace / "docs/WORKSPACE-DIRECTORIES-20261112.csv",
    ]
    with access(root, write=True):
        con = duckdb.connect(str(root / "warehouse.duckdb"), read_only=True)
        try:
            paths = set()
            for (encoded,) in con.execute("SELECT files_json FROM catalog_tables").fetchall():
                paths.update(json.loads(encoded))
            paths.update(r[0] for r in con.execute("SELECT object_path FROM catalog_files").fetchall() if r[0])
            paths.update(r[0] for r in con.execute("SELECT parquet_path FROM catalog_members").fetchall() if r[0])
            for p in paths:
                path = Path(p).resolve(strict=True)
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("Current data still depends on an external path: " + p)
            for (sql,) in con.execute("SELECT sql FROM duckdb_views() WHERE internal=false").fetchall():
                if any(str(p) in sql for p in targets[:2]):
                    raise ValueError("A view still refers to the old layout")
        finally:
            con.close()
        import sqlite3
        state = sqlite3.connect(f"file:{root / '_sync/state.sqlite'}?mode=ro", uri=True)
        try:
            for (p,) in state.execute("SELECT path FROM versions WHERE status='complete'"):
                if not Path(p).is_relative_to(root) or not Path(p).is_file():
                    raise ValueError("Sync state still depends on the old layout")
        finally:
            state.close()
        before = shutil.disk_usage(workspace).free
        removed = []
        for target in targets:
            if not target.exists():
                continue
            if target.is_symlink():
                raise ValueError("Refusing a symlink cleanup target: " + str(target))
            if apply:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            removed.append(str(target.relative_to(workspace)))
        report = {
            "applied": apply, "checked_referenced_files": len(paths),
            "targets": removed,
            "filesystem_free_bytes_delta": shutil.disk_usage(workspace).free - before,
            "note": "Filesystem free-space delta can include other concurrent activity; source data, models and experiment records were preserved.",
        }
        if apply:
            atomic_json(root / "_ops/cleanup.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(retire(args.workspace, apply=args.apply), ensure_ascii=False))


if __name__ == "__main__":
    main()
