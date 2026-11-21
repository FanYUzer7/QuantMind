"""Consolidate existing local objects without copying their contents or downloading."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

import duckdb

from backend.services.engine.data_platform.own_data.current import save_sources
from backend.services.engine.data_platform.own_data.publication import current_release
from backend.services.engine.data_platform.own_data.sync import atomic_json
from backend.services.engine.data_platform.own_data.warehouse import identifier, literal


def migrate(old_root, target):
    old_root, target = Path(old_root).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError("Migration target already exists; inspect before retrying")
    active = current_release(old_root)
    pointer = json.loads((old_root / "CURRENT.json").read_text())
    source = json.loads((old_root / pointer["source_list"]).read_text())
    target.mkdir(parents=True)
    (target / "_ops").mkdir()
    (target / ".access.lock").touch()
    db = target / "warehouse.duckdb"
    shutil.copyfile(active["database"], db)
    con = duckdb.connect(str(db))
    mapping = {}
    try:
        paths = set()
        for (encoded,) in con.execute("SELECT files_json FROM catalog_tables").fetchall():
            paths.update(json.loads(encoded))
        for table, column in (("catalog_files", "object_path"), ("catalog_members", "parquet_path")):
            paths.update(r[0] for r in con.execute(f"SELECT {column} FROM {table}").fetchall() if r[0])
        for original in sorted(paths):
            p = Path(original)
            if not p.is_file():
                raise FileNotFoundError("Referenced migration object missing: " + original)
            key = hashlib.sha256(original.encode()).hexdigest()
            dest = target / "data" / key[:2] / (key + p.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, dest)
            mapping[original] = str(dest)
        views = con.execute("SELECT view_name,sql FROM duckdb_views() WHERE internal=false").fetchall()
        values = re.compile(r"'(?:''|[^'])*'")
        con.execute("BEGIN TRANSACTION")
        # Existing dependencies remain valid until every view has been rewritten.
        for name, sql in views:
            def replace_path(match):
                value = match.group(0)[1:-1].replace("''", "'")
                return literal(mapping[value]) if value in mapping else match.group(0)
            rewritten = values.sub(replace_path, sql)
            if rewritten != sql:
                con.execute(rewritten.replace("CREATE VIEW", "CREATE OR REPLACE VIEW", 1))
        for table, column in (("catalog_files", "object_path"), ("catalog_members", "parquet_path")):
            con.execute("CREATE TEMP TABLE migration_paths (old VARCHAR,new VARCHAR)")
            con.executemany("INSERT INTO migration_paths VALUES (?,?)", list(mapping.items()))
            con.execute(f"UPDATE {table} SET {column}=p.new FROM migration_paths p WHERE {table}.{column}=p.old")
            con.execute("DROP TABLE migration_paths")
        updates = []
        for name, encoded in con.execute("SELECT table_name,files_json FROM catalog_tables").fetchall():
            updates.append((json.dumps([mapping[p] for p in json.loads(encoded)]), name))
        con.executemany("UPDATE catalog_tables SET files_json=? WHERE table_name=?", updates)
        save_sources(con, source)
        con.execute("COMMIT")
        bad = []
        for name, sql in con.execute("SELECT view_name,sql FROM duckdb_views() WHERE internal=false").fetchall():
            if str(old_root.parent) + "/own-v" in sql:
                bad.append(name)
            con.execute(f"SELECT * FROM {identifier(name)} LIMIT 0")
        if bad:
            raise ValueError("Unmigrated view paths: " + str(bad))
        con.execute("CHECKPOINT")
    finally:
        con.close()
    shutil.copytree(old_root / "_sync", target / "_sync")
    import sqlite3
    state = sqlite3.connect(target / "_sync/state.sqlite")
    try:
        state.execute("CREATE TEMP TABLE migration_paths (old TEXT PRIMARY KEY,new TEXT)")
        state.executemany("INSERT INTO migration_paths VALUES (?,?)", list(mapping.items()))
        state.execute("DELETE FROM versions WHERE path NOT IN (SELECT old FROM migration_paths)")
        state.execute("UPDATE versions SET path=(SELECT new FROM migration_paths WHERE old=versions.path)")
        state.commit()
    finally:
        state.close()
    shutil.copyfile(old_root / "FIELD-DICTIONARY.md", target / "FIELD-DICTIONARY.md")
    atomic_json(target / "STORE.json", {"mode": "current", "schema_version": 1})
    report = {"mode": "current", "objects": len(mapping), "source": str(active["database"]), "target": str(target)}
    atomic_json(target / "_ops/migration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--resume-ledger", action="store_true")
    args = parser.parse_args()
    if args.resume_ledger:
        print(json.dumps(resume_ledger(args.source, args.target), ensure_ascii=False))
    else:
        print(json.dumps(migrate(args.source, args.target), ensure_ascii=False))


def resume_ledger(old_root, target):
    """Recover only after catalog relinking completed; never rebuild/replace data."""
    import sqlite3

    old_root, target = Path(old_root).resolve(), Path(target).resolve()
    if (target / "STORE.json").exists():
        raise ValueError("Current store is already ready")
    con = duckdb.connect(str(target / "warehouse.duckdb"), read_only=True)
    try:
        for name, sql in con.execute("SELECT view_name,sql FROM duckdb_views() WHERE internal=false").fetchall():
            if str(old_root.parent) + "/own-v" in sql:
                raise ValueError("Catalog migration is not complete")
            con.execute(f"SELECT * FROM {identifier(name)} LIMIT 0")
    finally:
        con.close()
    state_path = target / "_sync/state.sqlite"
    # Copy from the quiescent old state via SQLite backup, including any WAL.
    source = sqlite3.connect(f"file:{old_root / '_sync/state.sqlite'}?mode=ro", uri=True)
    state = sqlite3.connect(state_path)
    try:
        source.backup(state)
        mapping = []
        for (original,) in state.execute("SELECT DISTINCT path FROM versions").fetchall():
            if not original:
                continue
            key = hashlib.sha256(original.encode()).hexdigest()
            dest = target / "data" / key[:2] / (key + Path(original).suffix)
            if dest.is_file():
                mapping.append((original, str(dest)))
        state.execute("CREATE TEMP TABLE migration_paths (old TEXT PRIMARY KEY,new TEXT)")
        state.executemany("INSERT INTO migration_paths VALUES (?,?)", mapping)
        state.execute("DELETE FROM versions WHERE path NOT IN (SELECT old FROM migration_paths)")
        state.execute("UPDATE versions SET path=(SELECT new FROM migration_paths WHERE old=versions.path)")
        state.commit()
    finally:
        source.close()
        state.close()
    shutil.copyfile(old_root / "FIELD-DICTIONARY.md", target / "FIELD-DICTIONARY.md")
    (target / ".access.lock").touch()
    report = {"mode": "current", "resumed_ledger": True, "ledger_paths": len(mapping), "target": str(target)}
    atomic_json(target / "STORE.json", {"mode": "current", "schema_version": 1})
    atomic_json(target / "_ops/migration.json", report)
    return report


if __name__ == "__main__":
    main()
