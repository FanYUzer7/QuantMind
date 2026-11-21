"""One mutable warehouse. Readers share a lock; updates commit one transaction."""

from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from .warehouse import identifier, literal


@contextmanager
def access(root, *, write=False):
    root = Path(root)
    lock = root / ".access.lock"
    handle = lock.open("rb" if not write and lock.exists() else "a")
    try:
        try:
            fcntl.flock(
                handle,
                (fcntl.LOCK_EX if write else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Owned data is busy; finish the active query/update before retrying"
            ) from exc
        yield
    finally:
        handle.close()


def is_current(root):
    return (Path(root) / "STORE.json").is_file()


def owned_read(function):
    """Keep a whole synchronous loading/backtest operation coherent with updates."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        import os
        root = os.getenv("QM_OWN_WAREHOUSE_ROOT", "")
        if root and is_current(root):
            with access(root):
                return function(*args, **kwargs)
        return function(*args, **kwargs)
    return wrapped


def sources(connection):
    exists = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='own_sources'"
    ).fetchone()[0]
    if not exists:
        return {"manifests": {}, "entries": []}
    row = connection.execute("SELECT payload FROM own_sources").fetchone()
    return json.loads(row[0]) if row else {"manifests": {}, "entries": []}


def save_sources(connection, payload):
    connection.execute("CREATE TABLE IF NOT EXISTS own_sources (payload VARCHAR)")
    connection.execute("DELETE FROM own_sources")
    connection.execute("INSERT INTO own_sources VALUES (?)", [json.dumps(payload)])


def object_changes(previous, entries, datasets):
    old = {
        (d, o["key"]): o
        for d, _, o in previous.get("entries", [])
        if d in datasets
    }
    new = {(d, o["key"]): o for d, _, o in entries}
    from .sync import fingerprint

    return {
        "added": len(new.keys() - old.keys()),
        "removed": len(old.keys() - new.keys()),
        "revised": sum(
            fingerprint(old[k]) != fingerprint(new[k]) for k in old.keys() & new.keys()
        ),
    }


def clear_dataset(builder, name):
    # Preserve the schema so dependent canonical views remain valid, with zero rows.
    schema = builder.con.execute(f"DESCRIBE {identifier(name)}").fetchall()
    columns = ",".join(
        f"CAST(NULL AS {row[1]}) AS {identifier(row[0])}" for row in schema
    )
    builder.con.execute(
        f"CREATE OR REPLACE VIEW {identifier(name)} AS SELECT {columns} WHERE false"
    )
    builder.con.execute(
        "UPDATE catalog_tables SET files_json='[]', row_count=0 WHERE table_name=?",
        [name],
    )


def prune_superseded(root, connection, state):
    """Collect managed files no longer referenced by data or an unfinished plan."""
    live = set()
    for (encoded,) in connection.execute("SELECT files_json FROM catalog_tables").fetchall():
        live.update(json.loads(encoded))
    live.update(r[0] for r in connection.execute("SELECT object_path FROM catalog_files").fetchall())
    live.update(r[0] for r in connection.execute("SELECT parquet_path FROM catalog_members").fetchall())
    for plan_path in (root / "_ops/runs").glob("*/plan.json"):
        for dataset, _, obj in json.loads(plan_path.read_text()).get("entries", []):
            row = state.lookup(dataset, obj)
            if row:
                live.add(row[0])
    removed = 0
    for base in (root / "data", root / "objects/quantdb"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file() or path.name.endswith((".proof.json", ".part")):
                continue
            if str(path) not in live:
                path.unlink()
                path.with_suffix(".proof.json").unlink(missing_ok=True)
                state.db.execute("DELETE FROM versions WHERE path=?", [str(path)])
                removed += 1
    state.db.commit()
    return removed


def update_current(root, run, plan, state, eastmoney, *, drop_datasets=()):
    """Apply a complete manifest per selected dataset. No release or Qlib output."""
    from backend.scripts.build_own_warehouse import Builder
    from .publication import research_views
    from .sync import atomic_json

    root = Path(root).resolve()
    if not is_current(root):
        raise ValueError("Migrate to the current warehouse before updating")
    (root / "_ops").mkdir(exist_ok=True)
    if not Path(eastmoney).is_dir():
        raise ValueError("Saved Eastmoney source directory is unavailable")
    selected = {d: [] for d in plan["manifests"]}
    seen = set()
    for dataset, _, obj in plan["entries"]:
        if dataset not in selected:
            raise ValueError("Plan entry has no dataset manifest")
        key = (dataset, obj["key"])
        if key in seen:
            raise ValueError("Duplicate logical object in update plan")
        seen.add(key)
        row = state.lookup(dataset, obj)
        if not row or row[2] != "complete":
            raise ValueError("Planned object is not complete")
        path = Path(row[0])
        if not path.is_file() or path.stat().st_size != obj["size"]:
            raise ValueError("Planned object missing or corrupt")
        selected[dataset].append((obj, path))
    if set(drop_datasets) & set(selected):
        raise ValueError("Cannot update and drop the same dataset")
    with access(root, write=True):
        b = Builder(root, Path("/no-offline-import"), Path(eastmoney))
        try:
            b.con.execute("BEGIN TRANSACTION")
            previous = sources(b.con)
            changes = object_changes(
                previous, plan["entries"], set(selected) | set(drop_datasets)
            )
            for dataset in [*selected, *drop_datasets]:
                name = "quantdb_" + dataset
                files = [str(p) for _, p in selected.get(dataset, [])]
                old = b.con.execute(
                    "SELECT files_json FROM catalog_tables WHERE table_name=?", [name]
                ).fetchone()
                if not files:
                    if old:
                        clear_dataset(b, name)
                    continue
                if old and sorted(json.loads(old[0])) == sorted(files):
                    continue
                # Every changed partition is checked, including historical revisions.
                old_paths = set(json.loads(old[0])) if old else set()
                for obj, path in selected[dataset]:
                    if str(path) in old_paths:
                        continue
                    rel = f"read_parquet({literal(path)})"
                    cols = {r[0] for r in b.con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
                    if "symbol" in cols and obj.get("trade_date"):
                        counts = b.con.execute(
                            f"SELECT count(*),count(DISTINCT symbol) FROM {rel}"
                        ).fetchone()
                        if counts[0] != counts[1]:
                            raise ValueError("Duplicate symbols in revised daily partition")
                        date_key = next((c for c in ("time", "date", "trade_date") if c in cols), None)
                        if date_key:
                            wrong = b.con.execute(
                                f"SELECT count(*) FROM {rel} WHERE try_cast({identifier(date_key)} AS DATE) IS DISTINCT FROM ?::DATE",
                                [obj["trade_date"]],
                            ).fetchone()[0]
                            if wrong:
                                raise ValueError("Partition rows disagree with the manifest date")
                    if dataset in {"daily_unadjusted", "etf_kline"}:
                        ratio = b.con.execute(
                            f"SELECT median(close*volume/amount) FROM {rel} WHERE close>0 AND volume>0 AND amount>0"
                        ).fetchone()[0]
                        if ratio is not None and not 1000 <= ratio <= 100000:
                            raise ValueError("QuantDB volume/amount unit check failed")
                b.con.execute("DELETE FROM catalog_fields WHERE table_name=?", [name])
                b.publish(name, files, "quantdb", "Current dataset; replaced on validated manual updates")
            b.eastmoney_import(incremental=True)
            research_views(b, root)
            untouched = [e for e in previous.get("entries", []) if e[0] not in set(selected) | set(drop_datasets)]
            manifests = {**previous.get("manifests", {}), **plan["manifests"]}
            for dataset in drop_datasets:
                manifests.pop(dataset, None)
            save_sources(b.con, {"manifests": manifests, "entries": untouched + plan["entries"]})
            b.con.execute("COMMIT")
            b.con.execute("CHECKPOINT")
            # Cleanup is maintenance after the successful commit. Its failure must
            # not misreport a committed data update as rolled back.
            try:
                pruned = prune_superseded(root, b.con, state)
                from .warehouse import export_dictionary
                export_dictionary(b.con, root / "FIELD-DICTIONARY.md")
                cleanup_error = None
            except Exception as exc:
                pruned, cleanup_error = 0, type(exc).__name__
        except BaseException:
            try:
                b.con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            b.con.close()
    report = {
        "status": "complete", "mode": "current", "run_id": Path(run).name,
        "updated_at": datetime.now(timezone.utc).isoformat(), **changes,
        "qlib_generated": False,
        "superseded_objects_removed": pruned, "cleanup_error": cleanup_error,
    }
    atomic_json(root / "_ops/last-update.json", report)
    return report
