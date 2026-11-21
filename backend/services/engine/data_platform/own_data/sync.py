"""Manifest-based, version-preserving QuantDB sync. No market-data publishing here.

Billable GETs are recorded before dispatch. Failed/uncertain GETs require explicit
retry, while successful payload proofs recover without reissuing a paid request.
API credentials never follow redirects to the CDN.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urljoin, urlparse

import pyarrow.parquet as pq
import requests

from .warehouse import sha256

HOST = "https://www.quantdb.cn"
LOCAL = threading.local()


def atomic_json(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".tmp")
    with part.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def fingerprint(obj):
    etag = str(obj.get("etag") or obj.get("sha256") or "").strip('"')
    if not etag:
        raise ValueError("Manifest needs an ETag or SHA256; size alone is unsafe")
    return hashlib.sha256(
        json.dumps([obj["key"], int(obj["size"]), etag]).encode()
    ).hexdigest()


def safe_relative(obj):
    raw = obj.get("relative_path") or obj["key"]
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError("Unsafe object path")
    return path


def sessions():
    if not hasattr(LOCAL, "api"):
        LOCAL.api, LOCAL.cdn = requests.Session(), requests.Session()
    return LOCAL.api, LOCAL.cdn


class Remote:
    def __init__(self, key):
        if not key:
            raise ValueError("QUANTDB_API_KEY is not configured")
        self.key = key

    def json(self, path, params=None):
        api, _ = sessions()
        for attempt in range(5):
            response = api.get(
                HOST + path,
                params=params,
                headers={"X-API-Key": self.key},
                allow_redirects=False,
                timeout=(15, 60),
            )
            if response.status_code != 429:
                break
            response.close()
            time.sleep(min(20, 2**attempt))
        with response:
            if response.status_code != 200:
                raise RuntimeError(f"Metadata HTTP {response.status_code}: {path}")
            return response.json()

    def usage(self):
        raw = self.json("/api/v1/auth/usage")
        sub = raw.get("subscription") or {}
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "used_traffic": int(raw.get("used_traffic") or 0),
            "traffic_limit": int(raw.get("traffic_limit") or 0),
            "package_traffic": int(raw.get("package_traffic") or 0),
            "subscription": {
                k: sub.get(k) for k in ("plan_id", "status", "started_at", "ended_at")
            },
        }

    def manifest(self, dataset, category):
        raw = self.json(
            "/api/v1/data/download/manifest",
            {
                "category_id": category,
                "sub_category": dataset,
                "layout": "auto",
            },
        )
        files = raw.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Empty/missing manifest: " + dataset)
        if raw.get("total_count", len(files)) != len(files):
            raise ValueError("Incomplete manifest: " + dataset)
        keys = set()
        for obj in files:
            if not isinstance(obj.get("size"), int) or obj["size"] <= 0:
                raise ValueError("Missing positive object size: " + dataset)
            if obj["key"] in keys:
                raise ValueError("Duplicate object key: " + dataset)
            keys.add(obj["key"])
            fingerprint(obj)
            safe_relative(obj)
        if raw.get("total_size", sum(o["size"] for o in files)) != sum(
            o["size"] for o in files
        ):
            raise ValueError("Manifest byte count mismatch: " + dataset)
        return raw

    def download(self, entry, target):
        dataset, category, obj = entry
        api, cdn = sessions()
        # No automatic retry of the authenticated (potentially billable) GET.
        response = api.get(
            HOST + "/api/v1/data/download",
            params={
                "category_id": category,
                "sub_category": dataset,
                "object_key": obj["key"],
                "layout": "auto",
            },
            headers={"X-API-Key": self.key},
            allow_redirects=False,
            stream=True,
            timeout=(15, 120),
        )
        body_url = None
        if response.status_code in (301, 302, 303, 307, 308):
            url = urljoin(HOST, response.headers.get("Location", ""))
            response.close()
            if urlparse(url).scheme != "https":
                raise RuntimeError("Invalid download redirect")
            body_url = url
            response = cdn.get(url, stream=True, timeout=(15, 180))
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        for attempt in range(3):
            written = 0
            md5 = hashlib.md5()
            digest = hashlib.sha256()
            try:
                with response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Download HTTP {response.status_code}")
                    with part.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            written += len(chunk)
                            if written > obj["size"]:
                                raise ValueError("Response exceeds planned object size")
                            md5.update(chunk)
                            digest.update(chunk)
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                break
            except requests.RequestException:
                if body_url is None or attempt == 2:
                    raise
                # Retry only the existing signed CDN URL, never the paid API GET.
                time.sleep(attempt + 1)
                response = cdn.get(body_url, stream=True, timeout=(15, 180))
        if written != obj["size"]:
            raise ValueError("Downloaded size differs from manifest")
        etag = str(obj.get("etag") or "").strip('"').lower()
        if re.fullmatch(r"[a-f0-9]{32}", etag) and etag != md5.hexdigest():
            raise ValueError("Downloaded MD5 differs from ETag")
        if obj.get("sha256") and obj["sha256"] != digest.hexdigest():
            raise ValueError("Downloaded SHA256 differs from manifest")
        pq.ParquetFile(part)
        os.replace(part, target)
        proof = {
            "fingerprint": fingerprint(obj),
            "sha256": digest.hexdigest(),
            "size": written,
        }
        atomic_json(target.with_suffix(".proof.json"), proof)
        return proof


class SyncState:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "_sync").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "_sync/state.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS versions (
            dataset TEXT, object_key TEXT, fingerprint TEXT, size INTEGER, etag TEXT,
            path TEXT, sha256 TEXT, status TEXT, error TEXT, requested_at TEXT,
            PRIMARY KEY(dataset,object_key,fingerprint))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, dataset TEXT,object_key TEXT,fingerprint TEXT,
            size INTEGER,status TEXT,error TEXT,requested_at TEXT,finished_at TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS versions_by_path ON versions(path)")
        self.db.commit()

    def close(self):
        self.db.close()

    def seed(self, plan_path, old_ledger, workspace):
        if self.db.execute("SELECT count(*) FROM versions").fetchone()[0]:
            return 0
        plan = json.loads(Path(plan_path).read_text())
        ledger = sqlite3.connect(f"file:{old_ledger}?mode=ro", uri=True)
        old = {
            r[0]: r[1:]
            for r in ledger.execute(
                "SELECT object_key,path,sha256,size,status FROM downloads"
            )
        }
        ledger.close()
        count = 0
        for dataset, _, obj in plan["entries"]:
            row = old.get(obj["key"])
            if not row or row[3] != "complete":
                continue
            path = Path(row[0])
            if not path.is_absolute():
                path = Path(workspace) / path
            if not path.is_file() or path.stat().st_size != obj["size"]:
                raise ValueError("Existing seed object is missing")
            # Previous ledger was fully SHA-verified; verify before adopting too.
            if sha256(path) != row[1]:
                raise ValueError("Existing seed object checksum mismatch")
            self.complete(dataset, obj, path, row[1], commit=False)
            count += 1
        self.db.commit()
        return count

    def complete(self, dataset, obj, path, digest, *, commit=True):
        self.db.execute(
            "INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,?,'complete',NULL,NULL)",
            [
                dataset,
                obj["key"],
                fingerprint(obj),
                obj["size"],
                obj.get("etag", ""),
                str(Path(path).resolve()),
                digest,
            ],
        )
        if commit:
            self.db.commit()

    def lookup(self, dataset, obj):
        return self.db.execute(
            "SELECT path,sha256,status FROM versions WHERE dataset=? AND object_key=? AND fingerprint=?",
            [dataset, obj["key"], fingerprint(obj)],
        ).fetchone()

    def target(self, dataset, obj):
        return self.root / "objects/quantdb" / dataset / (fingerprint(obj) + ".parquet")

    def recover(self, dataset, obj):
        target = self.target(dataset, obj)
        proof_file = target.with_suffix(".proof.json")
        if not target.exists() or not proof_file.exists():
            return False
        proof = json.loads(proof_file.read_text())
        if (
            proof.get("fingerprint") != fingerprint(obj)
            or proof.get("size") != target.stat().st_size
            or sha256(target) != proof.get("sha256")
        ):
            raise ValueError("Completed payload proof is invalid")
        self.complete(dataset, obj, target, proof["sha256"])
        return True

    def plan(self, manifests, categories, retry_failed=False):
        todo, retained, reused = [], [], 0
        for dataset, manifest in manifests.items():
            for obj in manifest["files"]:
                row = self.lookup(dataset, obj)
                if row and row[2] == "complete":
                    path = Path(row[0])
                    if not path.exists() or path.stat().st_size != obj["size"]:
                        raise ValueError(
                            "Local object missing/corrupt; explicit repair needed"
                        )
                    if sha256(path) != row[1]:
                        raise ValueError(
                            "Cached object checksum mismatch; explicit repair needed"
                        )
                    reused += 1
                elif self.recover(dataset, obj):
                    reused += 1
                elif row and not retry_failed:
                    raise RuntimeError(
                        "Failed/uncertain paid request retained; run with --retry-failed after inspecting the report"
                    )
                else:
                    todo.append((dataset, categories[dataset], obj))
                retained.append((dataset, categories[dataset], obj))
        return {
            "entries": retained,
            "todo": todo,
            "reused_objects": reused,
            "download_bytes": sum(o["size"] for _, _, o in todo),
        }

    def transfer(self, plan, remote, workers, on_progress=None):
        tasks = iter(plan["todo"])
        complete, failures = 0, []
        pending = {}

        def submit(pool):
            try:
                entry = next(tasks)
            except StopIteration:
                return False
            dataset, _, obj = entry
            self.db.execute(
                "INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,NULL,'requested',NULL,?)",
                [
                    dataset,
                    obj["key"],
                    fingerprint(obj),
                    obj["size"],
                    obj.get("etag", ""),
                    str(self.target(dataset, obj)),
                    datetime.now(timezone.utc).isoformat(),
                ],
            )
            self.db.commit()
            attempt = self.db.execute(
                "INSERT INTO attempts(dataset,object_key,fingerprint,size,status,requested_at) VALUES (?,?,?,?,'requested',?)",
                [
                    dataset,
                    obj["key"],
                    fingerprint(obj),
                    obj["size"],
                    datetime.now(timezone.utc).isoformat(),
                ],
            ).lastrowid
            self.db.commit()
            pending[pool.submit(remote.download, entry, self.target(dataset, obj))] = (
                entry,
                attempt,
            )
            return True

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers):
                if not submit(pool):
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    (dataset, _, obj), attempt = pending.pop(future)
                    try:
                        proof = future.result()
                        self.complete(
                            dataset, obj, self.target(dataset, obj), proof["sha256"]
                        )
                        self.db.execute(
                            "UPDATE attempts SET status='complete',finished_at=? WHERE id=?",
                            [datetime.now(timezone.utc).isoformat(), attempt],
                        )
                        self.db.commit()
                        complete += 1
                    except Exception as exc:
                        # Do not serialize HTTP exceptions containing signed URLs.
                        error = type(exc).__name__
                        if isinstance(exc, RuntimeError) and str(exc).startswith(
                            "Download HTTP "
                        ):
                            error += ":" + str(exc)
                        self.db.execute(
                            "UPDATE versions SET status='failed',error=? WHERE dataset=? AND object_key=? AND fingerprint=?",
                            [error, dataset, obj["key"], fingerprint(obj)],
                        )
                        self.db.commit()
                        self.db.execute(
                            "UPDATE attempts SET status='failed',error=?,finished_at=? WHERE id=?",
                            [error, datetime.now(timezone.utc).isoformat(), attempt],
                        )
                        self.db.commit()
                        failures.append(
                            {"dataset": dataset, "key": obj["key"], "error": error}
                        )
                    if on_progress:
                        on_progress(complete, len(failures))
                    # Stop issuing new billable requests after a failure.
                    if not failures:
                        submit(pool)
        if failures:
            raise RuntimeError(
                f"Download failed ({len(failures)} objects); completed versions retained"
            )
        return complete
