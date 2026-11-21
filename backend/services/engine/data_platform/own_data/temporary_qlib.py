"""Task-scoped Qlib conversion. No persistent provider or daily rebuild."""

from contextlib import contextmanager
from datetime import date, timedelta
import fcntl
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .qlib_export import build_provider
from .warehouse import Warehouse


def temporary_root(root):
    base = Path(os.getenv("QM_OWN_TEMP_ROOT") or (Path(root) / "_ops"))
    folder = base / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def cleanup_abandoned(root):
    """An advisory owner lock distinguishes live jobs across containers/processes."""
    folder = temporary_root(root)
    with (folder / ".cleanup.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        for candidate in folder.glob("qlib-task-*"):
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            # Creation and ownership registration also happen under this guard.
            with (candidate / ".owner.lock").open("a") as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(candidate)


@contextmanager
def temporary_provider(root, *, symbols, start, end, history_days=366):
    if not start or not end or str(start) > str(end):
        raise ValueError("Owned Qlib tasks require an explicit valid date range")
    if history_days < 0:
        raise ValueError("history_days must be nonnegative")
    cleanup_abandoned(root)
    folder = temporary_root(root)
    with (folder / ".cleanup.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        temporary = TemporaryDirectory(prefix="qlib-task-", dir=folder)
        path = Path(temporary.name)
        owner = (path / ".owner.lock").open("a")
        fcntl.flock(owner, fcntl.LOCK_EX)
    try:
        with Warehouse(root) as warehouse:
            build_provider(
                warehouse.connection, path,
                symbols=symbols,
                start=date.fromisoformat(str(start)[:10]) - timedelta(days=history_days),
                end=end,
            )
            yield path
    finally:
        # Remove while ownership remains locked, so another collector cannot race.
        try:
            temporary.cleanup()
        finally:
            owner.close()


def task_symbols(hub, universe, benchmark):
    if isinstance(universe, (list, tuple)):
        symbols = list(universe)
    elif universe in (None, "all", "all_a"):
        return None  # explicitly requested all securities, bounded by task dates
    elif Path(str(universe)).is_file():
        symbols = [line.split()[0] for line in Path(universe).read_text().splitlines() if line.strip()]
    else:
        symbols = hub.fetch_universe_stocks(str(universe))["symbol"].tolist()
    if not symbols:
        raise ValueError("Owned Qlib task has an empty universe")
    return list(dict.fromkeys([*symbols, *([benchmark] if benchmark else [])]))
