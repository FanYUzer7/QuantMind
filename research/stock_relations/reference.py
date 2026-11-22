"""Load the hash-pinned v1 input/backtest adapter from its local paper record.

This intentionally references existing research inputs and archived source rather
than duplicating datasets or silently changing the experiment's preprocessing.
"""

import hashlib
import importlib
import sys


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_reference(workspace, cfg):
    code = workspace / "experiments/paper-records" / cfg["reference_run"] / "code"
    for name, expected in cfg["reference_code_sha256"].items():
        if sha256(code / name) != expected:
            raise ValueError(f"Reference source checksum mismatch: {name}")
    sys.path.insert(0, str(code))
    runner = importlib.import_module("stock_relation_research")
    for name in (
        "factor_research",
        "factor_correlation_research",
        "model_portfolio_backtest",
        "paper_records",
        "stock_relation_research",
    ):
        module = importlib.import_module(name)
        if module.__file__ is None or not str(module.__file__).startswith(
            str(code) + "/"
        ):
            raise ValueError(f"Unexpected imported research source: {name}")
        module.ROOT = workspace
    return runner
