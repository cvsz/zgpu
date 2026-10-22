"""Validate an external code-agent refactor proposal before manual review.

This tool never runs generated code, calls a model, or merges a pull request.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ALLOWED_AREAS = frozenset({
    "isa", "compiler", "simulator", "runtime", "kernels", "llm",
    "cluster", "distributed", "tests", "docs", "self_coding",
})
REQUIRED_GATES = ("ruff_check", "ruff_format", "pytest", "codeql", "independent_review")
MAX_BYTES = 65536


def validate_contract(payload: object) -> list[str]:
    """Return errors for an untrusted proposed refactor contract."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["contract must be a JSON object"]
    if payload.get("schema") != "zgpu.refactor.v1":
        errors.append("unsupported schema")
    if payload.get("target_repo") != "cvsz/zgpu":
        errors.append("target_repo must be cvsz/zgpu")
    if payload.get("mode") != "proposal-only":
        errors.append("mode must be proposal-only")
    areas = payload.get("areas")
    if not isinstance(areas, list) or not areas or any(
        not isinstance(area, str) or area not in ALLOWED_AREAS for area in areas
    ):
        errors.append("areas must be a nonempty subset of supported directories")
    elif len(set(areas)) != len(areas):
        errors.append("duplicate areas")
    for field in ("objective", "rollback"):
        val = payload.get(field)
        if not isinstance(val, str) or not 1 <= len(val.strip()) <= 2000:
            errors.append(f"{field} must contain 1..2000 characters")
    gates = payload.get("gates")
    if not isinstance(gates, dict) or any(gates.get(name) is not True for name in REQUIRED_GATES):
        errors.append("all required gates must be set to true")
    if payload.get("auto_merge") is not False:
        errors.append("auto_merge must be false")
    if payload.get("network_execution") is not False:
        errors.append("network_execution must be false")
    name = payload.get("task_id")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", name):
        errors.append("task_id must be a lowercase slug (3..64 characters)")
    return errors


def load_contract(path: Path) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("refactor contract exceeds 64 KiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(data)
    if errors:
        raise ValueError("; ".join(errors))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate zGPU external code-agent refactor proposals")
    parser.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    try:
        data = load_contract(args.contract)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Invalid refactor contract: {exc}\n")
    print(f"Valid proposal contract: {data['task_id']} (NOT approved for execution or merge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
