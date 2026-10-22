"""Tests for the zCoder-to-zGPU refactor contract boundary."""

import json

import pytest

from scripts.validate_refactor_contract import load_contract, validate_contract


def valid():
    return {
        "schema": "zgpu.refactor.v1",
        "task_id": "compiler-parser-hardening",
        "target_repo": "cvsz/zgpu",
        "mode": "proposal-only",
        "areas": ["compiler", "tests"],
        "objective": "Refactor parsing without changing zISA semantics",
        "rollback": "Revert the scoped pull request",
        "gates": {
            "ruff_check": True,
            "ruff_format": True,
            "pytest": True,
            "codeql": True,
            "independent_review": True,
        },
        "auto_merge": False,
        "network_execution": False,
    }


def test_valid_proposal():
    assert validate_contract(valid()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auto_merge", True),
        ("network_execution", True),
        ("mode", "execute"),
        ("target_repo", "cvsz/zcoder"),
        ("areas", ["../../etc"]),
        ("task_id", "../secret"),
        ("gates", {}),
    ],
)
def test_rejects_unsafe_proposals(field, value):
    contract = valid()
    contract[field] = value
    assert validate_contract(contract)


def test_rejects_invalid_types():
    assert validate_contract([])


def test_load_rejects_oversized_contract(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        load_contract(path)


def test_load_valid_contract(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(valid()), encoding="utf-8")
    assert load_contract(path)["task_id"] == "compiler-parser-hardening"
