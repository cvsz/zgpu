"""Pure offline measurement tests: no model download or network required."""

import pytest

from benchmarks.benchmark_ollama import parse_measurement, main


def test_ollama_uses_actual_eval_counters():
    result = parse_measurement({"done": True, "eval_count": 12, "eval_duration": 2_000_000_000,
                                "prompt_eval_count": 4, "total_duration": 3_000_000_000})
    assert result["generated_tokens"] == 12
    assert result["decode_tokens_per_second"] == 6
    assert result["prompt_tokens"] == 4


@pytest.mark.parametrize("data", [
    {}, {"done": False, "eval_count": 2, "eval_duration": 50},
    {"done": True, "eval_count": True, "eval_duration": 50},
    {"done": True, "eval_count": 2, "eval_duration": -1},
    {"done": True, "eval_count": "2", "eval_duration": 1},
])
def test_rejects_incomplete_or_invalid_responses(data):
    with pytest.raises(ValueError):
        parse_measurement(data)


def test_zero_duration_avoids_division_by_zero():
    assert parse_measurement({"done": True, "eval_count": 0, "eval_duration": 0})["decode_tokens_per_second"] == 0


def test_cli_rejects_bad_limits_before_network():
    with pytest.raises(SystemExit):
        main(["--model", "tiny", "--repeats", "0"])
