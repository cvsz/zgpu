"""Regression tests for truthful benchmark throughput accounting."""

import pytest

from benchmarks.metrics import generation_metrics


def test_counts_actual_tokens_when_context_window_stops_generation():
    metrics = generation_metrics([1, 2, 3], [1, 2, 3, 4], 0.5, 8)
    assert metrics.generated_tokens == 1
    assert metrics.tokens_per_second == 2.0
    assert metrics.ms_per_token == 500.0


def test_zero_generated_tokens_is_not_infinite_speed():
    metrics = generation_metrics([1, 2], [1, 2], 0.0, 4)
    assert metrics.generated_tokens == 0
    assert metrics.tokens_per_second == 0
    assert metrics.ms_per_token == 0


@pytest.mark.parametrize(
    "prompt,output,elapsed,requested",
    [([1], [2], 0.5, 1), ([1], [1, 2, 3], 0.5, 1), ([1], [1], -0.5, 0),
     ([1], [1], float("nan"), 0), ([1], [1], 0.5, -1)],
)
def test_rejects_invalid_measurement(prompt, output, elapsed, requested):
    with pytest.raises(ValueError):
        generation_metrics(prompt, output, elapsed, requested)
