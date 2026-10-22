"""Correctness-first measurement of autoregressive token throughput."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationMetrics:
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    ms_per_token: float


def generation_metrics(
    prompt: list[int], output: list[int], elapsed_seconds: float, requested_tokens: int
) -> GenerationMetrics:
    """Report actual appended tokens, never requested token count as throughput.

    Requires the model output to preserve the prompt prefix and not exceed
    the requested generation budget. Timing must be finite and nonnegative.
    """
    if requested_tokens < 0:
        raise ValueError("requested_tokens must be nonnegative")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if output[: len(prompt)] != prompt:
        raise ValueError("output must preserve the prompt prefix")
    generated = len(output) - len(prompt)
    if generated > requested_tokens:
        raise ValueError("generated token count exceeds requested limit")
    if generated and elapsed_seconds == 0:
        raise ValueError("elapsed_seconds must be positive when tokens were generated")
    throughput = generated / elapsed_seconds if generated else 0.0
    latency = elapsed_seconds * 1000.0 / generated if generated else 0.0
    return GenerationMetrics(generated, elapsed_seconds, throughput, latency)
