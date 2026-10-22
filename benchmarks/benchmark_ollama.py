"""Opt-in, localhost-only Ollama benchmark with actual server token accounting.

This measures Ollama, not zGPU acceleration. The two runtimes are not equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def parse_measurement(data: dict) -> dict:
    """Use Ollama's eval counters and nanosecond durations, not guessed token counts."""
    if data.get("done") is not True:
        raise ValueError("Ollama response is incomplete")
    count = data.get("eval_count")
    duration = data.get("eval_duration")
    if type(count) is not int or count < 0 or type(duration) is not int or duration < 0:
        raise ValueError("Invalid Ollama evaluation counters")
    tps = count * 1e9 / duration if duration > 0 else 0.0
    if not math.isfinite(tps):
        raise ValueError("Nonfinite throughput")
    return {
        "generated_tokens": count,
        "eval_duration_ns": duration,
        "decode_tokens_per_second": tps,
        "prompt_tokens": data.get("prompt_eval_count"),
        "total_duration_ns": data.get("total_duration"),
    }


def run_once(model: str, prompt: str, num_predict: int, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter_ns()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Ollama HTTP status {resp.status}")
        body = resp.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("Ollama response too large")
    wall_ns = time.perf_counter_ns() - started
    parsed = parse_measurement(json.loads(body))
    parsed["wall_duration_ns"] = wall_ns
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark an explicitly installed local Ollama model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain SIMD in one paragraph.")
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.model.strip() or not 1 <= args.num_predict <= 4096:
        parser.error("model must be nonempty and num-predict must be 1..4096")
    if not 1 <= args.repeats <= 100 or not 0 <= args.warmups <= 20:
        parser.error("repeats must be 1..100 and warmups 0..20")
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 3600:
        parser.error("timeout must be 1..3600 seconds")
    try:
        for _ in range(args.warmups):
            run_once(args.model, args.prompt, args.num_predict, args.timeout)
        samples = [run_once(args.model, args.prompt, args.num_predict, args.timeout) for _ in range(args.repeats)]
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        parser.exit(1, f"Benchmark unavailable/invalid: {exc}\n")
    result = {
        "backend": "ollama-local",
        "model": args.model,
        "requested_tokens": args.num_predict,
        "warmups": args.warmups,
        "samples": samples,
        "median_decode_tokens_per_second": statistics.median(s["decode_tokens_per_second"] for s in samples),
        "note": "Standalone Ollama reference only; no zGPU performance acceleration claim",
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
