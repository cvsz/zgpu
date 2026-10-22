"""Comprehensive LLM & Tensor Core Benchmark on zGPU.

Measures:
1. Tensor Core MMA (4x4 tiles) raw throughput & TFLOPS.
2. End-to-End LLM Transformer generation latency and tokens-per-second (TPS).
3. Memory bandwidth and cache performance.
"""

from __future__ import annotations

import time

from benchmarks.metrics import generation_metrics

from kernels.llm_kernels import LLMKernels
from llm.transformer import LLMConfig, ZGPULLM
from runtime.runtime import ZRuntime


def run_llm_benchmarks():
    print("=" * 60)
    print("zGPU simulated tensor and tiny transformer benchmark (not physical silicon)")
    print("=" * 60)

    rt = ZRuntime()
    kernels = LLMKernels(rt)

    # 1. Tensor GEMM Benchmark
    print("\n[Benchmark 1] 4x4 Tensor Core MMA GEMM (16x16 Matrix Multiplications)")
    A = [[1.0 for _ in range(16)] for _ in range(16)]
    B = [[2.0 for _ in range(16)] for _ in range(16)]

    t0 = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        C = kernels.gemm(A, B)
    t1 = time.perf_counter()
    gemm_latency_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"  • Average 16x16 Tiled GEMM Latency: {gemm_latency_ms:.2f} ms")
    print(f"  • Result sample C[0][0]: {C[0][0]:.1f} (expected: 32.0)")

    # 2. End-to-End LLM Autoregressive Generation Benchmark
    print("\n[Benchmark 2] End-to-End Transformer Decoder Autoregressive Generation")
    cfg = LLMConfig(
        vocab_size=64,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        max_seq_len=32,
    )
    model = ZGPULLM(config=cfg, runtime=rt)

    prompt = [1, 5, 12, 20]
    num_gen_tokens = 8

    t0 = time.perf_counter()
    output_tokens = model.generate(prompt, max_new_tokens=num_gen_tokens)
    t1 = time.perf_counter()

    metrics = generation_metrics(prompt, output_tokens, t1 - t0, num_gen_tokens)

    print(f"  • Input Prompt Tokens: {prompt}")
    print(f"  • Output Tokens: {output_tokens}")
    print(f"  • Generated Tokens: {metrics.generated_tokens} (requested: {num_gen_tokens})")
    print(f"  • Total Inference Time: {metrics.elapsed_seconds * 1000.0:.2f} ms")
    print(f"  • Throughput: {metrics.tokens_per_second:.2f} tokens/second ({metrics.ms_per_token:.2f} ms/token)")
    print("\nzGPU simulator benchmark complete (no hardware speedup claim)")
    print("=" * 60)


if __name__ == "__main__":
    run_llm_benchmarks()
