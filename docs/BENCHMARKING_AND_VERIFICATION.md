# zGPU Benchmarking, Performance Counters & Verification Guide

This document describes the automated test harnesses, performance counter monitoring, and verification protocols available in **zGPU**.

---

## 1. Automated Test Suites

The testbench is structured under `tests/` using `pytest`:

```bash
# Run the complete test suite
PYTHONPATH=/home/cvsz/zGPU python3 -m pytest /home/cvsz/zGPU/tests/ -v
```

### Coverage Index:
- **`tests/test_zgpu.py`**:
  - ISA encoder and decoder validation.
  - Two-pass assembler label resolution.
  - Cycle-accurate SM execution and register state.
- **`tests/test_full_stack.py`**:
  - JIT Kernel Compiler compilation.
  - Device memory allocation (`malloc`, `memcpy_h2d`, `memcpy_d2h`).
  - Asynchronous grid dispatch.
  - End-to-end vector addition pipeline.
- **`tests/test_llm.py`**:
  - Tensor Core Linear projection layer accuracy.
  - Transformer forward pass and causal masking.
  - Autoregressive text generation.
  - JSON-RPC server methods (`system.health`, `llm.generate`).
  - INT8 and INT4 symmetric quantization and bit-packing.
  - Continuous Key-Value Cache state updates.

---

## 2. Hardware Performance Counters

The SM simulator and memory controllers maintain low-overhead performance counters:

| Counter | Metric | Meaning |
|---|---|---|
| `total_cycles` | Latency / Throughput | Clock cycles required to retire the active warps |
| `total_instructions` | Execution Count | Total SIMT instructions completed across all warps |
| `ipc` | Instructions Per Cycle | $\frac{\text{total\_instructions}}{\text{total\_cycles}}$ (Target $\ge 4.0$ on dual-issue warps) |
| `dram_reads` | Memory Ingress | Number of DRAM bus read cycles |
| `dram_writes` | Memory Egress | Number of DRAM bus write cycles |
| `shared_mem_bank_conflicts` | Contention | Total cycles stalled due to bank conflicts |

---

## 3. Running Benchmarks

### Silicon SIMT & Tensor Core Benchmark
```bash
python3 /home/cvsz/zGPU/benchmarks/run_benchmarks.py
```
- **Benchmark 1**: 1024-element parallel vector addition validating coalesced memory throughput and bank conflicts.
- **Benchmark 2**: Hardware Tensor Core 4×4 MMA GEMM precision verification.

### End-to-End LLM Acceleration Benchmark
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/benchmarks/benchmark_llm.py
```
- Measures average 16×16 tiled GEMM latency.
- Measures tokens-per-second (TPS) and per-token generation latency for Transformer decoder inference.
