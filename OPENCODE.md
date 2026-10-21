# OPENCODE.md — zGPU Project Instructions for OpenCode

> Project-specific instructions for OpenCode AI agents working on the zGPU codebase.

## About This Project

**zGPU** is a clean-room, full-stack synthesizable SIMT GPU architecture built entirely in pure Python (with SystemVerilog RTL for hardware synthesis). The stack includes:

- **Hardware RTL** — Synthesizable SystemVerilog SIMT core with ALU/FPU lanes and divergence stack
- **ISA** — Custom 32-bit `zISA` instruction encoding with two-pass assembler
- **Simulator** — Cycle-accurate microarchitectural model with GTO warp scheduler, 64KB 32-banked shared memory, coalesced DRAM, and 4×4×4 tensor MMA cores
- **Kernels** — Block-tiled GEMM, RMSNorm, FlashSoftmax, SwiGLU
- **Neural Networks** — `Linear` layers and projections accelerated via tensor core tiles
- **LLM Engine** — Transformer decoder, causal self-attention, autoregressive generation
- **Rasterizer** — Fixed-function triangle rasterizer with Early-Z buffer ROPs
- **Compiler** — JIT AST → vectorized zISA code generator
- **Runtime** — Host driver with device memory allocator and async grid dispatcher
- **Server** — High-throughput JSON-RPC 2.0 inference endpoint

## Rules — Non-Negotiable

### Test-First Verification
```bash
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v
```
Every change MUST pass this. Run it. Do not skip it. Do not assume it passes.

### No Unverified Claims
If you haven't run the tests or benchmarks, do not claim that something works or is faster. Say "I have not verified this" instead.

### Pure Python Only
Core modules (`simulator/`, `compiler/`, `runtime/`, `kernels/`, `nn/`, `llm/`, `isa/`) must run with Python 3 standard library only. No required heavy binary dependencies.

### Don't Break Documentation
Preserve all existing comments and docstrings unrelated to your changes.

## Hard Constraints

| Rule | Value |
|------|-------|
| Tensor core tile size | 4×4 (multiples of 4) |
| SIMT warp width | 32 threads (lockstep + divergence stack) |
| ISA word size | 32-bit fixed-width |
| Shared memory | 64KB, 32 banks |

## Directory Map

```
rtl/            → SystemVerilog RTL (SIMT core, ALU lane, divergence stack)
isa/            → Instruction codec, bitfield packing, assembler
simulator/      → Execution model, warp scheduler, memory, registers, tensor units
kernels/        → SIMT & MMA compute kernels
nn/             → Neural network layers
llm/            → Transformer engine & JSON-RPC server
rasterizer/     → Graphics pipeline (rasterizer, depth buffer)
compiler/       → AST → zISA JIT compiler
runtime/        → Host driver API (malloc, memcpy, grid dispatch)
benchmarks/     → Performance benchmark suite
tests/          → pytest verification suite
docs/           → Architecture docs, ISA reference, roadmap
dashboard/      → Real-time telemetry dashboard
distributed/    → Multi-GPU distributed support
cluster/        → Cluster discovery & orchestration
self_coding/    → Self-coding engine
agi/            → AGI acceleration
torch_compat/   → PyTorch compatibility
```

## Essential Commands

```bash
# Tests (mandatory before every change)
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v

# Benchmarks
PYTHONPATH=$(pwd) python3 benchmarks/run_benchmarks.py
PYTHONPATH=$(pwd) python3 benchmarks/benchmark_llm.py

# LLM server
PYTHONPATH=$(pwd) python3 llm/server.py 8880

# Linting
ruff check .
ruff format --check .
```

## Style Guide

- **Python**: PEP 8, type hints, docstrings on public functions/classes
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `ci:`)
- **SystemVerilog**: `logic` not `reg`/`wire`, `i_`/`o_` port prefixes, parameterized modules
- **Branching**: `feat/`, `fix/`, `docs/`, `refactor/`, `perf/`, `test/`

## Key Docs

| Document | Path |
|----------|------|
| Architecture | `docs/ARCHITECTURE.md` |
| ISA Reference | `docs/ISA_REFERENCE.md` |
| Memory Hierarchy | `docs/MEMORY_HIERARCHY.md` |
| LLM & Tensor Engine | `docs/LLM_AND_TENSOR_ENGINE.md` |
| Roadmap | `docs/ROADMAP.md` |

## Don't Do This

- Import `numpy`, `torch`, or similar as required dependencies in core code
- Change tensor tile size from 4×4 without approval
- Change ISA encoding width from 32-bit
- Skip test verification and claim things work
- Delete tests or docs without justification
