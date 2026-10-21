# CLAUDE.md — zGPU Project Instructions for Claude Code

> This file provides project-specific instructions for Claude Code (claude-code CLI and Anthropic Claude agents) when working on the zGPU codebase.

## Project Overview

**zGPU** is a clean-room, full-stack synthesizable SIMT GPU architecture. The entire stack — from cycle-accurate hardware simulator to LLM inference server — is implemented in pure Python with SystemVerilog RTL for synthesis. It features a custom 32-bit ISA (`zISA`), GTO warp scheduler, 4×4×4 tensor MMA cores, JIT compiler, and high-throughput JSON-RPC server.

## Mandatory Rules

### 1. Zero Broken Tests
Every change must maintain 100% green on the test suite. Always verify:
```bash
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v
```
Do NOT mark a task as complete without running this command.

### 2. No Unverified Claims
Never assert that code works, a speedup is real, or a bug is fixed unless you have executed the relevant tests or benchmarks. If you cannot run code, state that explicitly.

### 3. Pure Python — No Heavy Dependencies
All core modules (`simulator/`, `compiler/`, `runtime/`, `kernels/`, `nn/`, `llm/`, `isa/`) must execute with the Python 3 standard library alone. External binary wheels (numpy, torch, etc.) are forbidden as required imports in core code — optional/fallback usage only.

### 4. Preserve Documentation
Do not delete or rewrite comments, docstrings, or documentation that are unrelated to your changes.

## Architecture Invariants

These are hard constraints — do not violate them:

- **Tensor Core tiles**: 4×4 matrix dimensions (multiples of 4 only)
- **SIMT warp width**: 32 threads in lockstep with dynamic divergence stack masking
- **ISA encoding**: 32-bit fixed-width instruction words
- **Shared memory**: 64KB, 32-banked

## Repository Structure

```
zgpu/
├── rtl/            # Synthesizable SystemVerilog (SIMT core, ALU lane, divergence stack)
├── isa/            # 32-bit instruction codec, bitfield packing, two-pass assembler
├── simulator/      # Cycle-accurate execution — warp scheduling, memory, registers, tensor units
├── kernels/        # SIMT & MMA kernels (GEMM, RMSNorm, FlashSoftmax, SwiGLU)
├── nn/             # Neural network layers (Linear, projections via Tensor Core tiles)
├── llm/            # Transformer decoder, causal attention, generation, JSON-RPC server
├── rasterizer/     # Triangle rasterizer, barycentric interpolation, Early-Z ROPs
├── compiler/       # AST → vectorized zISA code generator (JIT)
├── runtime/        # Host driver — malloc, memcpy_h2d/memcpy_d2h, grid dispatch
├── benchmarks/     # Throughput, IPC, latency benchmarks
├── tests/          # Comprehensive pytest verification suite
├── docs/           # Architecture spec, ISA reference, memory docs, roadmap
├── dashboard/      # Real-time telemetry & control dashboard
├── distributed/    # Distributed GPU cluster support
├── cluster/        # Cluster discovery & orchestration
├── self_coding/    # Self-coding engine
├── agi/            # AGI acceleration modules
└── torch_compat/   # PyTorch compatibility layer
```

## Common Commands

```bash
# Full test suite (MUST pass before any PR/commit)
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v

# Run silicon benchmarks
PYTHONPATH=$(pwd) python3 benchmarks/run_benchmarks.py

# Run LLM acceleration benchmark
PYTHONPATH=$(pwd) python3 benchmarks/benchmark_llm.py

# Start JSON-RPC LLM inference server on port 8880
PYTHONPATH=$(pwd) python3 llm/server.py 8880

# Lint & format check
ruff check .
ruff format --check .
```

## Code Conventions

- **Python**: PEP 8, type hints where practical, docstrings on public APIs
- **Commits**: Conventional Commits — `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `ci:`
- **SystemVerilog**: `logic` over `reg`/`wire`, port prefixes `i_`/`o_`, parameterized designs
- **Branches**: `feat/`, `fix/`, `docs/`, `refactor/`, `perf/`, `test/`

## Reference Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Microarchitecture specification
- [`docs/ISA_REFERENCE.md`](docs/ISA_REFERENCE.md) — zISA instruction set reference
- [`docs/MEMORY_HIERARCHY.md`](docs/MEMORY_HIERARCHY.md) — Memory hierarchy design
- [`docs/LLM_AND_TENSOR_ENGINE.md`](docs/LLM_AND_TENSOR_ENGINE.md) — LLM & tensor acceleration
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Project roadmap

## Anti-Patterns — Avoid These

- Adding `numpy`/`torch`/`scipy` as required imports in core modules
- Changing tensor tile size from 4×4 without explicit user approval
- Modifying the 32-bit ISA word width
- Claiming tests pass without actually running them
- Removing existing tests or documentation without clear justification
- Making changes to `rtl/` without understanding the synthesis implications
