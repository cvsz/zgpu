# GEMINI.md — zGPU Project Instructions for Gemini

> These instructions apply to all Gemini-based coding agents (Gemini CLI, Gemini Code Assist, Gemini in IDEs) working on the zGPU codebase.

## Project Overview

**zGPU** is a clean-room, full-stack synthesizable SIMT GPU architecture implemented entirely in Python (simulator, compiler, runtime, kernels, neural network layers, LLM engine) with SystemVerilog RTL for hardware synthesis. It includes a custom 32-bit ISA (`zISA`), cycle-accurate microarchitectural simulator, JIT compiler, and a JSON-RPC LLM inference server.

## Critical Rules — Always Follow

1. **Zero Broken Tests**: Every code change MUST pass `pytest tests/` with 100% green. Run the full test suite before declaring any task complete:
   ```bash
   PYTHONPATH=$(pwd) python3 -m pytest tests/ -v
   ```

2. **No Unverified Claims**: Never state that a feature works, a speedup is achieved, or a bug is fixed without actually running the tests or benchmarks. If you cannot execute code, say so explicitly.

3. **Pure Python Portability**: All core code (`simulator/`, `compiler/`, `runtime/`, `kernels/`, `nn/`, `llm/`, `isa/`) must run with the Python 3 standard library only. No heavy external binary dependencies unless strictly optional.

4. **Preserve Existing Comments & Docstrings**: Do not remove or rewrite documentation that is unrelated to your changes.

## Architecture Constraints

| Constraint | Specification |
|---|---|
| Tensor Core tile dimensions | 4×4 (all matrix ops must use multiples of 4) |
| SIMT warp width | 32 threads in lockstep |
| Divergence handling | Dynamic divergence stack with active mask |
| ISA word size | 32-bit fixed-width encoding |
| Shared memory | 64KB, 32-banked |

## Directory Layout

| Directory | What It Contains |
|---|---|
| `rtl/` | Synthesizable SystemVerilog — SIMT core, ALU/FPU lane, divergence stack |
| `isa/` | 32-bit instruction codec, bitfield packing, two-pass assembler |
| `simulator/` | Cycle-accurate execution model — GTO warp scheduler, memory hierarchy, register files, tensor MMA units |
| `kernels/` | Vectorized SIMT & MMA kernels — GEMM, RMSNorm, FlashSoftmax, SwiGLU |
| `nn/` | High-level neural network layers (`Linear`, projections) |
| `llm/` | Transformer decoder, causal attention, autoregressive generation, JSON-RPC server |
| `rasterizer/` | Fixed-function triangle rasterizer, barycentric interpolation, Early-Z ROPs |
| `compiler/` | AST → vectorized zISA code generator (JIT) |
| `runtime/` | Host driver API — `malloc`, `memcpy_h2d`/`memcpy_d2h`, async grid dispatch |
| `benchmarks/` | Throughput, IPC, and latency benchmark harnesses |
| `tests/` | Comprehensive pytest suite covering all components |
| `docs/` | Architecture spec, ISA reference, memory hierarchy docs, roadmap |

## Key Commands

```bash
# Run all tests
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v

# Run benchmarks
PYTHONPATH=$(pwd) python3 benchmarks/run_benchmarks.py

# Run LLM benchmark
PYTHONPATH=$(pwd) python3 benchmarks/benchmark_llm.py

# Start LLM inference server
PYTHONPATH=$(pwd) python3 llm/server.py 8880

# Lint
ruff check .
```

## Code Style

- Follow PEP 8
- Use type hints where practical
- Write docstrings for public APIs
- Conventional Commits for commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `ci:`
- SystemVerilog: use `logic` over `reg`/`wire`, prefix ports with `i_`/`o_`

## Key Documentation

- Architecture: `docs/ARCHITECTURE.md`
- ISA reference: `docs/ISA_REFERENCE.md`
- Memory hierarchy: `docs/MEMORY_HIERARCHY.md`
- Roadmap: `docs/ROADMAP.md`

## What NOT To Do

- Do not introduce `numpy`, `torch`, or any heavy C-extension dependency as a required import in core modules
- Do not change tensor tile dimensions away from 4×4 without explicit approval
- Do not modify the 32-bit ISA encoding width
- Do not skip running tests and claim "everything should work"
- Do not remove or rewrite existing tests without justification
