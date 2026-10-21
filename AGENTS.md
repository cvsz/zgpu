# AGENTS.md — zGPU Architecture & Development Contract

> Universal instructions for all AI coding agents (Codex, Copilot, Cursor, Aider, Devin, etc.) working on the zGPU codebase.
> For tool-specific instructions, see also: [GEMINI.md](GEMINI.md) · [CLAUDE.md](CLAUDE.md) · [OPENCODE.md](OPENCODE.md)

---

## 1. System Identity & Mission

**zGPU** is a clean-room, full-stack synthesizable SIMT GPU architecture with:
- Cycle-accurate SystemVerilog hardware RTL (`rtl/`)
- Custom 32-bit `zISA` instruction codec & assembler (`isa/`)
- Cycle-accurate SM simulator, GTO warp scheduler, and 4x4x4 Tensor MMA Cores (`simulator/`)
- JIT Kernel Compiler (`compiler/`)
- Host Runtime Driver (`runtime/`)
- High-Performance Neural Network & LLM Acceleration Engine (`nn/`, `llm/`, `kernels/`)
- FlashAttention-3 tiled kernel (`kernels/flash_attn/`)
- Speculative Decoding engine (`llm/speculative/`)
- Fixed-function Graphics Rasterizer (`rasterizer/`)
- Distributed Tensor Parallelism & Network CPU Clustering (`distributed/`, `cluster/`)
- SuperAGI Autonomous Agent Framework (`agi/`)
- Self-Coding Kernel Generator & Evaluator (`self_coding/`)
- PyTorch-compatible ZTensor interface (`torch_compat/`)
- Real-time Telemetry Dashboard (`dashboard/`)
- High-throughput JSON-RPC inference server (`llm/server.py`)

---

## 2. Core Operational Rules

### 2.1 Zero Broken Tests
Every change **must** maintain 100% green status on the full test suite:
```bash
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v
```
**Never** declare a task complete without running this command.

### 2.2 No Unverified Claims
Never state that a feature or speedup is verified without running benchmarks or pytest test suites. If you cannot execute code, state that explicitly.

### 2.3 Pure Python Portability
All core algorithms, compilers, and simulator code must execute with pure Python 3 standard library without requiring external heavy binary wheels unless optional.

### 2.4 Precision & Alignment
- Tensor Cores require **4×4** matrix tile dimensions (multiples of 4).
- SIMT Warps run **32 threads** in lockstep with dynamic divergence stack masking.
- ISA encoding is **32-bit** fixed-width.
- Shared memory is **64KB, 32-banked**.

### 2.5 Preserve Documentation
Do not remove or rewrite comments, docstrings, or documentation unrelated to your changes.

---

## 3. Directory Layout & Module Responsibilities

| Directory | Responsibility |
|---|---|
| `rtl/` | Synthesizable SystemVerilog (SIMT core, ALU lane, divergence stack) |
| `isa/` | 32-bit instruction encoding, bitfield packing, two-pass assembler |
| `simulator/` | Microarchitectural execution model, warp scheduling, memory hierarchy, register files, tensor units |
| `kernels/` | Vectorized SIMT & MMA kernels (GEMM, RMSNorm, Softmax, SwiGLU) |
| `kernels/flash_attn/` | FlashAttention-3 tiled online softmax kernel |
| `nn/` | High-level neural network layers (`Linear`, projections) |
| `llm/` | Transformer decoder, causal attention, autoregressive text generation, JSON-RPC server |
| `llm/speculative/` | Speculative decoding engine (draft + verify) |
| `rasterizer/` | Fixed-function triangle rasterizer, barycentric interpolation, Early-Z ROPs |
| `compiler/` | High-level AST to vectorized `zISA` code generator, CLI (`zcc`), Web Studio |
| `runtime/` | Host driver API, device memory allocator (`malloc`, `memcpy`), grid dispatcher |
| `distributed/` | Tensor Parallelism, All-Reduce crossbar, multi-device decomposition |
| `cluster/` | Network CPU pool, work-stealing dispatcher, subnet discovery scanner |
| `dashboard/` | Real-time telemetry & control web dashboard |
| `agi/` | SuperAGI framework — agent core, ToT planner, vector memory, multi-agent swarm |
| `self_coding/` | Autonomous kernel code generator and cycle-accurate evaluator |
| `torch_compat/` | ZTensor with `__zgpu_array_interface__`, PyTorch interop |
| `benchmarks/` | Automated throughput, IPC, and latency test harnesses |
| `tests/` | Automated unit & integration verification suite |
| `docs/` | Architecture spec, ISA reference, memory hierarchy, roadmap |

---

## 4. Key Commands

```bash
# Full test suite (MUST pass before any commit)
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v

# Silicon benchmarks
PYTHONPATH=$(pwd) python3 benchmarks/run_benchmarks.py

# LLM acceleration benchmark
PYTHONPATH=$(pwd) python3 benchmarks/benchmark_llm.py

# Start LLM JSON-RPC server
PYTHONPATH=$(pwd) python3 llm/server.py 8880

# Lint & format
ruff check .
ruff format --check .
```

---

## 5. Code Style

- **Python**: PEP 8, type hints where practical, docstrings on public APIs
- **Commits**: Conventional Commits — `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `ci:`
- **SystemVerilog**: `logic` over `reg`/`wire`, port prefixes `i_`/`o_`, parameterized modules
- **Branches**: `feat/`, `fix/`, `docs/`, `refactor/`, `perf/`, `test/`

---

## 6. Anti-Patterns — Never Do These

- Import `numpy`, `torch`, or heavy C-extension dependencies as required in core modules
- Change tensor tile dimensions from 4×4 without explicit approval
- Modify the 32-bit ISA encoding width
- Skip running tests and claim "everything should work"
- Remove or rewrite existing tests without clear justification
- Make changes to `rtl/` without understanding synthesis implications
