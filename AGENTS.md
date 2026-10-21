# AGENTS.md — zGPU Architecture & Development Contract

## 1. System Identity & Mission
**zGPU** is a clean-room, full-stack synthesizable SIMT GPU architecture with:
- Cycle-accurate SystemVerilog hardware RTL (`rtl/`)
- Custom 32-bit `zISA` instruction codec & assembler (`isa/`)
- Cycle-accurate SM simulator, GTO warp scheduler, and 4x4x4 Tensor MMA Cores (`simulator/`)
- JIT Kernel Compiler (`compiler/`)
- Host Runtime Driver (`runtime/`)
- High-Performance Neural Network & LLM Acceleration Engine (`nn/`, `llm/`, `kernels/`)
- High-throughput JSON-RPC inference server (`llm/server.py`)

## 2. Core Operational Rules
- **No Unverified Claims**: Never state that a feature or speedup is verified without running benchmarks or pytest test suites.
- **Precision & Alignment**:
  - Tensor Cores require 4x4 matrix tile dimensions (multiples of 4).
  - SIMT Warps run 32 threads in lockstep with dynamic divergence stack masking.
- **Portability**: All core algorithms, compilers, and simulator code must execute with pure Python 3 standard library without requiring external heavy binary wheels unless optional.
- **Zero Broken Tests**: Every change must maintain 100% green status on `pytest /home/cvsz/zGPU/tests/`.

## 3. Directory Layout & Module Responsibilities
- `rtl/`: Synthesizable SystemVerilog (SIMT core, ALU lane, divergence stack).
- `isa/`: 32-bit instruction encoding, bitfield packing, two-pass assembler.
- `simulator/`: Microarchitectural execution model, warp scheduling, memory hierarchy, register files, and tensor units.
- `kernels/`: Vectorized SIMT & MMA kernels (GEMM, RMSNorm, Softmax, SwiGLU).
- `nn/`: High-level neural network layers (`Linear`, etc.).
- `llm/`: Transformer decoder, causal attention, autoregressive text generation, JSON-RPC server.
- `runtime/`: Host driver API, device memory allocator (`malloc`, `memcpy`), grid dispatcher.
- `compiler/`: High-level AST to vectorized `zISA` code generator.
- `benchmarks/`: Automated throughput, IPC, and latency test harnesses.
- `tests/`: Automated unit & integration verification suite.
