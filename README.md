# zGPU: Complete End-to-End GPU Architecture & Silicon Prototype

[![CI](https://github.com/cvsz/zgpu/actions/workflows/ci.yml/badge.svg)](https://github.com/cvsz/zgpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![GitHub Release](https://img.shields.io/github/v/release/cvsz/zgpu)](https://github.com/cvsz/zgpu/releases)
[![GitHub Issues](https://img.shields.io/github/issues/cvsz/zgpu)](https://github.com/cvsz/zgpu/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/cvsz/zgpu)](https://github.com/cvsz/zgpu/pulls)

**zGPU** is a full-stack, synthesizable and cycle-accurate SIMT GPU architecture with graphics rasterization, tensor cores, silicon RTL, a JIT compiler, neural network primitives, and a hardware-accelerated Large Language Model (LLM) engine — all implemented from scratch.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         zGPU Architecture Stack                        │
├─────────────────────────────────────────────────────────────────────────┤
│  LLM Engine     │  Transformer Decoder • Causal Attention • Generation │
│  Neural Network │  Linear Layers • Projections • Tensor Core Tiles     │
│  Kernels        │  GEMM • RMSNorm • FlashSoftmax • SwiGLU • FlashAttn │
│  Compiler       │  Python AST → Vectorized zISA JIT Code Generator     │
│  Runtime        │  Host Driver • malloc • memcpy • Grid Dispatch       │
│  Simulator      │  Cycle-Accurate SM • GTO Scheduler • Memory • MMA   │
│  ISA            │  32-bit zISA Codec • Two-Pass Assembler              │
│  Hardware RTL   │  SystemVerilog • 32-Lane SIMT Core • Div Stack       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Synthesizable RTL** — SystemVerilog SIMT core with 32-lane ALU/FPU and hardware divergence stack
- **Custom ISA** — 32-bit fixed-width `zISA` instruction encoding with R/I/S/B/M-type formats
- **Cycle-Accurate Simulator** — GTO warp scheduler, 64KB 32-banked shared memory, coalesced DRAM
- **4×4×4 Tensor MMA Cores** — Hardware matrix multiply-accumulate for deep learning acceleration
- **JIT Compiler** — Python AST → vectorized zISA assembly code generator
- **LLM Engine** — Transformer decoder with causal attention, KV-cache, INT8/INT4 quantization
- **FlashAttention-3** — Tiled online softmax kernel eliminating DRAM roundtrips
- **Speculative Decoding** — Draft-and-verify acceleration for autoregressive generation
- **Graphics Pipeline** — Fixed-function triangle rasterizer with Early-Z buffer ROPs
- **Distributed Computing** — Tensor parallelism, network CPU clustering, automated node discovery
- **JSON-RPC Server** — High-throughput LLM inference API endpoint
- **Pure Python** — Entire software stack runs on Python 3.11+ standard library (no external deps)

---

## 📂 Repository Structure

| Directory | Component | Description |
|---|---|---|
| [`rtl/`](rtl/) | **Silicon Hardware** | Synthesizable SystemVerilog: 32-lane SIMT Core, ALU/FPU lane, hardware Divergence Stack |
| [`isa/`](isa/) | **Instruction Set** | 32-bit `zISA` instruction codec, bitfield packing, two-pass bytecode assembler |
| [`simulator/`](simulator/) | **Microarchitecture** | GTO Warp Scheduler, 64KB 32-banked Shared Memory, Coalesced DRAM, 4×4×4 Tensor MMA Core |
| [`kernels/`](kernels/) | **Compute Kernels** | Block-tiled GEMM, RMSNorm, FlashSoftmax, SwiGLU, FlashAttention-3 |
| [`nn/`](nn/) | **Neural Network** | `Linear` and projection layers accelerated via Tensor Core tiles |
| [`llm/`](llm/) | **LLM Engine** | Causal Self-Attention, Transformer blocks, KV-Cache, quantization, speculative decoding, JSON-RPC server |
| [`rasterizer/`](rasterizer/) | **Graphics Pipeline** | Triangle rasterizer, barycentric perspective interpolation, Early-Z buffer ROPs |
| [`compiler/`](compiler/) | **JIT Compiler** | AST code generator, CLI driver (`zcc`), Web Compiler Studio |
| [`runtime/`](runtime/) | **Host Driver** | Memory allocator (`malloc`), host↔device transfers, async grid dispatcher |
| [`distributed/`](distributed/) | **Tensor Parallelism** | Multi-device All-Reduce crossbar, column/row-parallel decomposition |
| [`cluster/`](cluster/) | **Cluster Computing** | Network CPU pool, work-stealing dispatcher, automated subnet discovery |
| [`dashboard/`](dashboard/) | **Telemetry Dashboard** | Real-time SM IPC, DRAM bandwidth, bank conflicts, live LLM inference |
| [`agi/`](agi/) | **Autonomous Agents** | SuperAGI framework, Tree-of-Thought planner, vector memory, multi-agent swarm |
| [`self_coding/`](self_coding/) | **Self-Coding Engine** | Kernel synthesis, code generation, autonomous benchmarking & evolution |
| [`torch_compat/`](torch_compat/) | **PyTorch Interop** | `ZTensor` with `__zgpu_array_interface__`, memory-mapped weight ingestion |
| [`benchmarks/`](benchmarks/) | **Benchmark Suite** | 1024-thread parallel vector add, Tensor MMA GEMM, LLM TPS benchmarks |
| [`tests/`](tests/) | **Test Suite** | Comprehensive pytest verification covering all components |
| [`docs/`](docs/) | **Documentation** | Architecture spec, ISA reference, memory hierarchy, roadmap |

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+ (no external dependencies required for core functionality)

### Run the Complete Test Suite

```bash
git clone https://github.com/cvsz/zgpu.git
cd zgpu
PYTHONPATH=$(pwd) python3 -m pytest tests/ -v
```

### Run Silicon Benchmarks

```bash
PYTHONPATH=$(pwd) python3 benchmarks/run_benchmarks.py
```

### Run LLM & Tensor Acceleration Benchmark

```bash
PYTHONPATH=$(pwd) python3 benchmarks/benchmark_llm.py
```

### Start the LLM Inference Server

```bash
PYTHONPATH=$(pwd) python3 llm/server.py 8880
```

### Compile zISA Assembly

```bash
PYTHONPATH=$(pwd) python3 compiler/cli.py kernel.zasm -o kernel.bin
```

---

## 🖥️ Web Services

| Service | Port | URL | Description |
|---|---|---|---|
| **Compiler Studio** | `9100` | `http://localhost:9100/` | Web IDE for writing, compiling, and disassembling `zISA` kernels |
| **Telemetry Dashboard** | `9000` | `http://localhost:9000/` | Live SM IPC, DRAM bandwidth, bank conflicts, LLM inference, AGI agents |
| **LLM JSON-RPC Server** | `8880` | `http://localhost:8880/rpc` | JSON-RPC 2.0 API (`system.health`, `llm.generate`) |
| **CPU Worker Node** | `9200` | `http://localhost:9200/` | Distributed CPU compute worker for cluster offloading |

---

## 🏗️ Architecture Overview

### Streaming Multiprocessor (SM)

```
┌──────────────────────────────────────────────────────┐
│            Streaming Multiprocessor (SM)              │
│                                                       │
│  Warp Pool (16 Warps) → GTO Scheduler → Decode       │
│                            │                          │
│         ┌──────────────────┼──────────────────┐       │
│         ▼                  ▼                  ▼       │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐│
│  │ 32× INT ALU  │ │ 32× FP32 FMA│ │ Tensor MMA Core││
│  │   Lanes      │ │    Lanes     │ │  (4×4×4 MMA)   ││
│  └──────────────┘ └──────────────┘ └────────────────┘│
│         │                  │                  │       │
│         ▼                  ▼                  ▼       │
│  ┌────────────────────────────────────────────────┐   │
│  │  Load/Store Unit + Memory Coalescer            │   │
│  └──────────────────┬─────────────────────────────┘   │
│                     ▼                                 │
│  ┌────────────────────────────────────────────────┐   │
│  │  64KB Shared Memory (32 Banks)                 │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       ▼
              ┌────────────────┐
              │ L2 Cache / DRAM│
              └────────────────┘
```

### Key Specifications

| Parameter | Value |
|---|---|
| Execution Model | SIMT — 32 threads/warp in lockstep |
| Tensor Core | 4×4×4 FP16→FP32 MMA |
| Shared Memory | 64KB, 32 banks |
| ISA | 32-bit fixed-width (`zISA`) |
| Register File | 128 × 32-bit per thread |
| Warp Scheduler | Greedy-Then-Oldest (GTO) |
| Divergence | Hardware stack with active mask |
| Precision | FP32, FP16, BF16, INT32, INT8, INT4 |

---

## 📖 Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Microarchitecture specification & block diagrams |
| [ISA Reference](docs/ISA_REFERENCE.md) | Complete `zISA` instruction catalog & encoding |
| [Memory Hierarchy](docs/MEMORY_HIERARCHY.md) | Register files, shared memory, L1/L2 caches, DRAM |
| [LLM & Tensor Engine](docs/LLM_AND_TENSOR_ENGINE.md) | Tensor cores, GEMM, quantization, KV-cache |
| [Compiler Service](docs/COMPILER_SERVICE.md) | JIT compiler, CLI, Web Compiler Studio |
| [Benchmarking](docs/BENCHMARKING_AND_VERIFICATION.md) | Performance counters, benchmark procedures |
| [Roadmap](docs/ROADMAP.md) | Engineering milestones & next capabilities |
| [All Documentation →](docs/INDEX.md) | Complete documentation index |

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting PRs.

See also: [Code of Conduct](CODE_OF_CONDUCT.md) · [Security Policy](SECURITY.md)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
