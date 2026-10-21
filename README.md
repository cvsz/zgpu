# zGPU: Complete End-to-End GPU Architecture & Silicon Prototype

**zGPU** is a full-stack, synthesizable and cycle-accurate SIMT GPU architecture with graphics rasterization, tensor cores, silicon RTL, a JIT compiler, neural network primitives, and a hardware-accelerated Large Language Model (LLM) engine.

---

## 1. Directory Layout & Components

| Directory | Component | Description |
|---|---|---|
| [`docs/`](docs/) | **Architecture Specification** | Microarchitecture spec, memory hierarchies, 32-bit `zISA` encoding, and AI Master Prompt. |
| [`rtl/`](rtl/) | **Silicon Hardware (SystemVerilog)** | Synthesizable RTL: 32-lane SIMT Core, SIMT ALU/FPU lane, and hardware Divergence Stack. |
| [`isa/`](isa/) | **Instruction Set Architecture** | Instruction codec and two-pass bytecode assembler. |
| [`simulator/`](simulator/) | **Cycle-Accurate Microarchitecture** | GTO Warp Scheduler, 64KB 32-banked Shared Memory, Coalesced DRAM, and 4x4x4 Tensor MMA Core. |
| [`kernels/`](kernels/) | **Accelerated Kernels** | Block-tiled GEMM, RMSNorm, FlashSoftmax, and SwiGLU vector operations. |
| [`nn/`](nn/) | **Neural Network Layers** | High-level `Linear` and projection layers accelerated via Tensor Core tiles. |
| [`llm/`](llm/) | **LLM & Transformer Engine** | Causal Self-Attention, Transformer decoder blocks, autoregressive generation, and JSON-RPC server. |
| [`rasterizer/`](rasterizer/) | **Graphics Rendering Pipeline** | Fixed-function triangle rasterizer, barycentric perspective interpolation, and Early-Z buffer ROPs. |
| [`compiler/`](compiler/) | **JIT Kernel Compiler** | AST code generator producing vectorized zISA assembly from high-level operations. |
| [`runtime/`](runtime/) | **Host Driver API** | Memory allocator (`malloc`), host-to-device transfers (`memcpy_h2d`/`memcpy_d2h`), and asynchronous grid dispatcher. |
| [`benchmarks/`](benchmarks/) | **Benchmark Suite** | 1024-thread parallel vector add, Tensor MMA GEMM, and End-to-End LLM TPS benchmarks. |
| [`tests/`](tests/) | **Verification Testbench** | Comprehensive pytest suite covering all components. |

---

## 2. Quickstart & Verification

### Run Complete Test Suite
```bash
PYTHONPATH=/home/cvsz/zGPU python3 -m pytest /home/cvsz/zGPU/tests/
```

### Run Silicon Benchmarks
```bash
python3 /home/cvsz/zGPU/benchmarks/run_benchmarks.py
```

### Run LLM & Tensor Acceleration Benchmark
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/benchmarks/benchmark_llm.py
```

### Start zGPU Accelerated LLM Inference Server
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/llm/server.py 8880
```

---

## 3. Web Services & Active Port URLs

| Service | Port | Local URL | Description |
|---|---|---|---|
| **zGPU Online Compiler Studio** | `9100` | **`http://localhost:9100/`** | Web IDE for writing, compiling, assembling, and disassembling `zISA` SIMT kernels with live hex word viewer. |
| **Real-Time Telemetry & Control Dashboard** | `9000` | **`http://localhost:9000/`** | Live dashboard monitoring SM IPC, DRAM bandwidth, shared memory bank conflicts, interactive LLM inference, and SuperAGI agent loops. |
| **zGPU LLM Acceleration JSON-RPC Server** | `8880` | **`http://localhost:8880/rpc`** | High-throughput JSON-RPC 2.0 API (`system.health`, `llm.generate`) for external LLM inference offloading. |
