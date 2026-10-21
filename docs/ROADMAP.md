# zGPU Engineering Roadmap & Next Development Milestones

This document details the prioritized engineering roadmap, architectural extensions, and next actionable capabilities for **zGPU**.

---

## 1. Executive Priority Matrix

| Phase | Strategic Domain | Focus Area | Status |
|---|---|---|---|
| **Phase 1** | **Multi-GPU & Interconnect** | NVLink/zLink Crossbar, Distributed Data Parallel (DDP), Tensor Parallelism ($TP$) | ✅ **Delivered** — `distributed/tensor_parallel.py` |
| **Phase 2** | **Silicon Tape-Out & FPGA** | Verilator C++ testbench, Xilinx Vivado bitstream synthesis, synthesis timing reports | 🔲 Planned |
| **Phase 3** | **PyTorch / ONNX Integration** | Custom PyTorch C++ `torch.ops.zgpu`, ONNX Runtime execution provider | 🟡 Partial — `torch_compat/tensor.py` |
| **Phase 4** | **FlashAttention-3 & Speculative Decoding** | Hardware tiled online softmax with draft token verification | ✅ **Delivered** — `kernels/flash_attn/`, `llm/speculative/` |
| **Phase 5** | **SuperAGI Autonomous Multi-Agent Swarm** | Peer-to-peer agent communication, shared associative memory crossbar | ✅ **Delivered** — `agi/` |
| **Phase 6** | **Network CPU Clustering & Discovery** | Heterogeneous compute, work-stealing dispatch, automated node discovery | ✅ **Delivered** — `cluster/` |
| **Phase 7** | **Self-Coding Kernel Engine** | Autonomous kernel code generation, evaluation, and evolution | ✅ **Delivered** — `self_coding/` |

---

## 2. Completed Milestones

### ✅ Phase 1: Distributed Tensor Parallelism
- All-Reduce crossbar communication primitive
- Column-parallel and row-parallel `Linear` decomposition (Megatron-style)
- Multi-device tensor sharding and gathering

### ✅ Phase 4: FlashAttention-3 & Speculative Decoding
- **FlashAttention-3**: Tiled $Q, K, V$ blocks inside 64KB shared memory, online softmax with running max/sum, eliminating DRAM round-trips
- **Speculative Decoding**: Draft model produces candidate tokens, target model verifies in parallel via single Tensor MMA pass

### ✅ Phase 5: SuperAGI Multi-Agent Swarm
- Agent core with tool execution loop and GPU-accelerated inference
- Tree-of-Thought (ToT) planner with candidate ranking
- Fast associative vector memory store
- Multi-agent swarm coordination with task decomposition

### ✅ Phase 6: Network CPU Clustering
- HTTP worker daemon (port 9200) for CPU GEMM offloading
- Work-stealing task dispatcher across heterogeneous network nodes
- Automated subnet discovery scanner (TCP probe + UDP beacon)
- NetworkPool with hybrid GPU+CPU powerup routing

### ✅ Phase 7: Self-Coding Kernel Engine
- Autonomous kernel code generator from high-level specs
- Cycle-accurate evaluator with performance metrics
- Real-time telemetry dashboard (port 9000)

---

## 3. Active Development — Next Priorities

### 3.1 Silicon Synthesis & FPGA Prototyping (Phase 2)
- **Verilator Co-Simulation**:
  - Wrap `rtl/zgpu_sm_core.sv` and `rtl/zgpu_lane.sv` with Verilator to create a high-speed C++ compiled silicon emulator.
  - Compare cycle counts between the Verilog RTL and Python cycle-accurate simulator.
- **FPGA Synthesis Harness**:
  - Target Xilinx UltraScale+ / AMD Zynq devices with AXI4 bus interfaces.
  - Generate synthesis reports (LUTs, BRAMs, DSPs, max clock frequency $F_{\max}$).

### 3.2 Full PyTorch / ONNX Integration (Phase 3)
- **Native Tensor Backend**:
  - Implement `torch.autograd.Function` bindings wrapping `runtime.runtime.ZRuntime`.
  - Complete zero-copy tensor handoff using `__zgpu_array_interface__`.
- **GGUF / Safetensors Weight Loader**:
  - Fast memory-mapped weight ingestion directly into zGPU global memory.
- **ONNX Runtime Execution Provider**:
  - Register zGPU as an ONNX Runtime backend for graph-level model inference.

### 3.3 Production Hardening
- **Continuous Profiling**: Automated regression tracking with cycle-count baselines
- **Memory Safety**: Bounds-checked device memory with guard pages
- **Error Codes**: Structured error reporting across the full stack
- **Logging**: Configurable verbosity levels for simulator, compiler, runtime

### 3.4 Extended Kernel Library
- **Grouped Query Attention (GQA)**: Multi-head to fewer KV-head mapping
- **Rotary Position Embeddings (RoPE)**: Hardware-accelerated frequency rotation
- **FP8 Tensor Core Support**: E4M3/E5M2 precision for Hopper-class workloads
- **Fused MLP Kernels**: SwiGLU + down-projection in a single kernel launch

---

## 4. Long-Term Vision

| Horizon | Capability |
|---|---|
| **H1** | Complete FPGA prototype running real RTL on Xilinx/AMD hardware |
| **H2** | PyTorch backend capable of running unmodified HuggingFace models |
| **H3** | Multi-chip scale-out with real network interconnect (RDMA/RoCE) |
| **H4** | ASIC tape-out feasibility study with standard cell libraries |
| **H5** | Browser-based GPU simulator via WebAssembly compilation |
