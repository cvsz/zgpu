# zGPU Documentation Portal

Welcome to the comprehensive documentation suite for **zGPU**: a complete, synthesizable SIMT GPU architecture with silicon SystemVerilog RTL, cycle-accurate microarchitecture simulation, a JIT compiler, neural network primitives, and hardware-accelerated Large Language Model (LLM) execution.

---

## Documentation Index

### 1. Architecture & Hardware Specifications
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Executive summary, compute cluster organization, Streaming Multiprocessor (SM) microarchitecture, and block diagrams.
- **[ISA_REFERENCE.md](ISA_REFERENCE.md)**: Complete 32-bit `zISA` instruction catalog, register conventions, opcode tables, and binary word encoding formats.
- **[MEMORY_HIERARCHY.md](MEMORY_HIERARCHY.md)**: Register files, 32-bank Shared Memory arbitration, L1/L2 write-back caches, memory coalescing, and hardware divergence stack.

### 2. Deep Learning & LLM Acceleration
- **[LLM_AND_TENSOR_ENGINE.md](LLM_AND_TENSOR_ENGINE.md)**: Hardware 4x4x4 Tensor MMA Cores, block-tiled GEMM, vectorized RMSNorm, FlashSoftmax, SwiGLU, continuous KV-Cache, INT8/INT4 quantization, and JSON-RPC inference protocol.

### 3. Benchmarks & Testing
- **[BENCHMARKING_AND_VERIFICATION.md](BENCHMARKING_AND_VERIFICATION.md)**: Verification testbench index, hardware performance counters (IPC, memory reads/writes, bank conflicts), and benchmark execution procedures.

### 4. Compiler Toolchain & Web Studio
- **[COMPILER_SERVICE.md](COMPILER_SERVICE.md)**: Python AST JIT compiler, CLI driver (`zcc`), and interactive Web Compiler Studio (`http://localhost:9100/`).

### 5. Advanced Capabilities & Scaling
- **[ADVANCED_NEXT_CAPABILITIES.md](ADVANCED_NEXT_CAPABILITIES.md)**: FlashAttention-3 tiled online softmax, Speculative Decoding engine, Distributed Tensor Parallelism (TP), PyTorch/ZTensor device memory interop, and SuperAGI Swarm.

### 6. SuperAGI Autonomous Intelligence
- **[SUPER_AGI_ACCELERATION.md](SUPER_AGI_ACCELERATION.md)**: Hardware-accelerated autonomous agent architecture, fast associative vector memory, Tree-of-Thought (ToT) candidate ranking, and sub-5ms agent loop benchmarks.

### 7. Autonomous Self-Coding & Real-Time Dashboard
- **[SELF_CODING_AND_DASHBOARD.md](SELF_CODING_AND_DASHBOARD.md)**: Autonomous kernel code generator, cycle-accurate evaluator, real-time hardware performance counters, and live web dashboard.

### 8. Heterogeneous Network Compute & CPU Powerup
- **[NETWORK_CPU_POWERUP.md](NETWORK_CPU_POWERUP.md)**: Heterogeneous Network CPU clustering, work-stealing matrix dispatcher, HTTP worker daemon (Port `9200`), and 1.77× compute powerup benchmarks.

### 9. Automated Network GPU & CPU Discovery
- **[NETWORK_GPU_CPU_DISCOVERY.md](NETWORK_GPU_CPU_DISCOVERY.md)**: Automated discovery of network GPU and CPU nodes via concurrent TCP/HTTP subnet probing and UDP beacon listener.

### 10. Strategic Roadmap & Next Capabilities
- **[ROADMAP.md](ROADMAP.md)**: Prioritized engineering milestones, completed deliverables, and active development priorities including FPGA synthesis and PyTorch integration.

### 11. Governance & AI Development Contracts
- **[zgpu-master-prompt.md](zgpu-master-prompt.md)**: Universal Master Prompt for all AI coding agents working in the zGPU ecosystem.
- **[../AGENTS.md](../AGENTS.md)**: Repository agent contract, evidence states, test mandates, and operational directives.

---

## Community & Contributing

| Document | Description |
|---|---|
| [README.md](../README.md) | Project overview, quickstart, and architecture summary |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to contribute — setup, branching, testing, PR guidelines |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Contributor Covenant Code of Conduct v2.1 |
| [SECURITY.md](../SECURITY.md) | Security vulnerability reporting policy |
| [CHANGELOG.md](../CHANGELOG.md) | Version history and release notes |
| [LICENSE](../LICENSE) | MIT License |
