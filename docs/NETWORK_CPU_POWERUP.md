# zGPU Heterogeneous Network CPU Clustering & Powerup Specification

This document details the **Network CPU Compute Pool** and **Work-Stealing Dispatcher** that harnesses network-connected CPUs to power up and augment local **zGPU** processing capacity.

---

## 1. Architectural Concept: Heterogeneous Compute Powerup

While local zGPU SIMT cores and Tensor MMA units excel at dense matrix tiles, network-connected CPU nodes can concurrently execute parallel row slices of large GEMM, RMSNorm, and token projection workloads:

```
                          +-----------------------------------+
                          |      zGPU Host Application        |
                          +-----------------+-----------------+
                                            |
                              +-------------v-------------+
                              |   NetworkCPUPool Manager  |
                              |   (Work-Stealing / Slice) |
                              +-------+-------------------+
                                      |
              +-----------------------+-----------------------+
              |                                               |
+-------------v--------------+                 +--------------v-------------+
| Local zGPU Tensor Cores    |                 | Network CPU Workers        |
| • 4x4x4 MMA Silicon Tile   |                 | (HTTP / JSON-RPC Port 9200)|
| • Hardware Shared Memory   |                 | • Multi-threaded SIMD      |
| • Fast Tile Accumulation   |                 | • High-throughput Cache    |
+----------------------------+                 +----------------------------+
```

---

## 2. Key Components

### 2.1 Network CPU Worker Service (`cluster/worker/cpu_worker.py`)
- Zero-dependency HTTP JSON-RPC 2.0 daemon listening on port `9200`.
- Endpoints:
  - `compute.gemm`: Executes chunk matrix multiplication ($A_{\text{chunk}} \times B$).
  - `compute.rmsnorm`: Computes vectorized Root-Mean-Square normalization.
  - `worker.ping`: Verifies node health and measures round-trip network latency.

### 2.2 Network CPU Pool & Hybrid Dispatcher (`cluster/network_pool.py`)
- **Node Fleet Registry**: Registers and health-checks available network worker IPs (`register_node`, `ping_nodes`).
- **Heterogeneous Work-Stealing**:
  - Partitions $M \times K$ input matrices dynamically between local zGPU Tensor Cores and network workers based on configured ratio `gpu_fraction`.
  - Automatically aligns GPU slice boundaries to $4 \times 4$ tile multiples for hardware Tensor MMA compatibility.
  - Fail-safe fallback: automatically re-routes execution to local zGPU if a remote network node becomes unresponsive.

---

## 3. Benchmarks & Speedup Evidence

Measured using 32×32 GEMM in [`benchmarks/benchmark_cpu_powerup.py`](file:///home/cvsz/zGPU/benchmarks/benchmark_cpu_powerup.py):
- **Baseline Pure zGPU (32×32 GEMM)**: `69.12 ms`
- **Heterogeneous Hybrid (50% GPU + 50% CPU)**: **`39.06 ms`** (**~1.77× Speedup**)
- **Precision Validation**: 100% numerical match with exact analytical float outputs.

---

## 4. Launching the Network CPU Worker

To run a remote CPU worker node on any local or network machine:
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/cluster/worker/cpu_worker.py 9200
```
