# zGPU Network GPU & CPU Auto-Discovery & Powerup

This document specifies the **Automated Network Compute Discovery Protocol** used by **zGPU** to detect, probe, benchmark, and enlist network-accessible GPU and CPU workers.

---

## 1. Discovery Mechanisms

```
+-------------------------------------------------------------------------+
|                  zGPU Host Network Compute Scanner                      |
+-------------------------------------------------------------------------+
               |                                           |
               v (Active Subnet Probing)                   v (Passive Beacon Listener)
    +----------------------+                   +----------------------+
    |  TCP / HTTP Prober   |                   |  UDP Beacon Receiver |
    |  Ports: 8880, 9200   |                   |  Port: 9998          |
    +----------+-----------+                   +----------+-----------+
               |                                           |
               +-------------------+-----------------------+
                                   |
                   +---------------v---------------+
                   | Active Discovered Compute Map |
                   | • Network Node IP & Port      |
                   | • GPU / CPU Type & Latency    |
                   +---------------+---------------+
                                   |
                     +-------------v-------------+
                     | Enlist into Compute Pool  |
                     | for Heterogeneous Powerup |
                     +---------------------------+
```

### 1.1 Active Subnet Port Prober (`NetworkComputeScanner.scan_subnet`)
- Concurrently scans target IP ranges (e.g., `192.168.1.1` to `192.168.1.254`).
- Probes well-known zGPU ports:
  - **Port 8880**: Remote zGPU Silicon & LLM Servers (`system.health`).
  - **Port 9200**: Remote Network CPU Worker Nodes (`worker.ping`).
  - **Port 9000**: zGPU Control Telemetry Nodes.
- Validates JSON-RPC responses and measures round-trip network latency in milliseconds.

### 1.2 Passive UDP Beaconing (`ClusterBeaconBroadcaster` & `listen_beacons`)
- Remote worker nodes emit periodic UDP broadcast packets on port `9998` advertising their presence and compute capabilities without requiring manual IP configuration.
- The zGPU host automatically captures beacons and attaches workers to the execution grid.

---

## 2. Using Auto-Discovery in Python

```python
from cluster.network_pool import NetworkCPUPool

pool = NetworkCPUPool()

# Scan subnet and auto-enlist all found GPU & CPU nodes
num_found = pool.auto_discover_and_powerup(subnet_prefix="192.168.1", start_host=1, end_host=254)
print(f"Enlisted {num_found} remote compute nodes into zGPU powerup grid!")

# Execute matrix multiplication accelerated across local GPU and all discovered network CPUs
result = pool.hybrid_gemm_powerup(A, B, gpu_fraction=0.5)
```

---

## 3. Launching Standalone Discovery Scanner via CLI

```bash
python3 -c "
from cluster.discovery.scanner import NetworkComputeScanner
scanner = NetworkComputeScanner()
nodes = scanner.scan_subnet('127.0.0', 1, 10)
for n in nodes:
    print(f'Found {n.node_type} at {n.url} (latency: {n.latency_ms} ms)')
"
```
