"""Network CPU Compute Pool & Work Stealing Dispatcher for zGPU.

Aggregates multiple remote CPU compute nodes over the network.
Partitions compute workloads between local zGPU Tensor Cores and remote network CPUs
to achieve maximum aggregated throughput ("powerup" burst mode).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass

from kernels.llm_kernels import LLMKernels
from runtime.runtime import ZRuntime

log = logging.getLogger("zGPU-ClusterPool")


@dataclass
class NetworkNode:
    node_id: str
    url: str
    active: bool = True
    latency_ms: float = 0.0


class NetworkCPUPool:
    """Manages a fleet of network CPU workers to augment local zGPU compute capacity."""

    def __init__(self, runtime: ZRuntime | None = None):
        self.runtime = runtime or ZRuntime()
        self.kernels = LLMKernels(self.runtime)
        self.nodes: dict[str, NetworkNode] = {}

    def register_node(self, node_id: str, url: str):
        """Add a network CPU worker endpoint (e.g. http://192.168.1.100:9200)."""
        self.nodes[node_id] = NetworkNode(node_id=node_id, url=url.rstrip("/"))

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)

    def ping_nodes(self) -> dict[str, bool]:
        """Verify reachability of all registered network CPU workers."""
        status = {}
        for nid, node in list(self.nodes.items()):
            try:
                req_data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "worker.ping", "params": {}}).encode(
                    "utf-8"
                )
                req = urllib.request.Request(node.url, data=req_data, headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    node.latency_ms = (time.perf_counter() - t0) * 1000.0
                    node.active = "result" in res
                    status[nid] = True
            except Exception:
                node.active = False
                status[nid] = False
        return status

    def dispatch_remote_gemm(self, node_id: str, A_chunk: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """Dispatch a matrix multiplication chunk to a specific network CPU worker."""
        node = self.nodes.get(node_id)
        if not node or not node.active:
            raise RuntimeError(f"Node {node_id} is inactive or unreachable")

        req_data = json.dumps(
            {"jsonrpc": "2.0", "id": 10, "method": "compute.gemm", "params": {"A": A_chunk, "B": B}}
        ).encode("utf-8")

        req = urllib.request.Request(node.url, data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["result"]["matrix"]

    def hybrid_gemm_powerup(
        self,
        A: list[list[float]],
        B: list[list[float]],
        gpu_fraction: float = 0.5,
    ) -> list[list[float]]:
        """Hybrid heterogeneous parallel GEMM:
        Splits rows of A between local zGPU Tensor Cores and network CPU workers,
        merging the final output matrix seamlessly.
        """
        M = len(A)
        if M == 0:
            return []

        active_network_nodes = [n for n in self.nodes.values() if n.active]

        # Fallback to pure zGPU if no network CPU workers are active
        if not active_network_nodes or gpu_fraction >= 1.0:
            return self.kernels.gemm(A, B)

        # Calculate row split
        gpu_rows = max(1, int(M * gpu_fraction))
        # Align gpu_rows to multiple of 4 for Tensor Core tile alignment
        gpu_rows = (gpu_rows // 4) * 4
        if gpu_rows == 0:
            gpu_rows = min(4, M)

        a_gpu = A[:gpu_rows]
        a_network = A[gpu_rows:]

        # 1. Compute GPU slice on local zGPU Tensor Cores
        gpu_out = self.kernels.gemm(a_gpu, B)

        if not a_network:
            return gpu_out

        # 2. Compute network slice on remote CPU worker
        target_node = active_network_nodes[0]
        try:
            cpu_out = self.dispatch_remote_gemm(target_node.node_id, a_network, B)
        except Exception as e:
            log.warning("Remote worker failed (%s). Falling back to local zGPU.", e)
            cpu_out = self.kernels.gemm(a_network, B)

        # 3. Concatenate row outputs
        return gpu_out + cpu_out

    def auto_discover_and_powerup(self, subnet_prefix: str = "127.0.0", start_host: int = 1, end_host: int = 2) -> int:
        """Scan local subnet and automatically register all discovered GPU and CPU nodes."""
        from cluster.discovery.scanner import NetworkComputeScanner

        scanner = NetworkComputeScanner()
        discovered = scanner.scan_subnet(base_ip_prefix=subnet_prefix, start_host=start_host, end_host=end_host)
        for node in discovered:
            self.register_node(f"{node.node_type}_{node.ip}_{node.port}", node.url)
        return len(discovered)
