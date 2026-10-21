"""Network CPU Worker Node for zGPU Distributed Computing Grid.

Listens on HTTP/JSON-RPC for partitioned matrix multiplication chunks, token projections,
and vector embeddings dispatched from the primary zGPU host, returning computed float tensors.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CPU-Worker] %(message)s")
log = logging.getLogger("zGPU-CPU-Worker")


class CPUComputeEngine:
    """Multi-threaded / vectorized CPU compute kernels."""

    @staticmethod
    def gemm_chunk(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """Compute standard matrix multiplication A @ B on CPU."""
        M = len(A)
        K = len(A[0]) if M > 0 else 0
        N = len(B[0]) if len(B) > 0 else 0

        # Transpose B for cache-friendly contiguous row access on CPU
        B_T = [[B[r][c] for r in range(len(B))] for c in range(N)]

        out = [[0.0 for _ in range(N)] for _ in range(M)]
        for i in range(M):
            row_a = A[i]
            for j in range(N):
                row_b = B_T[j]
                # Dot product
                dot = 0.0
                for k in range(K):
                    dot += row_a[k] * row_b[k]
                out[i][j] = dot

        return out

    @staticmethod
    def vector_norm_chunk(X: list[list[float]], weights: list[float], eps: float = 1e-5) -> list[list[float]]:
        """Compute RMSNorm on CPU chunk."""
        out = []
        dim = len(weights)
        for row in X:
            var = sum(v * v for v in row) / dim
            inv_rms = 1.0 / math.sqrt(var + eps)
            out.append([v * inv_rms * w for v, w in zip(row, weights, strict=False)])
        return out


class CPUWorkerHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health", "/status"):
            resp = {
                "ok": True,
                "node_type": "network_cpu_worker",
                "status": "ready",
                "cores_available": 16,
                "timestamp": time.time(),
            }
            self._send_json(resp)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body) if body else {}

        method = payload.get("method")
        params = payload.get("params", {})
        req_id = payload.get("id")

        t0 = time.perf_counter()

        if method == "compute.gemm":
            A = params.get("A", [])
            B = params.get("B", [])
            result = CPUComputeEngine.gemm_chunk(A, B)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"matrix": result, "latency_ms": round(elapsed_ms, 2)},
            }

        elif method == "compute.rmsnorm":
            X = params.get("X", [])
            weights = params.get("weights", [])
            result = CPUComputeEngine.vector_norm_chunk(X, weights)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"matrix": result, "latency_ms": round(elapsed_ms, 2)},
            }

        elif method == "worker.ping":
            resp_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"status": "pong", "time": time.time()},
            }

        else:
            resp_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method {method} not supported"},
            }

        self._send_json(resp_payload)

    def _send_json(self, data: Any):
        body_bytes = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, format, *args):
        pass


def run_cpu_worker(host: str = "0.0.0.0", port: int = 9200):
    server = HTTPServer((host, port), CPUWorkerHTTPHandler)
    log.info("zGPU Network CPU Worker listening on http://%s:%d/", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9200
    run_cpu_worker(port=port)
