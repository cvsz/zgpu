import json
from unittest.mock import MagicMock, patch
import pytest

from cluster.worker.cpu_worker import CPUComputeEngine, CPUWorkerHTTPHandler
from cluster.network_pool import NetworkCPUPool, NetworkNode


def test_cpu_compute_engine_gemm():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    # C[0][0] = 1*5 + 2*7 = 19
    # C[0][1] = 1*6 + 2*8 = 22
    # C[1][0] = 3*5 + 4*7 = 43
    # C[1][1] = 3*6 + 4*8 = 50
    out = CPUComputeEngine.gemm_chunk(A, B)
    assert out[0][0] == 19.0
    assert out[0][1] == 22.0
    assert out[1][0] == 43.0
    assert out[1][1] == 50.0


def test_cpu_worker_http_handler_gemm():
    handler = CPUWorkerHTTPHandler.__new__(CPUWorkerHTTPHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    req_body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "compute.gemm",
        "params": {"A": [[1.0, 1.0]], "B": [[2.0], [2.0]]}
    }).encode("utf-8")

    handler.rfile = MagicMock()
    handler.rfile.read.return_value = req_body
    handler.headers = {"Content-Length": str(len(req_body))}
    handler.do_POST()

    written_data = b"".join(call.args[0] for call in handler.wfile.write.call_args_list)
    res = json.loads(written_data.decode("utf-8"))
    assert res["id"] == 1
    assert res["result"]["matrix"] == [[4.0]]


def test_network_pool_hybrid_powerup():
    pool = NetworkCPUPool()
    pool.register_node("node_alpha", "http://192.168.1.50:9200")

    A = [[1.0 for _ in range(8)] for _ in range(8)]
    B = [[2.0 for _ in range(8)] for _ in range(8)]

    # Mock remote worker response for CPU portion (4x8) @ (8x8) -> 4x8 of 16.0
    cpu_mock_chunk = [[16.0 for _ in range(8)] for _ in range(4)]

    with patch.object(pool, "dispatch_remote_gemm", return_value=cpu_mock_chunk):
        out = pool.hybrid_gemm_powerup(A, B, gpu_fraction=0.5)
        assert len(out) == 8
        assert len(out[0]) == 8
        # GPU half (first 4 rows) should be 16.0, CPU half (last 4 rows) should be 16.0
        for r in range(8):
            for c in range(8):
                assert pytest.approx(out[r][c], rel=1e-3) == 16.0
