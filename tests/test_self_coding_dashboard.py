import json
from unittest.mock import MagicMock

from dashboard.server import DashboardBackend, DashboardHTTPHandler
from self_coding.evaluator import KernelEvaluator
from self_coding.generator import SelfCodingEngine


def test_self_coding_engine():
    engine = SelfCodingEngine()
    kernel = engine.synthesize_vector_op("test_vec_add", "add")
    assert kernel.name == "test_vec_add"
    assert kernel.bytecode is not None
    assert len(kernel.bytecode) > 0
    assert "VADD" in kernel.source_code

    relu_kernel = engine.synthesize_activation_kernel("test_relu", "relu")
    assert relu_kernel.name == "test_relu"
    assert len(relu_kernel.bytecode) > 0


def test_kernel_evaluator():
    engine = SelfCodingEngine()
    kernel = engine.synthesize_vector_op("test_eval_add", "fadd")
    evaluator = KernelEvaluator()
    report = evaluator.evaluate_kernel(kernel, num_warps=1)
    assert report.valid is True
    assert report.cycles_elapsed > 0
    assert report.instructions_retired > 0


def test_dashboard_backend_services():
    backend = DashboardBackend()
    telem = backend.get_system_telemetry()
    assert "performance_counters" in telem
    assert telem["warp_size"] == 32

    # Test LLM generation on dashboard backend
    llm_res = backend.generate_llm("test prompt", max_tokens=4)
    assert "output_text" in llm_res
    assert llm_res["tokens_generated"] == 4

    # Test self-coding synthesis via backend
    sc_res = backend.trigger_self_coding("dashboard_add", "fadd")
    assert sc_res["valid"] is True

    # Test AGI step via backend
    agi_res = backend.trigger_agi_step("run test task")
    assert agi_res["step_num"] >= 1
    assert "thought" in agi_res


def test_dashboard_http_handler_endpoints():
    handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Test GET /api/telemetry
    handler.path = "/api/telemetry"
    handler.do_GET()

    written_data = b"".join(call.args[0] for call in handler.wfile.write.call_args_list)
    telem = json.loads(written_data.decode("utf-8"))
    assert "device_name" in telem
    assert telem["device_name"] == "zGPU Silicon SIMT Accelerator v1.0"
