"""zGPU Silicon Real-Time Telemetry & Operations Dashboard Server.

Provides a live web interface for:
- Hardware performance counters (IPC, cycles, DRAM bandwidth, bank conflicts).
- Active SM warp scheduler states and SIMT lane visualizers.
- Live LLM inference test prompt submission and real-time generation benchmark.
- Self-coding kernel generator and evaluator.
- SuperAGI autonomous agent execution monitor.
- Network Compute Discovery & CPU/GPU powerup cluster manager.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from agi.agent_core import SuperAGIAgent
from agi.swarm import SuperAGISwarm
from cluster.discovery.scanner import NetworkComputeScanner
from cluster.network_pool import NetworkCPUPool
from kernels.flash_attn.flash_attention import FlashAttention3Kernel
from llm.transformer import LLMConfig, ZGPULLM
from runtime.runtime import ZRuntime
from self_coding.evaluator import KernelEvaluator
from self_coding.generator import SelfCodingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("zGPU-Dashboard")


class DashboardBackend:
    """Telemetry aggregator, cluster manager, and service manager."""

    def __init__(self):
        self.runtime = ZRuntime()
        self.llm = ZGPULLM(config=LLMConfig(d_model=16, num_layers=2), runtime=self.runtime)
        self.agent = SuperAGIAgent(d_model=16, runtime=self.runtime)
        self.swarm = SuperAGISwarm(shared_memory_dim=16, runtime=self.runtime)
        self.self_coder = SelfCodingEngine()
        self.evaluator = KernelEvaluator(self.runtime)
        self.network_pool = NetworkCPUPool(runtime=self.runtime)
        self.scanner = NetworkComputeScanner()
        self.flash_attn = FlashAttention3Kernel(d_model=16, block_size=4, runtime=self.runtime)

        # Pre-seed memory
        for i in range(16):
            self.agent.memory.add(f"init_context_{i}", [0.1 * ((i + j) % 7 - 3) for j in range(16)])

    def get_system_telemetry(self) -> Dict[str, Any]:
        sm = self.runtime.device.sms[0]
        mem = self.runtime.device.global_mem
        return {
            "timestamp": time.time(),
            "device_name": "zGPU Silicon SIMT Accelerator v1.0",
            "sm_count": len(self.runtime.device.sms),
            "warp_size": 32,
            "clock_mhz": 1200,
            "performance_counters": {
                "cycles": sm.cycle_count,
                "instructions_retired": sm.instructions_retired,
                "ipc": round(sm.instructions_retired / max(1, sm.cycle_count), 2),
                "dram_reads": mem.read_transactions * 4,
                "dram_writes": mem.write_transactions * 4,
                "bank_conflicts": sm.shared_mem.bank_conflicts,
            },
            "llm_config": {
                "d_model": self.llm.config.d_model,
                "vocab_size": self.llm.config.vocab_size,
                "layers": self.llm.config.num_layers,
                "flash_attention_enabled": True,
            },
            "cluster_status": {
                "registered_nodes": len(self.network_pool.nodes),
                "active_nodes": sum(1 for n in self.network_pool.nodes.values() if n.active),
                "nodes": [
                    {"id": n.node_id, "url": n.url, "active": n.active, "latency_ms": round(n.latency_ms, 2)}
                    for n in self.network_pool.nodes.values()
                ],
            },
            "super_agi_status": {
                "active_memory_records": self.agent.memory.count(),
                "history_steps": len(self.agent.execution_history),
                "swarm_active": len(self.swarm.agents) > 0,
            },
        }

    def generate_llm(self, prompt: str, max_tokens: int = 16) -> Dict[str, Any]:
        t0 = time.perf_counter()
        token_ids = [ord(c) % self.llm.config.vocab_size for c in prompt] or [1]
        out_tokens = self.llm.generate(token_ids, max_new_tokens=max_tokens)
        new_tokens = out_tokens[len(token_ids):]
        text_out = "".join(chr(32 + (t % 95)) for t in new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prompt": prompt,
            "output_text": text_out,
            "tokens_generated": len(new_tokens),
            "latency_ms": round(latency_ms, 2),
            "throughput_tps": round(len(new_tokens) / max(0.001, latency_ms / 1000.0), 2),
        }

    def trigger_self_coding(self, op_name: str, op_type: str) -> Dict[str, Any]:
        kernel = self.self_coder.synthesize_vector_op(op_name, op_type)
        eval_rep = self.evaluator.evaluate_kernel(kernel, num_warps=1)
        return {
            "kernel_name": kernel.name,
            "source_code": kernel.source_code,
            "bytecode_length": len(kernel.bytecode) if kernel.bytecode else 0,
            "valid": eval_rep.valid,
            "cycles": eval_rep.cycles_elapsed,
            "ipc": round(eval_rep.ipc, 2),
            "error": eval_rep.error_message,
        }

    def trigger_agi_step(self, goal: str) -> Dict[str, Any]:
        res = self.agent.step(goal)
        return {
            "step_num": res.step_num,
            "thought": res.thought,
            "action": res.action,
            "score": round(res.score, 3),
            "observation": res.observation,
            "latency_ms": round(res.latency_ms, 2),
        }

    def trigger_network_scan(self, subnet_prefix: str = "127.0.0", start: int = 1, end: int = 2) -> Dict[str, Any]:
        discovered = self.scanner.scan_subnet(base_ip_prefix=subnet_prefix, start_host=start, end_host=end)
        for node in discovered:
            self.network_pool.register_node(f"{node.node_type}_{node.ip}_{node.port}", node.url)
        return {
            "nodes_found": len(discovered),
            "nodes": [
                {"ip": n.ip, "port": n.port, "type": n.node_type, "latency_ms": n.latency_ms, "url": n.url}
                for n in discovered
            ],
        }

    def trigger_hybrid_gemm(self, size: int = 16, gpu_fraction: float = 0.5) -> Dict[str, Any]:
        A = [[1.0 for _ in range(size)] for _ in range(size)]
        B = [[2.0 for _ in range(size)] for _ in range(size)]
        t0 = time.perf_counter()
        out = self.network_pool.hybrid_gemm_powerup(A, B, gpu_fraction=gpu_fraction)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "matrix_size": f"{size}x{size}",
            "gpu_fraction": gpu_fraction,
            "latency_ms": round(elapsed_ms, 2),
            "result_sample": out[0][0] if out and out[0] else 0.0,
        }


backend = DashboardBackend()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>zGPU Silicon Control & Telemetry Dashboard</title>
    <style>
        :root {
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --accent: #58a6ff;
            --success: #2ea043;
            --warning: #d29922;
            --danger: #f85149;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
        }
        h1, h2, h3 { color: #fff; margin-top: 0; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 15px; }
        .metric-val { font-size: 28px; font-weight: bold; color: var(--accent); margin: 8px 0; }
        .metric-sub { font-size: 13px; color: #8b949e; }
        button { background: #238636; color: white; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        button:hover { background: #2ea043; }
        button.secondary { background: #1f6feb; }
        button.secondary:hover { background: #388bfd; }
        input, textarea, select { background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 8px; border-radius: 6px; width: 100%; box-sizing: border-box; margin-bottom: 10px; }
        pre { background: #000; padding: 10px; border-radius: 6px; overflow-x: auto; color: #7ee787; font-size: 12px; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; background: #21262d; color: var(--accent); }
        .badge.online { color: var(--success); }
        .node-list { list-style: none; padding: 0; margin: 10px 0; font-size: 13px; }
        .node-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ zGPU Silicon & Autonomous Engine Dashboard</h1>
            <div class="metric-sub">Device: zGPU-SM-0 (32-Lane SIMT • GTO Warp Scheduler • 4x4 Tensor Cores • Network Grid)</div>
        </div>
        <div>
            <span class="badge online" id="live-indicator">LIVE TELEMETRY</span>
        </div>
    </div>

    <!-- Metrics Row 1 -->
    <div class="grid">
        <div class="card">
            <h3>Hardware Performance (SM-0)</h3>
            <div class="metric-val" id="metric-ipc">4.0 IPC</div>
            <div class="metric-sub">Total Cycles: <span id="metric-cycles">0</span> • Retired Insts: <span id="metric-insts">0</span></div>
            <div class="metric-sub">DRAM I/O: <span id="metric-dram">0 B</span> • Bank Conflicts: <span id="metric-conflicts">0</span></div>
        </div>
        <div class="card">
            <h3>Network Compute Powerup Grid</h3>
            <div class="metric-val" id="metric-cluster-nodes">0 Nodes</div>
            <div class="metric-sub">Auto-Discovery & UDP Beaconing Active</div>
            <div class="metric-sub">Active Workers: <span id="metric-active-nodes">0</span></div>
        </div>
        <div class="card">
            <h3>SuperAGI Engine & Memory</h3>
            <div class="metric-val" id="metric-agi-records">16 records</div>
            <div class="metric-sub">Vector Memory Dimension: 16-d</div>
            <div class="metric-sub">Autonomous Execution Steps: <span id="metric-agi-steps">0</span></div>
        </div>
        <div class="card">
            <h3>Hardware LLM & Acceleration</h3>
            <div class="metric-val">16-d Transformer</div>
            <div class="metric-sub">FlashAttention-3: Enabled • Speculative Decoding</div>
            <div class="metric-sub">Quantization: INT8 & INT4 Tensor Core MMA</div>
        </div>
    </div>

    <!-- Interactive Grid 2 -->
    <div class="grid">
        <!-- Network Discovery & Powerup -->
        <div class="card">
            <h3>🌐 Network GPU & CPU Auto-Discovery</h3>
            <label>Subnet Prefix (e.g. 127.0.0 or 192.168.1):</label>
            <input type="text" id="scan-subnet" value="127.0.0" />
            <button class="secondary" onclick="scanNetwork()">Scan & Enlist Compute Nodes</button>
            <button onclick="runHybridPowerup()">⚡ Test Heterogeneous Powerup GEMM</button>
            <pre id="cluster-output">Cluster discovery results...</pre>
        </div>

        <!-- LLM Generator -->
        <div class="card">
            <h3>⚡ Live LLM Inference</h3>
            <label>Prompt:</label>
            <input type="text" id="llm-prompt" value="Hello zGPU silicon" />
            <button onclick="runLLM()">Generate on zGPU</button>
            <pre id="llm-output">Output will appear here...</pre>
        </div>

        <!-- Self Coding Kernel Synthesizer -->
        <div class="card">
            <h3>🛠️ Self-Coding Kernel Generator</h3>
            <label>Kernel Name & Operation:</label>
            <input type="text" id="kernel-name" value="vec_fmul_fast" />
            <select id="kernel-op">
                <option value="fmul">VFMUL (Vector Float Multiply)</option>
                <option value="fadd">VFADD (Vector Float Add)</option>
                <option value="fsub">VFSUB (Vector Float Sub)</option>
            </select>
            <button onclick="runSelfCoding()">Synthesize & Verify on SM</button>
            <pre id="kernel-output">Kernel synthesis results...</pre>
        </div>

        <!-- SuperAGI Autonomous Loop -->
        <div class="card">
            <h3>🧠 SuperAGI Execution Loop</h3>
            <label>Goal Target:</label>
            <input type="text" id="agi-goal" value="Optimize tensor tile pipeline" />
            <button onclick="runAGIStep()">Trigger Autonomous Step</button>
            <pre id="agi-output">AGI planning and actions...</pre>
        </div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                document.getElementById('metric-ipc').innerText = data.performance_counters.ipc + ' IPC';
                document.getElementById('metric-cycles').innerText = data.performance_counters.cycles;
                document.getElementById('metric-insts').innerText = data.performance_counters.instructions_retired;
                document.getElementById('metric-dram').innerText = (data.performance_counters.dram_reads + data.performance_counters.dram_writes) + ' B';
                document.getElementById('metric-conflicts').innerText = data.performance_counters.bank_conflicts;
                document.getElementById('metric-agi-records').innerText = data.super_agi_status.active_memory_records + ' records';
                document.getElementById('metric-agi-steps').innerText = data.super_agi_status.history_steps;
                document.getElementById('metric-cluster-nodes').innerText = data.cluster_status.registered_nodes + ' Nodes';
                document.getElementById('metric-active-nodes').innerText = data.cluster_status.active_nodes;
            } catch (e) {
                console.error(e);
            }
        }

        async function scanNetwork() {
            const subnet = document.getElementById('scan-subnet').value;
            const res = await fetch('/api/cluster/scan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({subnet: subnet, start: 1, end: 2})
            });
            const data = await res.json();
            document.getElementById('cluster-output').innerText = JSON.stringify(data, null, 2);
            fetchTelemetry();
        }

        async function runHybridPowerup() {
            const res = await fetch('/api/cluster/hybrid_gemm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({size: 16, gpu_fraction: 0.5})
            });
            const data = await res.json();
            document.getElementById('cluster-output').innerText = JSON.stringify(data, null, 2);
            fetchTelemetry();
        }

        async function runLLM() {
            const prompt = document.getElementById('llm-prompt').value;
            const res = await fetch('/api/llm/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, max_tokens: 16})
            });
            const data = await res.json();
            document.getElementById('llm-output').innerText = JSON.stringify(data, null, 2);
            fetchTelemetry();
        }

        async function runSelfCoding() {
            const name = document.getElementById('kernel-name').value;
            const op = document.getElementById('kernel-op').value;
            const res = await fetch('/api/self_coding/synthesize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, op: op})
            });
            const data = await res.json();
            document.getElementById('kernel-output').innerText = JSON.stringify(data, null, 2);
            fetchTelemetry();
        }

        async function runAGIStep() {
            const goal = document.getElementById('agi-goal').value;
            const res = await fetch('/api/agi/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({goal: goal})
            });
            const data = await res.json();
            document.getElementById('agi-output').innerText = JSON.stringify(data, null, 2);
            fetchTelemetry();
        }

        setInterval(fetchTelemetry, 2000);
        fetchTelemetry();
    </script>
</body>
</html>
"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif url.path == "/api/telemetry":
            data = backend.get_system_telemetry()
            self._send_json(data)

        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw_body) if raw_body else {}

        if url.path == "/api/llm/generate":
            prompt = payload.get("prompt", "Hello")
            max_tokens = int(payload.get("max_tokens", 16))
            result = backend.generate_llm(prompt=prompt, max_tokens=max_tokens)
            self._send_json(result)

        elif url.path == "/api/self_coding/synthesize":
            name = payload.get("name", "custom_kernel")
            op = payload.get("op", "fadd")
            result = backend.trigger_self_coding(name, op)
            self._send_json(result)

        elif url.path == "/api/agi/step":
            goal = payload.get("goal", "Execute task")
            result = backend.trigger_agi_step(goal)
            self._send_json(result)

        elif url.path == "/api/cluster/scan":
            subnet = payload.get("subnet", "127.0.0")
            start = int(payload.get("start", 1))
            end = int(payload.get("end", 2))
            result = backend.trigger_network_scan(subnet_prefix=subnet, start=start, end=end)
            self._send_json(result)

        elif url.path == "/api/cluster/hybrid_gemm":
            size = int(payload.get("size", 16))
            fraction = float(payload.get("gpu_fraction", 0.5))
            result = backend.trigger_hybrid_gemm(size=size, gpu_fraction=fraction)
            self._send_json(result)

        else:
            self.send_error(404, "Endpoint Not Found")

    def _send_json(self, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def run_dashboard(host: str = "0.0.0.0", port: int = 9000):
    server = HTTPServer((host, port), DashboardHTTPHandler)
    log.info("zGPU Control & Telemetry Dashboard running at http://%s:%d/", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    run_dashboard(port=port)
