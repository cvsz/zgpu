"""High-Performance zGPU LLM JSON-RPC Server with Multi-Model Support.

Exposes a JSON-RPC 2.0 interface on standard HTTP that serves LLM generation
requests using zGPU Tensor Core and SIMT acceleration across multiple registered models.
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llm.model_registry import ModelRegistry
from llm.transformer import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("zGPU-Server")

# Global Model Registry managing multiple models
registry = ModelRegistry()


class JSONRPCHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Respond with health status, models list, or endpoints on GET requests."""
        if self.path in ("/", "/rpc", "/health"):
            data = {
                "jsonrpc": "2.0",
                "result": {
                    "healthy": True,
                    "engine": "zGPU-Silicon-SIMT-TensorCore",
                    "device": "zGPU-SM-0",
                    "status": "ready",
                    "available_models": [m["name"] for m in registry.list_models()],
                    "endpoints": {
                        "GET /models": "List available models and configuration",
                        "POST /rpc": "JSON-RPC 2.0 API (methods: system.health, llm.models, llm.add_model, llm.generate)",
                    },
                },
            }
            self._send_json(data)

        elif self.path in ("models", "/models", "/api/models"):
            data = {
                "jsonrpc": "2.0",
                "result": {
                    "models": registry.list_models(),
                },
            }
            self._send_json(data)

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")

        try:
            req = json.loads(raw_body)
            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")

            if method == "system.health":
                result = {
                    "healthy": True,
                    "engine": "zGPU-Silicon-SIMT-TensorCore",
                    "device": "zGPU-SM-0",
                    "status": "ready",
                    "models": [m["name"] for m in registry.list_models()],
                }
                resp_payload = {"jsonrpc": "2.0", "id": req_id, "result": result}

            elif method in ("llm.models", "models.list"):
                result = {"models": registry.list_models()}
                resp_payload = {"jsonrpc": "2.0", "id": req_id, "result": result}

            elif method in ("llm.add_model", "models.add"):
                model_name = params.get("name")
                if not model_name:
                    raise ValueError("Parameter 'name' is required")
                d_model = int(params.get("d_model", 16))
                if d_model % 4 != 0:
                    raise ValueError("d_model must be a multiple of 4 for zGPU Tensor Core tiling")
                num_layers = int(params.get("num_layers", 2))
                num_heads = int(params.get("num_heads", 2))
                d_ff = int(params.get("d_ff", d_model * 2))
                vocab_size = int(params.get("vocab_size", 128))
                max_seq_len = int(params.get("max_seq_len", 64))
                desc = params.get("description", f"Custom model {model_name}")
                tags = params.get("tags", ["custom"])

                cfg = LLMConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_layers=num_layers,
                    max_seq_len=max_seq_len,
                )
                registry.register_model(name=model_name, config=cfg, description=desc, tags=tags)
                result = {
                    "success": True,
                    "message": f"Model '{model_name}' successfully added to zGPU registry",
                    "model": model_name,
                    "config": {
                        "d_model": d_model,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "vocab_size": vocab_size,
                    },
                }
                resp_payload = {"jsonrpc": "2.0", "id": req_id, "result": result}

            elif method == "llm.generate":
                prompt = params.get("prompt", "")
                model_name = params.get("model")
                max_tokens = int(params.get("max_tokens", 16))
                result = registry.generate(prompt=prompt, model_name=model_name, max_tokens=max_tokens)
                resp_payload = {"jsonrpc": "2.0", "id": req_id, "result": result}

            else:
                resp_payload = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            log.exception("Error processing RPC request")
            resp_payload = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)},
            }

        self._send_json(resp_payload)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Override to suppress default noisy console logs
        pass


def run_server(host: str = "0.0.0.0", port: int = 8880):
    server = HTTPServer((host, port), JSONRPCHandler)
    log.info("zGPU Accelerated LLM Server listening on http://%s:%d/rpc", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down zGPU Server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8880
    run_server(port=port)
