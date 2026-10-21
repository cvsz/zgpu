"""High-Performance zGPU LLM JSON-RPC Server.

Exposes a JSON-RPC 2.0 interface on standard HTTP that serves LLM generation
requests using zGPU Tensor Core and SIMT acceleration.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llm.transformer import ZGPULLM, LLMConfig
from runtime.runtime import ZRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("zGPU-Server")


class ZGPUModelService:
    """Manages the zGPU device context and loaded LLM model."""

    def __init__(self, d_model: int = 16, num_layers: int = 2):
        self.runtime = ZRuntime()
        self.config = LLMConfig(
            vocab_size=128,
            d_model=d_model,
            num_heads=2,
            d_ff=32,
            num_layers=num_layers,
            max_seq_len=64,
        )
        log.info("Initializing zGPU LLM model (d_model=%d, layers=%d)...", d_model, num_layers)
        self.model = ZGPULLM(config=self.config, runtime=self.runtime)

    def simple_tokenize(self, text: str) -> list[int]:
        """Convert ASCII text to token IDs mod vocab_size."""
        return [ord(c) % self.config.vocab_size for c in text]

    def simple_detokenize(self, tokens: list[int]) -> str:
        """Map generated token IDs back to readable characters."""
        chars = []
        for t in tokens:
            # Printable ascii range 32-126
            c = chr(32 + (t % 95))
            chars.append(c)
        return "".join(chars)

    def generate(self, prompt: str, max_tokens: int = 16) -> dict[str, Any]:
        t0 = time.perf_counter()
        token_ids = self.simple_tokenize(prompt)
        if not token_ids:
            token_ids = [1]

        out_tokens = self.model.generate(token_ids, max_new_tokens=max_tokens)
        new_tokens = out_tokens[len(token_ids) :]
        text_out = self.simple_detokenize(new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "text": text_out,
            "tokens_used": len(out_tokens),
            "latency_ms": latency_ms,
            "model": "zGPU-Tensor-LLM-16d",
            "finish_reason": "length" if len(new_tokens) >= max_tokens else "stop",
        }


# Global singleton instance
service = ZGPUModelService()


class JSONRPCHandler(BaseHTTPRequestHandler):
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
                }
                resp_payload = {"jsonrpc": "2.0", "id": req_id, "result": result}

            elif method == "llm.generate":
                prompt = params.get("prompt", "")
                max_tokens = int(params.get("max_tokens", 16))
                result = service.generate(prompt=prompt, max_tokens=max_tokens)
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

        response_bytes = json.dumps(resp_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

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
