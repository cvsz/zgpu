"""zGPU Model Registry & Multi-Model Service Engine.

Supports multiple named LLM model architectures, parameter scaling,
and dynamic weight loading on top of the zGPU SIMT Tensor Core execution stack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from llm.transformer import ZGPULLM, LLMConfig
from runtime.runtime import ZRuntime

log = logging.getLogger("zGPU-ModelRegistry")


@dataclass
class ModelProfile:
    name: str
    description: str
    config: LLMConfig
    tags: list[str]


# Predefined zGPU Silicon Model Architectures
DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "zgpu-nano": ModelProfile(
        name="zgpu-nano",
        description="Ultra-lightweight edge inference model (8-dim, 1 layer)",
        config=LLMConfig(vocab_size=64, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_seq_len=32),
        tags=["edge", "ultra-fast", "nano"],
    ),
    "zgpu-small": ModelProfile(
        name="zgpu-small",
        description="Default balanced model for general SIMT inference (16-dim, 2 layers)",
        config=LLMConfig(vocab_size=128, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_seq_len=64),
        tags=["general", "balanced", "small"],
    ),
    "zgpu-medium": ModelProfile(
        name="zgpu-medium",
        description="Higher-capacity transformer decoder (32-dim, 4 layers, 4 heads)",
        config=LLMConfig(vocab_size=256, d_model=32, num_heads=4, d_ff=64, num_layers=4, max_seq_len=128),
        tags=["medium", "extended-context"],
    ),
    "zgpu-code": ModelProfile(
        name="zgpu-code",
        description="Optimized for autonomous code generation & ISA synthesis",
        config=LLMConfig(vocab_size=128, d_model=16, num_heads=4, d_ff=32, num_layers=3, max_seq_len=64),
        tags=["code", "self-coding", "isa"],
    ),
}


class ModelRegistry:
    """Manages loaded zGPU models and allows dynamic model instantiation."""

    def __init__(self, runtime: ZRuntime | None = None):
        self.runtime = runtime or ZRuntime()
        self.profiles: dict[str, ModelProfile] = dict(DEFAULT_PROFILES)
        self.loaded_models: dict[str, ZGPULLM] = {}
        self.default_model_name: str = "zgpu-small"

        # Pre-initialize the default model
        self.get_or_load(self.default_model_name)

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered model profiles and their loaded status."""
        result = []
        for name, profile in self.profiles.items():
            result.append(
                {
                    "name": name,
                    "description": profile.description,
                    "loaded": name in self.loaded_models,
                    "d_model": profile.config.d_model,
                    "num_layers": profile.config.num_layers,
                    "num_heads": profile.config.num_heads,
                    "vocab_size": profile.config.vocab_size,
                    "max_seq_len": profile.config.max_seq_len,
                    "tags": profile.tags,
                }
            )
        return result

    def register_model(
        self, name: str, config: LLMConfig, description: str = "", tags: list[str] | None = None
    ) -> None:
        """Register a new custom model specification."""
        self.profiles[name] = ModelProfile(
            name=name,
            description=description or f"Custom model {name}",
            config=config,
            tags=tags or ["custom"],
        )
        log.info("Registered model profile: %s", name)

    def get_or_load(self, name: str) -> ZGPULLM:
        """Fetch an already loaded model, or instantiate it on zGPU device memory."""
        if name not in self.profiles:
            raise ValueError(f"Unknown model '{name}'. Available: {list(self.profiles.keys())}")

        if name not in self.loaded_models:
            profile = self.profiles[name]
            log.info(
                "Loading model '%s' (d_model=%d, layers=%d) onto zGPU...",
                name,
                profile.config.d_model,
                profile.config.num_layers,
            )
            self.loaded_models[name] = ZGPULLM(config=profile.config, runtime=self.runtime)

        return self.loaded_models[name]

    def tokenize(self, model_name: str, text: str) -> list[int]:
        profile = self.profiles.get(model_name, self.profiles[self.default_model_name])
        vocab_size = profile.config.vocab_size
        return [ord(c) % vocab_size for c in text]

    def detokenize(self, tokens: list[int]) -> str:
        chars = []
        for t in tokens:
            c = chr(32 + (t % 95))
            chars.append(c)
        return "".join(chars)

    def generate(self, prompt: str, model_name: str | None = None, max_tokens: int = 16) -> dict[str, Any]:
        target_model = model_name or self.default_model_name
        model = self.get_or_load(target_model)

        t0 = time.perf_counter()
        token_ids = self.tokenize(target_model, prompt)
        if not token_ids:
            token_ids = [1]

        out_tokens = model.generate(token_ids, max_new_tokens=max_tokens)
        new_tokens = out_tokens[len(token_ids) :]
        text_out = self.detokenize(new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "text": text_out,
            "tokens_used": len(out_tokens),
            "tokens_generated": len(new_tokens),
            "latency_ms": round(latency_ms, 2),
            "model": target_model,
            "finish_reason": "length" if len(new_tokens) >= max_tokens else "stop",
        }
