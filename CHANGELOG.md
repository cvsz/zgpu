# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete GitHub repository infrastructure
  - CI/CD workflows (test matrix, lint, benchmarks)
  - Issue templates (bug report, feature request, question)
  - Pull request template with checklist
  - Dependabot configuration for GitHub Actions
  - CODEOWNERS for automated review assignments
  - GitHub Sponsors funding configuration
  - Docs deployment workflow
  - Release automation workflow
- Project metadata and packaging
  - `pyproject.toml` with full PEP 621 metadata
  - `Makefile` with development workflow commands
  - `ruff` linter and formatter configuration
- Community files
  - `CONTRIBUTING.md` — contributor guidelines
  - `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
  - `SECURITY.md` — vulnerability reporting policy
  - `CHANGELOG.md` — this changelog
  - `LICENSE` — MIT license
- AI agent instruction files
  - `AGENTS.md` — universal agent contract
  - `GEMINI.md` — Gemini-specific instructions
  - `CLAUDE.md` — Claude Code instructions
  - `OPENCODE.md` — OpenCode instructions
- Missing `__init__.py` files for all packages
  - `isa/`, `simulator/`, `kernels/`, `kernels/flash_attn/`
  - `llm/speculative/`, `rasterizer/`, `compiler/`, `runtime/`
  - `distributed/`, `cluster/worker/`, `dashboard/`, `torch_compat/`
- Python 3.14 support in CI test matrix

### Fixed
- `AstJitCompiler.compile_binary_expr()` — graceful fallback when
  `inspect.getsource()` fails (e.g., dynamically defined functions in pytest)

### Changed
- Updated `README.md` with comprehensive architecture overview, badges, directory
  table, quickstart, web services table, and documentation links
- Updated `.gitignore` with comprehensive patterns for Python, testing, IDE,
  synthesis, and OS artifacts
- Updated `ROADMAP.md` — marked completed milestones (FlashAttention-3,
  Speculative Decoding, Distributed TP, SuperAGI Swarm, Network CPU Clustering)

## [1.0.0] - 2026-10-22

### Added
- Full-stack synthesizable SIMT GPU architecture
- Cycle-accurate SystemVerilog hardware RTL (`rtl/`)
  - 32-lane SIMT Core with ALU/FPU lane
  - Hardware divergence stack
- Custom 32-bit `zISA` instruction codec and two-pass assembler (`isa/`)
- Cycle-accurate SM simulator with GTO warp scheduler (`simulator/`)
  - 64KB 32-banked shared memory
  - Coalesced DRAM controller
  - 4×4×4 Tensor MMA Core
- Accelerated SIMT & MMA kernels (`kernels/`)
  - Block-tiled GEMM
  - RMSNorm
  - FlashSoftmax
  - SwiGLU
- High-level neural network layers (`nn/`)
- LLM & Transformer Engine (`llm/`)
  - Causal self-attention
  - Transformer decoder blocks
  - Autoregressive text generation
  - High-throughput JSON-RPC inference server
  - KV-Cache with continuous batching
  - INT8/INT4 quantization
  - Speculative decoding engine
- FlashAttention-3 tiled online softmax kernel (`kernels/flash_attn/`)
- Fixed-function graphics rasterizer (`rasterizer/`)
  - Triangle rasterization with barycentric interpolation
  - Early-Z buffer ROPs
- JIT kernel compiler (`compiler/`)
  - AST to zISA code generator
  - CLI driver (`zcc`)
  - Web Compiler Studio
- Host runtime driver API (`runtime/`)
  - Device memory allocator
  - Host-to-device transfers
  - Asynchronous grid dispatcher
- Distributed tensor parallelism (`distributed/`)
  - All-Reduce crossbar
  - Column/row-parallel decomposition
- Network CPU clustering (`cluster/`)
  - Work-stealing dispatcher
  - Automated subnet discovery scanner
  - HTTP worker daemon
- SuperAGI autonomous agent framework (`agi/`)
  - Agent core with tool execution
  - Tree-of-Thought planner
  - Vector memory store
  - Multi-agent swarm coordination
- Self-coding kernel generator and evaluator (`self_coding/`)
- PyTorch-compatible ZTensor interface (`torch_compat/`)
- Real-time telemetry dashboard (`dashboard/`)
- Comprehensive benchmark suite (`benchmarks/`)
- Full pytest verification test suite (`tests/`)
- Architecture documentation (`docs/`)

[Unreleased]: https://github.com/cvsz/zgpu/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/cvsz/zgpu/releases/tag/v1.0.0
