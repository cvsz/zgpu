# Contributing to zGPU

Thank you for your interest in contributing to zGPU! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Architecture Overview](#architecture-overview)
- [Style Guide](#style-guide)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/zgpu.git
   cd zgpu
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/cvsz/zgpu.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+ (pure standard library — no heavy binary wheels required)
- `pytest` for running the test suite
- `ruff` for linting and formatting (optional but recommended)

### Install development tools

```bash
pip install pytest ruff
```

### Verify your setup

```bash
PYTHONPATH=$(pwd) python -m pytest tests/ -v
```

All tests must pass before you submit changes.

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```

2. **Follow branch naming conventions:**
   - `feat/` — New features
   - `fix/` — Bug fixes
   - `docs/` — Documentation updates
   - `refactor/` — Code refactoring
   - `perf/` — Performance improvements
   - `test/` — Test additions or fixes

3. **Make focused commits** with clear messages:
   ```
   feat: add 8x8 tensor core tile support
   fix: correct bank conflict detection in shared memory
   docs: update ISA reference for new FMMA instruction
   ```

## Testing

### Running tests

```bash
# Full test suite
PYTHONPATH=$(pwd) python -m pytest tests/ -v

# Specific component
PYTHONPATH=$(pwd) python -m pytest tests/test_zgpu.py -v

# Run benchmarks
PYTHONPATH=$(pwd) python benchmarks/run_benchmarks.py
```

### Test requirements

- **Zero Broken Tests**: Every PR must maintain 100% green status on `pytest tests/`.
- Add tests for any new functionality.
- Ensure benchmarks still run successfully for performance-related changes.

## Submitting Changes

1. **Push** your branch to your fork:
   ```bash
   git push origin feat/my-feature
   ```

2. **Open a Pull Request** against `main` on the upstream repository.

3. **Fill out the PR template** completely.

4. **Wait for CI** to pass — all checks must be green.

5. **Address review feedback** promptly.

## Architecture Overview

zGPU is a full-stack GPU architecture. Please read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before making changes. Key constraints:

| Rule | Description |
|------|-------------|
| **Tensor Core Alignment** | 4×4 matrix tile dimensions (multiples of 4) |
| **SIMT Warps** | 32 threads in lockstep with dynamic divergence stack masking |
| **Portability** | Core code runs with pure Python 3 standard library |
| **No Unverified Claims** | Never claim speedups without benchmark data |

### Module responsibilities

| Directory | Responsibility |
|-----------|---------------|
| `rtl/` | Synthesizable SystemVerilog RTL |
| `isa/` | 32-bit instruction encoding and assembler |
| `simulator/` | Cycle-accurate execution model |
| `kernels/` | Vectorized SIMT & MMA kernels |
| `nn/` | High-level neural network layers |
| `llm/` | Transformer engine & inference server |
| `rasterizer/` | Graphics rendering pipeline |
| `compiler/` | AST → zISA code generator |
| `runtime/` | Host driver API & memory management |

## Style Guide

### Python

- Follow PEP 8 conventions
- Use type hints where practical
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized
- Use `ruff` for linting: `ruff check .`

### SystemVerilog (RTL)

- Use `logic` instead of `reg`/`wire` where possible
- Prefix module ports with `i_` (input) and `o_` (output)
- Include module-level documentation comments
- Use parameterized designs for configurability

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `ci`, `chore`

## Questions?

- Open a [Discussion](https://github.com/cvsz/zgpu/discussions)
- File an [Issue](https://github.com/cvsz/zgpu/issues/new/choose)
- Read the [Documentation](docs/)

Thank you for contributing! 🚀
