# zGPU Self-Coding Engine & Real-Time Telemetry Dashboard

This document details the **Self-Coding Kernel Synthesis Engine** and the **Real-Time Control & Telemetry Dashboard** implemented in **zGPU**.

---

## 1. Autonomous Self-Coding Engine (`self_coding/`)

The Self-Coding subsystem enables zGPU to autonomously synthesize, assemble, verify, and mutate high-performance execution kernels without human intervention.

### 1.1 Vectorized SIMT Kernel Synthesis (`self_coding/generator.py`)
- Takes high-level algebraic operations (`add`, `sub`, `mul`, `fadd`, `fmul`, `relu`, `linear_projection`).
- Generates valid, register-allocated, 32-lane vectorized `zISA` assembly.
- Implements dynamic branch divergence (`BRA.DIV` and `JOIN`) for non-linear activations.

### 1.2 In-Simulator Kernel Evaluation & Verification (`self_coding/evaluator.py`)
- Emits and executes generated binary bytecode on the cycle-accurate zGPU SM simulator.
- Tracks clock cycles, retired instructions, Instructions Per Cycle (IPC), and detects potential deadlocks/divergence loops before deployment.

---

## 2. Real-Time Telemetry & Control Dashboard (`dashboard/`)

A live, lightweight operations dashboard served via standard Python HTTP server (default port `9000`).

### 2.1 Web UI Features:
- **Hardware Telemetry Monitor**: Displays live clock cycles, retired instructions, IPC, DRAM bytes read/written, and shared memory bank conflicts.
- **Interactive LLM Testing**: Submit real-time text prompts to the hardware-accelerated 16-d Transformer model and inspect latency and tokens-per-second (TPS).
- **Self-Coding Studio**: Select operations and trigger real-time zISA assembly synthesis and SM execution validation.
- **SuperAGI Autonomous Loop Visualizer**: Trigger perception-planning-action cycles and inspect Tree-of-Thought scores.

### 2.2 REST & Telemetry Endpoints:
- `GET /api/telemetry`: System performance counters, SM states, and vector memory record counts.
- `POST /api/llm/generate`: Generates text using the zGPU LLM engine (`prompt`, `max_tokens`).
- `POST /api/self_coding/synthesize`: Synthesizes and profiles a new kernel on SM-0 (`name`, `op`).
- `POST /api/agi/step`: Advances the SuperAGI agent autonomous cycle with goal targeting.

---

## 3. Quickstart

### Launch Dashboard:
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/dashboard/server.py 9000
```
Then navigate to `http://localhost:9000` in any web browser.
