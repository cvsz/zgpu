# zGPU Compiler Studio & Web IDE (zcc)

This document details the **zGPU Compiler Toolchain** (`compiler/`), its AST-level Python JIT compiler, CLI driver (`zcc`), and the interactive Web Studio service.

---

## 1. Components & Toolchain Architecture

| Component | File | Description |
|---|---|---|
| **Python AST JIT** | [`compiler/ast_compiler.py`](file:///home/cvsz/zGPU/compiler/ast_compiler.py) | Ingests native Python numerical functions, extracts arithmetic AST trees, and produces optimized 32-lane vectorized `zISA` assembly. |
| **CLI Compiler** | [`compiler/cli.py`](file:///home/cvsz/zGPU/compiler/cli.py) | Command-line bytecode compiler (`zcc`) compiling assembly to little-endian `.bin` words. |
| **Compiler Service** | [`compiler/service.py`](file:///home/cvsz/zGPU/compiler/service.py) | Full-featured HTTP Web IDE and JSON REST compiler API. |

---

## 2. Interactive Web Compiler Studio

The Web Compiler Studio provides an interactive assembly editor, one-click compilation, and binary bytecode inspection.

### Local URL:
```
http://localhost:9100/
```

### JSON Compilation REST API:
- **Endpoint**: `POST /api/compile`
- **Request Body**:
```json
{
  "source": "GET.TID R0\nVMUL R5, R0, R1\nHALT"
}
```
- **Response**:
```json
{
  "ok": true,
  "instructions_count": 3,
  "bytes_count": 12,
  "bytecode": [80, 524355, 127],
  "bytecode_hex": ["00000050", "00080043", "0000007F"]
}
```

---

## 3. Running the Compiler Services

### Launch the Web Compiler Studio:
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/compiler/service.py 9100
```

### Compile via Command Line:
```bash
PYTHONPATH=/home/cvsz/zGPU python3 /home/cvsz/zGPU/compiler/cli.py input.zasm -o kernel.bin
```
