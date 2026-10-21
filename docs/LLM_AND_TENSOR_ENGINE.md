# zGPU Tensor Core & LLM Acceleration Engine Specification

This document details the hardware acceleration pathways, tensor tiling strategies, and neural network primitives powering Large Language Model (LLM) inference on **zGPU**.

---

## 1. Hardware 4x4x4 Tensor Core MMA Architecture

The zGPU Tensor Matrix Multiply-Accumulate (MMA) Core performs matrix multiplication directly in silicon using 16 lanes of the warp:

$$\mathbf{D}_{4 \times 4} = \mathbf{A}_{4 \times 4} \times \mathbf{B}_{4 \times 4} + \mathbf{C}_{4 \times 4}$$

### Lane Mapping Layout:
Lanes 0 through 15 are mapped in row-major order:
- Lane $k = r \times 4 + c$ represents row $r \in [0..3]$, column $c \in [0..3]$.
- In instruction `MMA.SYNC R10, R1, R2, R3`:
  - `R1`: Register containing elements of matrix $\mathbf{A}$.
  - `R2`: Register containing elements of matrix $\mathbf{B}$.
  - `R3`: Register containing elements of accumulator $\mathbf{C}$.
  - `R10`: Destination register for result $\mathbf{D}$.

---

## 2. Block-Tiled GEMM Kernel (`kernels/llm_kernels.py`)

Arbitrary matrix multiplications $(M \times K) \times (K \times N) \rightarrow (M \times N)$ are decomposed into $4 \times 4$ hardware tiles:

```python
# Tiling loops over dimensions M, N, K in strides of 4
for i in range(0, M, 4):
    for j in range(0, N, 4):
        accum_tile = [0.0] * 16
        for k in range(0, K, 4):
            # Extract 4x4 slice of A and 4x4 slice of B
            tile_a = extract_tile(A, row=i, col=k)
            tile_b = extract_tile(B, row=k, col=j)
            # Execute hardware Tensor MMA
            accum_tile = llm_kernels.matmul_tensor_4x4(tile_a, tile_b, accum_tile)
        write_tile(out, row=i, col=j, data=accum_tile)
```

---

## 3. High-Performance LLM Transformer Primitives

### 3.1 Vectorized RMSNorm
Root Mean Square Layer Normalization (LLaMA architecture):
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{w}$$

### 3.2 Numerically Stable FlashSoftmax
Softmax with online maximum subtraction:
$$m = \max_{j}(x_j), \quad p_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}$$

### 3.3 SwiGLU Fused Activation
Fused gating in feed-forward networks:
$$\text{SwiGLU}(x, W_1, W_2, W_3) = \left(\text{SiLU}(x W_1) \otimes (x W_3)\right) W_2$$

---

## 4. Inference Acceleration Optimizations

### 4.1 Continuous KV-Cache (`llm/kv_cache.py`)
Stores computed Key and Value matrices in device memory across decoding steps:
- Eliminates $O(N^2)$ recomputation for preceding tokens.
- Sequences up to `max_seq_len` advance in $O(1)$ time per layer step.

### 4.2 Symmetric INT8 & INT4 Quantization (`llm/quantization.py`)
- **INT8 Symmetric**: Scales float weights to $[-127, 127]$ with zero zero-point offset ($2\times$ memory footprint reduction).
- **INT4 Packed**: Compresses two 4-bit nibbles $[-7, 7]$ per byte ($4\times$ bandwidth saving).

---

## 5. JSON-RPC Service Protocol (`llm/server.py`)

The server exposes standard JSON-RPC 2.0 endpoints on HTTP (`http://0.0.0.0:8880/rpc`):

### Request: `llm.generate`
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "llm.generate",
  "params": {
    "prompt": "Hello zGPU",
    "max_tokens": 16
  }
}
```

### Response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "text": "Hello zGPU accelerated response...",
    "tokens_used": 16,
    "latency_ms": 142.5,
    "model": "zGPU-Tensor-LLM-16d",
    "finish_reason": "length"
  }
}
```
