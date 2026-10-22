# zISA: zGPU Instruction Set Architecture Reference (v1.0)

This document provides the definitive specification of the 32-bit `zISA` instruction encoding, register conventions, execution semantics, and memory addressing modes for the **zGPU** architecture.

---

## 1. Register Architecture

Each Streaming Multiprocessor (SM) manages a pool of 16 warps. Each warp comprises **32 SIMT lanes**, where each lane possesses:
- **General Purpose Registers (GPRs)**: 128 scalar 32-bit registers (`R0` - `R127`).
- **Total Register Storage per Warp**: $32 \times 128 \times 4\text{ bytes} = 16\text{ KB}$.
- **Total Register Storage per SM**: $16 \times 16\text{ KB} = 256\text{ KB}$.

### Register Conventions:
| Register | Name / Designation | Typical Usage |
|---|---|---|
| `R0` | `%lane_id` / Thread Lane | Initialized to thread lane ID within the warp `[0..31]` |
| `R1` - `R4` | `%arg0` - `%arg3` | Kernel launch argument base addresses & strides |
| `R5` - `R9` | `%scratch` | Temporary address computation & offsets |
| `R10` - `R25` | `%accum` / `%tile` | Matrix tile accumulators & intermediate vectors |
| `R26` - `R127`| General Storage | Spill-free execution & deep neural network weights |

---

## 2. 32-Bit Instruction Word Encoding

All instructions are fixed 32-bit aligned words. Bitfield layout:

```
 31            24 23        19 18        14 13       7 6          0
+----------------+------------+------------+----------+------------+
|  imm8 / rs3    |    rs2     |    rs1     |    rd    |   opcode   |
+----------------+------------+------------+----------+------------+
     8 bits          5 bits       5 bits      7 bits      7 bits
```

- **`opcode` [6:0]**: 7-bit primary operation code.
- **`rd` [13:7]**: 7-bit destination register index (supports up to `R127`).
- **`rs1` [18:14]**: 5-bit first source register index (`R0` - `R31`).
- **`rs2` [23:19]**: 5-bit second source register index (`R0` - `R31`).
- **`imm8 / rs3` [31:24]**: 
  - For immediate-bearing instructions: 8-bit signed immediate value `[-128..127]`.
  - For `VFFMA` and `MMA.SYNC`: 5-bit third source register index `rs3` in bits `[28:24]`; bits `[31:29]` are reserved and must be zero.

The register file has 128 entries, but this encoding exposes only `R0`–`R31` as source operands. `R32`–`R127` are encodable as destinations only. The codec rejects source and destination indices that do not fit their respective fields instead of truncating them.

Fields unused by an instruction format are reserved and must be zero in an encoded instruction word.

---

## 3. Instruction Set Catalog

### 3.1 Control & Special Operations
| Opcode (Hex) | Mnemonic | Format | Description |
|---|---|---|---|
| `0x00` | `NOP` | - | No operation. Advances PC. |
| `0x50` | `GET_TID rd` | I-Type | Sets `rd = lane_id + warp_id * 32`. |
| `0x51` | `GET_BID rd` | I-Type | Sets `rd = block_id`. |
| `0x7F` | `HALT` | - | Signals warp completion and enters idle state. |

### 3.2 Vector Integer Arithmetic
Operates concurrently across all active lanes of the warp.
| Opcode (Hex) | Mnemonic | Format | Semantics |
|---|---|---|---|
| `0x01` | `VADD rd, rs1, rs2` | R-Type | `rd = (rs1 + rs2) & 0xFFFFFFFF` |
| `0x02` | `VSUB rd, rs1, rs2` | R-Type | `rd = (rs1 - rs2) & 0xFFFFFFFF` |
| `0x03` | `VMUL rd, rs1, rs2` | R-Type | `rd = (rs1 * rs2) & 0xFFFFFFFF` |
| `0x04` | `VAND rd, rs1, rs2` | R-Type | `rd = rs1 & rs2` |
| `0x05` | `VOR  rd, rs1, rs2` | R-Type | `rd = rs1 \| rs2` |
| `0x06` | `VXOR rd, rs1, rs2` | R-Type | `rd = rs1 ^ rs2` |

### 3.3 Vector Floating-Point Arithmetic (IEEE 754 Single-Precision)
| Opcode (Hex) | Mnemonic | Format | Semantics | Latency |
|---|---|---|---|---|
| `0x10` | `VFADD rd, rs1, rs2` | R-Type | `rd = rs1 + rs2` | 3 cycles |
| `0x11` | `VFSUB rd, rs1, rs2` | R-Type | `rd = rs1 - rs2` | 3 cycles |
| `0x12` | `VFMUL rd, rs1, rs2` | R-Type | `rd = rs1 * rs2` | 3 cycles |
| `0x13` | `VFFMA rd, rs1, rs2, rs3` | M-Type | `rd = (rs1 * rs2) + rs3` | 4 cycles |

### 3.4 Memory Operations (Global & Shared)
| Opcode (Hex) | Mnemonic | Format | Semantics |
|---|---|---|---|
| `0x20` | `VLOAD.G  rd, rs1, imm` | I-Type | Coalesced load: `rd = GlobalMem[rs1 + imm]` |
| `0x21` | `VSTORE.G rs1, rs2, imm`| S-Type | Coalesced store: `GlobalMem[rs1 + imm] = rs2` |
| `0x22` | `VLOAD.S  rd, rs1, imm` | I-Type | Shared memory load: `rd = SharedMem[(rs1 + imm) >> 2]` |
| `0x23` | `VSTORE.S rs1, rs2, imm`| S-Type | Shared memory store: `SharedMem[(rs1 + imm) >> 2] = rs2` |

### 3.5 Warp Divergence & Flow Control
| Opcode (Hex) | Mnemonic | Format | Semantics |
|---|---|---|---|
| `0x30` | `BRA target` | B-Type | Unconditional branch by signed relative offset or label. |
| `0x31` | `BRA.DIV target, rs1` | B-Type | Divergent branch to a signed relative offset or label when `rs1 != 0`; reconverges through `JOIN`. |
| `0x32` | `JOIN` | - | Reconvergence pop: restores previous `ActiveMask` from stack. |
| `0x33` | `BAR.SYNC` | - | Block barrier synchronization across all warps. |

### 3.6 Tensor Core Matrix Multiply-Accumulate
| Opcode (Hex) | Mnemonic | Format | Semantics |
|---|---|---|---|
| `0x40` | `MMA.SYNC rd, rs1, rs2, rs3` | M-Type | 4x4 matrix tile compute: $D = A \times B + C$. Lanes `[0..15]` hold 4x4 elements in row-major layout. |

---

## 4. Assembler Grammar & Syntax

`zASM` is a two-pass assembler supporting labels, comments, and immediate offsets.
```asm
# Example: Vector SAXPY (Y = a * X + Y)
GET_TID R0
VMUL R5, R0, 4          # offset = tid * 4
VADD R6, R2, R5         # addr_x = base_x + offset
VADD R7, R3, R5         # addr_y = base_y + offset

VLOAD.G R8, R6, 0       # x_val = *addr_x
VLOAD.G R9, R7, 0       # y_val = *addr_y
VFMUL R10, R1, R8       # ax = a * x
VFADD R11, R10, R9      # res = ax + y
VSTORE.G R7, R11, 0     # *addr_y = res
HALT
```

The assembler rejects malformed operand counts or separators, duplicate labels, out-of-range registers, and immediates outside `[-128..127]`. The decoder rejects words outside 32 bits, nonzero reserved fields, and unknown opcodes; unknown opcodes do not alias `NOP`.

The software simulator implements the opcodes in this catalog, including the 32-lane barrier rendezvous. The current `rtl/zgpu_lane.sv` is a partial execution lane, not a complete zISA decoder or implementation. It currently computes the integer ALU subset wired in its `case` statement; other operations in this catalog remain unsupported by RTL.
