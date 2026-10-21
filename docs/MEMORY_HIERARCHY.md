# zGPU Memory Hierarchy & Interconnect Architecture

This document specifies the storage topologies, bandwidth tiers, latency targets, and coalescing arbitration logic implemented in **zGPU**.

---

## 1. Storage Hierarchy Overview

```
+-------------------------------------------------------------------+
|               L0: Per-Thread Register File (RF)                   |
|  128 regs/thread × 32 lanes × 16 warps = 256 KB per SM (1 cycle)  |
+-------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------+
|               L1: Shared Memory / Scratchpad (SPM)                |
|           64 KB per SM (32 banks, 4 bytes/bank, 2 cycles)         |
+-------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------+
|               L2: Coherent Banked Cache (Write-Back)              |
|        512 KB - 2 MB unified across SM clusters (12 cycles)        |
+-------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------+
|           Device Memory: HBM2 / GDDR6 High-Bandwidth Memory       |
|            4 - 16 GB, 1024-bit bus, 100-200 cycles latency        |
+-------------------------------------------------------------------+
```

---

## 2. Shared Memory Banking & Conflict Arbitration

The 64 KB Shared Memory unit per SM is organized into **32 independent banks**, corresponding to the 32 lanes of a warp.
- **Word Size**: 32 bits (4 bytes) per bank.
- **Bank Index Calculation**:
  $$\text{Bank Index} = \left(\frac{\text{Byte Address}}{4}\right) \pmod{32}$$

### Conflict Resolution:
1. **Zero Conflict (Conflict-Free)**:
   - When all 32 lanes access distinct bank indices in the same cycle, the access completes in **1 cycle**.
2. **Multicast / Broadcast**:
   - When multiple lanes request the *exact same address*, the memory unit detects the identical tag and broadcasts the word in **1 cycle**.
3. **N-Way Bank Conflict**:
   - When multiple lanes request *different addresses* residing in the *same bank*, accesses are serialized into $N$ separate cycles. The hardware performance counter `shared_mem_bank_conflicts` increments accordingly.

---

## 3. Global Memory Coalescing Unit

Global memory instructions (`VLOAD.G`, `VSTORE.G`) route through the hardware Coalescing Unit in the Load/Store Unit (LSU):
1. **Address Inspection**: LSU gathers all 32 target virtual addresses requested by the active threads in the warp.
2. **Cache-Line Partitioning**:
   - Memory is cached in **64-byte cache lines**.
   - If thread addresses $[A_0, A_1, \dots, A_{31}]$ fall into consecutive addresses within a single 64-byte segment, they are merged into **1 single DRAM burst transaction**.
3. **Uncoalesced Stride Penalties**:
   - Strided or divergent pointer accesses that span multiple cache lines trigger multiple DRAM read cycles, scaling transaction latency linearly.

---

## 4. Hardware Divergence Stack

To preserve lockstep execution in SIMT warps when executing conditional branches (`BRA.DIV`), the SM incorporates a hardware **Divergence Stack**:
- **Stack Entry**: `{ActiveMask[31:0], ReconvergencePC[31:0]}`.
- **On Branch**:
  - `TakenMask = ActiveMask & BranchCondition`.
  - Push `{ActiveMask & ~TakenMask, FallthroughPC}`.
  - Set `ActiveMask = TakenMask` and jump to `TargetPC`.
- **On Reconvergence (`JOIN`)**:
  - Pop `{PreviousMask, TargetPC}` from the divergence stack.
  - Set `ActiveMask = PreviousMask`.
