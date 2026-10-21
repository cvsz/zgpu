"""Unit tests for zGPU ISA, SIMT Divergence, Memory Coalescing, and Execution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isa.instructions import Instruction, Opcode
from isa.assembler import Assembler
from simulator.gpu import ZGPU
from simulator.warp import Warp
from simulator.memory import GlobalMemory, SharedMemory
from simulator.sm import f32_to_bits, bits_to_f32


def test_instruction_encode_decode():
    inst = Instruction(op=Opcode.VADD, rd=5, rs1=2, rs2=3)
    encoded = inst.encode()
    decoded = Instruction.decode(encoded)
    assert decoded.op == Opcode.VADD
    assert decoded.rd == 5
    assert decoded.rs1 == 2
    assert decoded.rs2 == 3


def test_simt_warp_divergence():
    # If lane is odd (R1=1) -> add 10, else (R1=0) -> add 20
    # BRA.DIV branches to THEN_LABEL if R1 != 0, while pushing fallthrough to stack
    kernel = """
    BRA.DIV THEN_LABEL, R1
    # Fallthrough path: R1 == 0 (even lanes)
    VADD R3, R3, R20
    BRA END_LABEL
    THEN_LABEL:
    # Taken path: R1 == 1 (odd lanes)
    VADD R3, R3, R10
    END_LABEL:
    JOIN
    HALT
    """
    bytecode = Assembler.assemble(kernel)
    gpu = ZGPU(num_sms=1)
    sm = gpu.sms[0]
    sm.launch_warps(1)
    warp = sm.warps[0]

    for lane in range(32):
        warp.rf[lane][1] = 1 if (lane % 2 == 1) else 0  # condition
        warp.rf[lane][3] = 0
        warp.rf[lane][10] = 10
        warp.rf[lane][20] = 20

    while not sm.is_finished():
        sm.step_cycle(bytecode)

    for lane in range(32):
        expected = 10 if (lane % 2 == 1) else 20
        assert warp.rf[lane][3] == expected


def test_shared_memory_bank_conflicts():
    shm = SharedMemory()
    # All 32 lanes access distinct banks: 0..31 -> 0 conflicts
    shm.access_warp(list(range(32)), 0xFFFFFFFF)
    assert shm.bank_conflicts == 0

    # All 32 lanes access same bank (index 0, 32, 64, ...) -> 31 conflicts
    conflicting_indices = [i * 32 for i in range(32)]
    shm.access_warp(conflicting_indices, 0xFFFFFFFF)
    assert shm.bank_conflicts == 31
