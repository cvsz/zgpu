"""Boundary and malformed-input tests for the fixed-width zISA codec."""

import pytest

from isa.assembler import Assembler
from isa.instructions import Instruction, Opcode
from simulator.gpu import ZGPU
from simulator.sm import bits_to_f32, f32_to_bits


def test_vadd_golden_encoding():
    instruction = Instruction(op=Opcode.VADD, rd=5, rs1=2, rs2=3)

    assert instruction.encode() == 0x00188281
    assert Instruction.decode(instruction.encode()) == instruction


@pytest.mark.parametrize("immediate", [-128, -1, 0, 127])
def test_signed_immediate_boundaries_round_trip(immediate):
    instruction = Instruction(op=Opcode.VLOAD_G, rd=7, rs1=3, imm=immediate)

    assert Instruction.decode(instruction.encode()).imm == immediate


def test_vffma_encodes_and_decodes_third_source_register():
    instruction = Instruction(op=Opcode.VFFMA, rd=7, rs1=2, rs2=3, rs3=31)

    assert instruction.encode() == 0x1F188393
    decoded = Instruction.decode(instruction.encode())
    assert (decoded.op, decoded.rd, decoded.rs1, decoded.rs2, decoded.rs3) == (
        Opcode.VFFMA,
        7,
        2,
        3,
        31,
    )


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction(op=Opcode.VADD, rd=128),
        Instruction(op=Opcode.VADD, rs1=32),
        Instruction(op=Opcode.VFFMA, rs3=32),
        Instruction(op=Opcode.VLOAD_G, imm=-129),
        Instruction(op=Opcode.VLOAD_G, imm=128),
        Instruction(op=Opcode.VADD, rd=True),
    ],
)
def test_encode_rejects_out_of_range_or_non_integer_operands(instruction):
    with pytest.raises(ValueError):
        instruction.encode()


def test_encode_rejects_nonzero_reserved_fields():
    for instruction in (
        Instruction(op=Opcode.NOP, rd=1),
        Instruction(op=Opcode.VADD, rs3=1),
        Instruction(op=Opcode.MMA_SYNC, imm=1),
    ):
        with pytest.raises(ValueError, match="reserved"):
            instruction.encode()


@pytest.mark.parametrize("word", [-1, 1 << 32, True, 0x7E])
def test_decode_rejects_invalid_word_and_unknown_opcode(word):
    with pytest.raises(ValueError):
        Instruction.decode(word)


def test_decode_rejects_reserved_fma_bits():
    with pytest.raises(ValueError, match="Reserved bits"):
        Instruction.decode(0x20000013)


def test_decode_rejects_reserved_fields_on_simple_instructions():
    with pytest.raises(ValueError, match="Reserved bits"):
        Instruction.decode(0x00008000 | Opcode.NOP)


@pytest.mark.parametrize(
    "source",
    [
        "VADD R1, R2",
        "VADD R1, R2, R3, R4",
        "VADD R1,, R2, R3",
        "VADD R1, R2, R3,",
        "VADD R1 R2 R3",
        "VADD R1, R32, R3",
        "VADD R1, R+2, R3",
        "VLOAD.G R1, R2, 128",
        "NOP R1",
        "BRA 1, R2",
        "BRA.DIV 1",
        "HALT extra",
    ],
)
def test_assembler_rejects_malformed_or_out_of_range_operands(source):
    with pytest.raises(ValueError):
        Assembler.assemble(source)


def test_assembler_resolves_relative_branch_target():
    bytecode = Assembler.assemble("BRA DONE\nNOP\nDONE:\nHALT")

    assert Instruction.decode(bytecode[0]).imm == 2


def test_assembler_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="Duplicate label"):
        Assembler.assemble("LOOP:\nNOP\nLOOP:\nHALT")


@pytest.mark.parametrize(
    ("mnemonic", "expected"),
    [("VAND", 0x30), ("VOR", 0xFC), ("VXOR", 0xCC)],
)
def test_integer_bitwise_opcodes_execute(mnemonic, expected):
    gpu = ZGPU(num_sms=1)
    sm = gpu.sms[0]
    sm.launch_warps(1)
    warp = sm.warps[0]
    for lane in range(32):
        warp.rf[lane][1] = 0xF0
        warp.rf[lane][2] = 0x3C

    program = Assembler.assemble(f"{mnemonic} R3, R1, R2\nHALT")
    while not sm.is_finished():
        sm.step_cycle(program)

    assert [warp.rf[lane][3] for lane in range(32)] == [expected] * 32


def test_vfsub_opcode_executes():
    gpu = ZGPU(num_sms=1)
    sm = gpu.sms[0]
    sm.launch_warps(1)
    warp = sm.warps[0]
    for lane in range(32):
        warp.rf[lane][1] = f32_to_bits(7.5)
        warp.rf[lane][2] = f32_to_bits(2.25)

    program = Assembler.assemble("VFSUB R3, R1, R2\nHALT")
    while not sm.is_finished():
        sm.step_cycle(program)

    assert all(bits_to_f32(warp.rf[lane][3]) == pytest.approx(5.25) for lane in range(32))


def test_vffma_third_source_survives_assembler_and_executes():
    gpu = ZGPU(num_sms=1)
    sm = gpu.sms[0]
    sm.launch_warps(1)
    warp = sm.warps[0]
    for lane in range(32):
        warp.rf[lane][1] = f32_to_bits(2.0)
        warp.rf[lane][2] = f32_to_bits(3.0)
        warp.rf[lane][3] = f32_to_bits(4.0)

    program = Assembler.assemble("VFFMA R4, R1, R2, R3\nHALT")
    while not sm.is_finished():
        sm.step_cycle(program)

    assert all(bits_to_f32(warp.rf[lane][4]) == pytest.approx(10.0) for lane in range(32))


def test_bar_sync_waits_for_each_live_warp():
    gpu = ZGPU(num_sms=1)
    sm = gpu.sms[0]
    sm.launch_warps(2)
    program = Assembler.assemble("BAR.SYNC\nHALT")

    sm.step_cycle(program)
    assert sm.warps[0].stalled
    assert sm.warps[1].pc == 0

    sm.step_cycle(program)
    assert [warp.stalled for warp in sm.warps] == [False, False]
    assert [warp.pc for warp in sm.warps] == [1, 1]
