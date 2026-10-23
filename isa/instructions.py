"""zGPU Instruction Set Architecture definitions and binary codec."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0x00
    # Vector Arithmetic Integer
    VADD = 0x01
    VSUB = 0x02
    VMUL = 0x03
    VAND = 0x04
    VOR = 0x05
    VXOR = 0x06
    # Vector Arithmetic Float
    VFADD = 0x10
    VFSUB = 0x11
    VFMUL = 0x12
    VFFMA = 0x13
    # Memory Loads/Stores
    VLOAD_G = 0x20  # Vector Load Global
    VSTORE_G = 0x21  # Vector Store Global
    VLOAD_S = 0x22  # Vector Load Shared
    VSTORE_S = 0x23  # Vector Store Shared
    # Flow & Warp Divergence
    BRA = 0x30
    BRA_DIV = 0x31  # Branch on vector condition (divergence push)
    JOIN = 0x32  # Reconvergence pop
    BAR_SYNC = 0x33  # Thread block barrier
    # Tensor / Matrix Acceleration
    MMA_SYNC = 0x40  # 4x4 matrix multiply-accumulate
    # Special Registers & IDs
    GET_TID = 0x50  # Get thread lane ID
    GET_BID = 0x51  # Get block ID
    # Termination
    HALT = 0x7F


_OPCODE_MASK = 0x7F
_RD_MASK = 0x7F << 7
_RS1_MASK = 0x1F << 14
_RS2_MASK = 0x1F << 19
_RS3_MASK = 0x1F << 24
_IMM_MASK = 0xFF << 24
_USED_BITS_MASK = {
    Opcode.NOP: _OPCODE_MASK,
    Opcode.JOIN: _OPCODE_MASK,
    Opcode.BAR_SYNC: _OPCODE_MASK,
    Opcode.HALT: _OPCODE_MASK,
    Opcode.GET_TID: _OPCODE_MASK | _RD_MASK,
    Opcode.GET_BID: _OPCODE_MASK | _RD_MASK,
    Opcode.VADD: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VSUB: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VMUL: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VAND: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VOR: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VXOR: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VFADD: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VFSUB: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VFMUL: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK,
    Opcode.VFFMA: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK | _RS3_MASK,
    Opcode.VLOAD_G: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _IMM_MASK,
    Opcode.VSTORE_G: _OPCODE_MASK | _RS1_MASK | _RS2_MASK | _IMM_MASK,
    Opcode.VLOAD_S: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _IMM_MASK,
    Opcode.VSTORE_S: _OPCODE_MASK | _RS1_MASK | _RS2_MASK | _IMM_MASK,
    Opcode.BRA: _OPCODE_MASK | _IMM_MASK,
    Opcode.BRA_DIV: _OPCODE_MASK | _RS1_MASK | _IMM_MASK,
    Opcode.MMA_SYNC: _OPCODE_MASK | _RD_MASK | _RS1_MASK | _RS2_MASK | _RS3_MASK,
}


@dataclass(frozen=True)
class Instruction:
    op: Opcode
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    rs3: int = 0
    imm: int = 0
    comment: str = ""

    def encode(self) -> int:
        """Encode 32-bit binary instruction word."""
        if not isinstance(self.op, Opcode):
            raise ValueError(f"Invalid opcode: {self.op!r}")

        for name, value, maximum in (
            ("rd", self.rd, 0x7F),
            ("rs1", self.rs1, 0x1F),
            ("rs2", self.rs2, 0x1F),
            ("rs3", self.rs3, 0x1F),
        ):
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in range 0..{maximum}")
        if type(self.imm) is not int or not -128 <= self.imm <= 127:
            raise ValueError("imm must be an integer in range -128..127")
        uses_rs3 = self.op in (Opcode.VFFMA, Opcode.MMA_SYNC)
        if uses_rs3 and self.imm != 0:
            raise ValueError(f"imm is reserved for {self.op.name}")
        if not uses_rs3 and self.rs3 != 0:
            raise ValueError(f"rs3 is reserved for {self.op.name}")

        # [31:24] imm/rs3, [23:19] rs2, [18:14] rs1, [13:7] rd, [6:0] opcode
        raw = int(self.op)
        raw |= self.rd << 7
        raw |= self.rs1 << 14
        raw |= self.rs2 << 19
        if uses_rs3:
            raw |= self.rs3 << 24
        else:
            raw |= (self.imm & 0xFF) << 24
        if raw & ~_USED_BITS_MASK[self.op]:
            raise ValueError(f"Nonzero reserved fields for {self.op.name}")
        return raw

    @classmethod
    def decode(cls, word: int) -> Instruction:
        """Decode 32-bit instruction word."""
        if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
            raise ValueError("instruction word must be an integer in range 0..0xFFFFFFFF")

        op_val = word & 0x7F
        try:
            op = Opcode(op_val)
        except ValueError as exc:
            raise ValueError(f"Unknown opcode 0x{op_val:02X}") from exc

        if word & ~_USED_BITS_MASK[op]:
            raise ValueError(f"Reserved bits must be zero for {op.name}")

        rd = (word >> 7) & 0x7F
        rs1 = (word >> 14) & 0x1F
        rs2 = (word >> 19) & 0x1F
        rs3 = (word >> 24) & 0x1F

        if op in (Opcode.VFFMA, Opcode.MMA_SYNC):
            # These formats use bits [28:24] for rs3.
            imm = 0
        else:
            imm = (word >> 24) & 0xFF
            if imm & 0x80:
                imm -= 0x100

        return cls(op=op, rd=rd, rs1=rs1, rs2=rs2, rs3=rs3, imm=imm)
