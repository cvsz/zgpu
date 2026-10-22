"""zGPU Assembler: converts human-readable SIMT assembly to binary bytecode."""

from __future__ import annotations

from isa.instructions import Instruction, Opcode

OPCODE_MAP = {
    "NOP": Opcode.NOP,
    "VADD": Opcode.VADD,
    "VSUB": Opcode.VSUB,
    "VMUL": Opcode.VMUL,
    "VAND": Opcode.VAND,
    "VOR": Opcode.VOR,
    "VXOR": Opcode.VXOR,
    "VFADD": Opcode.VFADD,
    "VFSUB": Opcode.VFSUB,
    "VFMUL": Opcode.VFMUL,
    "VFFMA": Opcode.VFFMA,
    "VLOAD.G": Opcode.VLOAD_G,
    "VSTORE.G": Opcode.VSTORE_G,
    "VLOAD.S": Opcode.VLOAD_S,
    "VSTORE.S": Opcode.VSTORE_S,
    "BRA": Opcode.BRA,
    "BRA.DIV": Opcode.BRA_DIV,
    "JOIN": Opcode.JOIN,
    "BAR.SYNC": Opcode.BAR_SYNC,
    "MMA.SYNC": Opcode.MMA_SYNC,
    "GET.TID": Opcode.GET_TID,
    "GET.BID": Opcode.GET_BID,
    "HALT": Opcode.HALT,
}


def parse_reg(token: str) -> int:
    token = token.strip().upper()
    if token.startswith("R"):
        register_index = token[1:]
        if register_index.isascii() and register_index.isdigit():
            return int(register_index)
    raise ValueError(f"Invalid register name: {token}")


def _require_operand_count(tokens: list[str], minimum: int, maximum: int | None = None) -> None:
    operand_count = len(tokens) - 1
    maximum = minimum if maximum is None else maximum
    if not minimum <= operand_count <= maximum:
        expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
        raise ValueError(f"{tokens[0].upper()} expects {expected} operand(s), got {operand_count}")


class Assembler:
    @staticmethod
    def assemble(source: str) -> list[int]:
        labels: dict[str, int] = {}
        lines = source.strip().splitlines()
        clean_lines = []

        # Pass 1: Collect labels and strip comments
        pc = 0
        for line in lines:
            line = line.partition("#")[0].strip()
            if not line:
                continue
            if line.endswith(":"):
                label_name = line[:-1].strip()
                if not label_name:
                    raise ValueError("Label name cannot be empty")
                if label_name in labels:
                    raise ValueError(f"Duplicate label '{label_name}'")
                labels[label_name] = pc
            else:
                clean_lines.append((pc, line))
                pc += 1

        # Pass 2: Assemble instructions
        bytecode: list[int] = []
        for curr_pc, line in clean_lines:
            parts = line.split(None, 1)
            mnemonic = parts[0].upper()
            operand_text = parts[1].strip() if len(parts) > 1 else ""
            operands = [operand.strip() for operand in operand_text.split(",")] if operand_text else []
            if any(not operand for operand in operands):
                raise ValueError(f"Malformed operand separators in line: {line}")
            tokens = [mnemonic, *operands]
            if mnemonic not in OPCODE_MAP:
                raise ValueError(f"Unknown instruction '{mnemonic}' at line: {line}")

            op = OPCODE_MAP[mnemonic]
            rd = rs1 = rs2 = rs3 = imm = 0

            if op in (Opcode.NOP, Opcode.JOIN, Opcode.BAR_SYNC, Opcode.HALT):
                _require_operand_count(tokens, 0)
            elif op in (Opcode.GET_TID, Opcode.GET_BID):
                _require_operand_count(tokens, 1)
                rd = parse_reg(tokens[1])
            elif op in (
                Opcode.VADD,
                Opcode.VSUB,
                Opcode.VMUL,
                Opcode.VAND,
                Opcode.VOR,
                Opcode.VXOR,
                Opcode.VFADD,
                Opcode.VFSUB,
                Opcode.VFMUL,
            ):
                _require_operand_count(tokens, 3)
                rd = parse_reg(tokens[1])
                rs1 = parse_reg(tokens[2])
                rs2 = parse_reg(tokens[3])
            elif op == Opcode.VFFMA:
                _require_operand_count(tokens, 4)
                rd = parse_reg(tokens[1])
                rs1 = parse_reg(tokens[2])
                rs2 = parse_reg(tokens[3])
                rs3 = parse_reg(tokens[4])
            elif op in (Opcode.VLOAD_G, Opcode.VLOAD_S):
                _require_operand_count(tokens, 2, 3)
                rd = parse_reg(tokens[1])
                rs1 = parse_reg(tokens[2])  # Base address reg
                if len(tokens) > 3:
                    imm = int(tokens[3])
            elif op in (Opcode.VSTORE_G, Opcode.VSTORE_S):
                _require_operand_count(tokens, 2, 3)
                rs1 = parse_reg(tokens[1])  # Base address reg
                rs2 = parse_reg(tokens[2])  # Value to store
                if len(tokens) > 3:
                    imm = int(tokens[3])
            elif op == Opcode.BRA:
                _require_operand_count(tokens, 1)
                target = tokens[1]
                if target in labels:
                    imm = labels[target] - curr_pc
                else:
                    imm = int(target)
            elif op == Opcode.BRA_DIV:
                _require_operand_count(tokens, 2)
                target = tokens[1]
                if target in labels:
                    imm = labels[target] - curr_pc
                else:
                    imm = int(target)
                rs1 = parse_reg(tokens[2])  # Condition register
            elif op == Opcode.MMA_SYNC:
                _require_operand_count(tokens, 4)
                rd = parse_reg(tokens[1])
                rs1 = parse_reg(tokens[2])
                rs2 = parse_reg(tokens[3])
                rs3 = parse_reg(tokens[4])

            inst = Instruction(op=op, rd=rd, rs1=rs1, rs2=rs2, rs3=rs3, imm=imm)
            bytecode.append(inst.encode())

        return bytecode
