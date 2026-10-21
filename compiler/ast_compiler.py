"""Advanced Python AST to zISA Vectorized JIT Compiler."""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, Dict, List, Optional
from isa.assembler import Assembler


class AstJitCompiler:
    """Compiles Python functions into vectorized 32-lane zISA assembly & bytecode."""

    def __init__(self):
        self.reg_map: Dict[str, str] = {}
        self.next_reg = 12

    def compile_binary_expr(self, func: Callable) -> str:
        """Parses a Python elementwise arithmetic function and generates zISA."""
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        fn_def = tree.body[0]
        assert isinstance(fn_def, ast.FunctionDef)

        # Determine operator
        ret_stmt = None
        for stmt in fn_def.body:
            if isinstance(stmt, ast.Return):
                ret_stmt = stmt
                break

        inst = "VFADD"
        if ret_stmt and isinstance(ret_stmt.value, ast.BinOp):
            op = ret_stmt.value.op
            if isinstance(op, ast.Add):
                inst = "VFADD"
            elif isinstance(op, ast.Sub):
                inst = "VFSUB"
            elif isinstance(op, ast.Mult):
                inst = "VFMUL"

        zisa_code = f"""
        # Compiled from Python: {fn_def.name}
        # Vectorized 32-Lane SIMT Kernel
        VMUL R5, R0, R1         # byte_offset = tid * stride
        VADD R6, R3, R5         # addr_a = base_a + offset
        VADD R7, R4, R5         # addr_b = base_b + offset
        VADD R8, R2, R5         # addr_out = base_out + offset

        VLOAD.G R9, R6, 0       # val_a = *addr_a
        VLOAD.G R10, R7, 0      # val_b = *addr_b
        {inst} R11, R9, R10     # compute operation
        VSTORE.G R8, R11, 0     # *addr_out = result
        HALT
        """
        return zisa_code.strip()

    def compile_to_bytecode(self, source: str) -> List[int]:
        return Assembler.assemble(source)
