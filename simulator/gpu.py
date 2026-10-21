"""zGPU Top-Level Multi-SM Hardware Prototype & Device Orchestrator."""

from __future__ import annotations
from typing import List, Dict, Any
from simulator.memory import GlobalMemory
from simulator.sm import StreamingMultiprocessor


class ZGPU:
    """Complete zGPU hardware prototype with scalable SM cluster configuration."""

    def __init__(self, num_sms: int = 4, global_mem_mb: int = 64):
        self.num_sms = num_sms
        self.global_mem = GlobalMemory(global_mem_mb * 1024 * 1024)
        self.sms = [StreamingMultiprocessor(sm_id=i, global_mem=self.global_mem) for i in range(num_sms)]
        self.total_cycles = 0

    def launch_kernel(self, program: List[int], grid_dim: int, block_dim_warps: int):
        """Dispatch a grid of thread blocks across available SMs."""
        # Distribute blocks round-robin to SMs
        block_queue = list(range(grid_dim))

        while block_queue or any(not sm.is_finished() for sm in self.sms):
            for sm in self.sms:
                if sm.is_finished() and block_queue:
                    bid = block_queue.pop(0)
                    sm.launch_warps(num_warps=block_dim_warps, block_id=bid)

            # Cycle tick all SMs concurrently
            any_active = False
            for sm in self.sms:
                if not sm.is_finished():
                    sm.step_cycle(program)
                    any_active = True

            if any_active:
                self.total_cycles += 1

    def get_performance_counters(self) -> Dict[str, Any]:
        total_instructions = sum(sm.instructions_retired for sm in self.sms)
        total_conflicts = sum(sm.shared_mem.bank_conflicts for sm in self.sms)
        ipc = total_instructions / max(1, self.total_cycles)

        return {
            "total_cycles": self.total_cycles,
            "total_instructions": total_instructions,
            "ipc": round(ipc, 2),
            "dram_reads": self.global_mem.read_transactions,
            "dram_writes": self.global_mem.write_transactions,
            "shared_mem_bank_conflicts": total_conflicts,
        }
