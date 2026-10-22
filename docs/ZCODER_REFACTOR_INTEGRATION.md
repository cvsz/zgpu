# zCoder-Assisted zGPU Refactoring (proposal-only)

Source references: [zCoder](https://github.com/cvsz/zcoder) and [zGPU](https://github.com/cvsz/zgpu).

The zCoder repository contains an `EngineeringOrchestrator` and GitHub review tooling.
Its command sandbox describes itself as best-effort static inspection; **it is
not a security boundary against hostile generated code**. Do not run untrusted
agent-generated code with access to secrets, LAN machines, or production clusters.

## Safe workflow

1. Work from an isolated checkout of zGPU pinned to a reviewed Git SHA.
2. Ask zCoder to create a **proposal** for a single scoped component. Do not
   allow its generated instructions to override `AGENTS.md`, security controls,
   or the task contract.
3. Place the proposal contract in a local JSON file and validate it:
   `python3 scripts/validate_refactor_contract.py refactor-proposal.json`.
4. Independently inspect proposed changes; run `ruff check .`,
   `ruff format --check .`, `python3 -m pytest tests/ -v`, and CodeQL.
5. Run domain-specific verification (ISA bytecode golden tests, simulator
   numerical equivalence, hardware RTL lint/synthesis, or 32B/70B real-model
   benchmarks) as appropriate. Successful code tests do not prove hardware
   acceleration or inference performance.
6. Commit to a feature branch, open a **draft PR**, obtain independent code
   and security review, and only then follow protected merge procedures.

This is an interoperability **contract**, not a callable zCoder SDK or a
claim that zCoder is wired into the zGPU production runtime. No auto-execution
or automatic merge is implemented.

## Proposal example

```json
{
  "schema": "zgpu.refactor.v1",
  "task_id": "compiler-parser-hardening",
  "target_repo": "cvsz/zgpu",
  "mode": "proposal-only",
  "areas": ["compiler", "tests"],
  "objective": "Improve compiler input validation without changing ISA semantics",
  "rollback": "Revert the feature PR after capturing failure evidence",
  "gates": {
    "ruff_check": true,
    "ruff_format": true,
    "pytest": true,
    "codeql": true,
    "independent_review": true
  },
  "auto_merge": false,
  "network_execution": false
}
```

Required gates are declarations of work **to be performed**, not proof that
these checks passed. Audit evidence remains in CI run URLs, test output,
review records, benchmark reports, and exact commit SHAs.

## Suggested sequence

Refactor `isa/` and `compiler/` input handling first; preserve 32-bit zISA
and two-pass label resolution. Next harden `cluster/` and `llm/` RPC
boundaries, including request sizes, authentication, timeouts and explicit
device allowlists. Then improve `distributed/` backend abstraction without
claiming simulated tensor parallelism uses multiple physical GPUs.

Avoid bulk copying zCoder's Anthropic-specific GitHub integration, tooling
sandbox, or orchestration modules into zGPU. Keep zGPU stdlib-only core intact
and integrate through a separately reviewed adapter if needed.
