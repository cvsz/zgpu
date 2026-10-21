# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅ Active |

## Reporting a Vulnerability

If you discover a security vulnerability in zGPU, please report it responsibly.

**⚠️ Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Email**: Send details to **[INSERT SECURITY EMAIL]**
2. **GitHub Security Advisories**: Use the [private vulnerability reporting](https://github.com/cvsz/zgpu/security/advisories/new) feature

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Action | Timeframe |
|--------|-----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 1 week |
| Fix & disclosure | Within 90 days |

### Scope

The following components are in scope for security reports:

- **LLM Inference Server** (`llm/server.py`) — Network-exposed JSON-RPC endpoint
- **Runtime Driver** (`runtime/`) — Memory management and device interaction
- **Compiler** (`compiler/`) — Code generation from untrusted input
- **Simulator** (`simulator/`) — Execution of arbitrary instruction streams

### Out of Scope

- RTL (SystemVerilog) — Not executable in production without synthesis
- Documentation and benchmark scripts

## Security Best Practices

When contributing to zGPU:

- Validate all external inputs, especially in the JSON-RPC server
- Never execute user-supplied strings as code
- Bound all memory allocations
- Use safe integer arithmetic to prevent overflow
