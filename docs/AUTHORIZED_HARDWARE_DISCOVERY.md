# Authorized Hardware Discovery and 32B/70B Placement Planning

Status: **inventory and capacity estimation only**. This feature does not
probe a subnet, contact remote machines, enroll workers or execute models.

## Trust and consent model

The operator installs and runs the CLI directly on each owned or explicitly
authorized machine. Transfer each JSON inventory through an approved channel.
The `--authorized-host` flag records operator affirmation, **not** proof of
ownership, authentication, or remote attestation. Inventory files are
untrusted and should be manually reviewed before ranking.

Never use addresses found by network scanning as proof of authorization.
Do not send inventory files to external services without approval. Hostnames,
GPU UUIDs and hardware identifiers may be sensitive.

## Quickstart (Python 3.11+)

On each approved Linux/WSL2 worker:

```bash
python3 -m cluster.hardware_discovery inventory \
  --authorized-host --node-id node-02 \
  --output node-02.json
```

Repeat on other authorized workers using a distinct ID. The command
inspects only the local machine and never installs a daemon. Windows uses
the same Python CLI but does not yet provide GPU inventory in v1;
unknown capabilities are reported as unknown rather than invented.

On the coordinator, after manually transferring JSON files:

```bash
python3 -m cluster.hardware_discovery rank --model 32b-q4 node-*.json > ranking-32b.json
python3 -m cluster.hardware_discovery rank --model 70b-q4 node-*.json > ranking-70b.json
python3 -m pytest tests/test_hardware_discovery.py -v
```

Do not use `node-*.json` when unrelated untrusted JSON files exist in the
working directory; pass exact approved paths instead.

## Meaning of rankings

The planning profiles use conservative example weight budgets, **24 GiB for
32B Q4 and 52 GiB for 70B Q4**. These are not exact model file sizes;
the actual model architecture, quantization, KV cache, and context length
can significantly change requirements.

- Available RAM estimate: total RAM minus max(4 GiB, 20% of total).
- CUDA VRAM estimate: largest individually reported CUDA GPU minus 2 GiB.
- RAM and VRAM are **not** added; other GPUs are **not** summed.
- A device reported by PCI without usable VRAM/driver remains unknown.
- `single_node_candidate` means a capacity estimate, **not runnable proof**.
- Offline inventory hashes are change-detection aids, not attestation.
- The ranking deliberately does not claim that a 1 GbE distributed 70B
  model is feasible: that requires a separate sharding backend and
  end-to-end load/generation testing.

## Evidence and promotion gates

Before selecting a model-serving worker, verify identity and authorization
out of band, record full model SHA-256 and exact GGUF quantization, confirm
driver/backend and actual available memory, and measure real prompt and
generation performance. Record network iperf3 throughput, TTFT, tokens/sec,
peak RAM/VRAM, failure recovery and a 60-minute soak. CI success alone does
not establish production readiness.

For `llama.cpp RPC`, restrict access to an isolated network with authenticated
tunnels and perform independent security review before processing customer
content. No automatic production enrollment is provided.

## Security boundaries

- No remote execution or subnet scanning.
- No network listeners or open ports.
- File paths are supplied locally; the CLI does not fetch URLs.
- Inventory files have a 128 KiB cap and duplicate IDs fail closed.
- Never mark a benchmark as independently verified solely because its input
  JSON sets `verified: true`. Benchmark fields remain self-reported.
- Keep `cluster/network_pool.py` and existing unauthenticated workers off
  untrusted LANs until separately hardened.
