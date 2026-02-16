# SCXµ Architecture

## Runtime layer (`src/runtime/`)

Responsible for deterministic container concerns and validation:

- SCX4 header parsing
- section mapping
- shard loading orchestration
- Merkle verification
- deterministic replay hashing

This layer contains zero WebGPU business logic.

## GPU layer (`src/gpu/` + `src/wgsl/`)

Compute orchestration only:

- attention
- matmul-int4
- moe dispatch
- rotary transforms
- KV cache primitives

## Distributed layer (`src/distributed/`)

Cross-context coordination primitives:

- BroadcastChannel wrappers
- expert worker contracts
- deterministic merge behavior

## API and tokenizer layers (`src/api/`, `src/tokenizer/`)

- generation/sampling entry points
- BPE encode/decode helpers

## Supporting artifacts

- `docs/SCX4_SPEC.md`: formal container specification
- `docs/SCXQ2_SPEC.md`: shard framing specification
- `docs/CONFORMANCE_AND_CI.md`: CI and conformance policy
- `docs/MODEL_RELEASE_PACKAGING.md`: release packaging contract
