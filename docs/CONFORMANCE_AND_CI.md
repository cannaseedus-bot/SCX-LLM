# CI Pipeline and Conformance Test Design

## CI policy

The default CI workflow runs:

1. Dependency install
2. Production build
3. Test suite

See `.github/workflows/ci.yml`.

## Conformance suite

Conformance checks are represented by:

- SCX4 metadata loading
- Deterministic replay hashing
- Quantization error bounds
- Merkle tamper detection
- Tokenizer roundtrip invariants

## Determinism contract

For identical input payloads and routing selections, layer replay hashes must be identical across runs.

## Quantization contract

For reference vectors, INT4 dequantization must keep relative error under a fixed threshold (5% in tests).
