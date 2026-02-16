# SCXµ RFC Draft

## Deterministic Browser-Native Sparse Transformer Runtime

- **Status**: Draft
- **Intended audience**: Runtime implementers, model packagers, conformance tool authors
- **Version**: 0.1

## 1. Scope

This document defines normative runtime and packaging requirements for SCXµ-compliant implementations:

1. SCX4 deterministic container parsing and validation
2. SCXQ2 shard streaming and verification
3. INT4 block decode semantics
4. Deterministic sparse MoE routing behavior
5. Replay hash production for conformance

This RFC does not define model training procedures.

## 2. Terminology

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as described in RFC 2119.

## 3. Conformance Requirements

An implementation is **SCXµ-compliant** only if it satisfies all MUST-level statements in this document and in the linked SCX4/SCXQ2 specifications.

### 3.1 Container Validation

- Implementations **MUST** parse SCX4 as little-endian binary.
- Implementations **MUST** verify section integrity against the SCX4 Merkle root before executing any section payload.
- Implementations **MUST NOT** mutate model weights at runtime.

### 3.2 INT4 Decode

For each packed nibble `v`:

```text
if v >= 8 then v = v - 16
real = (v - zero_point) * scale
```

- Implementations **MUST** apply the above signed conversion and affine dequantization exactly.
- Implementations **MUST** use deterministic arithmetic ordering for reproducible outputs.

### 3.3 Sparse Routing

- Router scores **MUST** be computed from `softmax(x · W_router)`.
- Implementations **MUST** select top-k experts by descending score.
- Score ties **MUST** be resolved by lower expert index.
- Top-k output merge **MUST** be weighted by selected router scores.

### 3.4 Distributed Execution

- Implementations **MAY** distribute expert execution across tabs/workers.
- Distributed merge logic **MUST** preserve deterministic ordering and numerical behavior equivalent to single-context execution.

### 3.5 Replay Hashing

For each layer, implementations **MUST** produce:

```text
SHA256(input_tensor || router_weights || selected_experts || output_tensor)
```

- Byte layout and endianness for hash inputs **MUST** remain stable across platforms.

## 4. Security Considerations

- SCX4 and SCXQ2 payloads **MUST** be integrity-verified before use.
- Execution engines **MUST NOT** evaluate dynamic code from model artifacts.
- Runtime components **SHOULD** minimize attack surface by limiting accepted binary section types.

## 5. Performance Guidance (Non-Normative)

Typical target ranges on desktop iGPU hardware:

- 5–15 tokens/s
- 35–40 MB quantized weight footprint
- ~65 MB total runtime memory with KV cache

## 6. Future Work

- WebRTC multi-device expert meshes
- Alternate low-bit quantization tracks (INT1 / ternary)
- WASM fallback backend for non-WebGPU environments
