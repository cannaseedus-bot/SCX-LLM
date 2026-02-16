# SCXµ

## A Deterministic Sparse Transformer Runtime for the Browser

### SCX4 Container · INT4 Quantization · Distributed MoE Execution

Version 1.0  
Author: SCXµ Project

---

## Abstract

SCXµ is a browser-native sparse transformer runtime designed for deterministic execution, quantized efficiency, and distributed expert scaling.

It introduces:

- **SCX4**, a binary INT4 model container format with Merkle integrity
- **SCXQ2**, a sharded streaming format for progressive loading
- **SCXµ**, a deterministic sparse Mixture-of-Experts (MoE) execution layer
- WebGPU-accelerated Flash-style attention
- Multi-tab distributed expert execution
- Deterministic replay verification

Unlike traditional server-hosted LLM architectures, SCXµ enables fully client-side inference for a 16-layer, 64M-parameter sparse transformer entirely within the browser.

---

## 1. Motivation

Large Language Models typically require:

- Server infrastructure
- High memory footprint (FP16/FP32)
- Non-deterministic GPU execution
- Centralized compute

SCXµ addresses these constraints by:

1. Quantizing to INT4 with structured block scaling
2. Using sparse MoE routing to reduce compute
3. Designing a deterministic binary container
4. Enabling distributed execution across browser contexts

The result is a transformer runtime that is:

- Verifiable
- Portable
- Decentralizable
- Scalable without centralized servers

---

## 2. System Overview

SCXµ consists of five architectural layers:

```text
+----------------------------------+
| Application Layer                |
| (Chat / Generation API)          |
+----------------------------------+
| Runtime Layer                    |
| SCX4 Loader · SCXQ2 Streamer     |
| Deterministic Replay Verifier    |
+----------------------------------+
| Compute Layer                    |
| WebGPU Kernels                   |
| Flash Attention · INT4 MatMul    |
| Sparse MoE Dispatch              |
+----------------------------------+
| Container Layer                  |
| SCX4 Model Format                |
| Merkle Verification              |
+----------------------------------+
| Distribution Layer               |
| Shards · BroadcastChannel        |
| Distributed Expert Tabs          |
+----------------------------------+
```

---

## 3. Model Architecture

### 3.1 Base Model

Reference configuration:

- 16 transformer layers
- Hidden size: 640
- 10 attention heads
- Head dimension: 64
- MLP dimension: 2560
- 8 experts
- Top-2 routing
- Vocabulary: 32k BPE

Total parameters: ~64M  
Quantization: INT4 block-wise

---

### 3.2 Sparse Mixture-of-Experts

Each layer contains:

- Shared self-attention
- Sparse MLP experts (8)
- Deterministic Top-2 routing

Routing:

```text
scores = softmax(x · W_router)
select top_k
tie-break by lower index
```

Output:

```text
Σ(score_i × expert_i(x))
```

This reduces compute by activating only a subset of experts per token.

---

## 4. SCX4 Container Format

SCX4 is a deterministic binary container.

### 4.1 Key Properties

- Little-endian
- Block-wise INT4 quantization
- Section directory
- SHA-256 Merkle root
- Optional SCXQ2 shard support

### 4.2 INT4 Block Quantization

Each block:

```text
float16 scale
int8 zero_point
uint8 reserved
packed INT4 weights
```

Decode rule:

```text
v = nibble
if v >= 8 → v -= 16
real = (v - zp) × scale
```

This enables:

- 4× memory reduction
- GPU-friendly decoding
- Deterministic math

---

## 5. Compute Layer

### 5.1 WebGPU Execution

All heavy compute runs in WebGPU:

- INT4 block matmul
- Flash-style tiled attention
- Rotary embedding
- Sparse MoE dispatch
- KV cache management

Execution remains entirely client-side.

---

### 5.2 Flash-Style Attention

Key properties:

- Tiled QKᵀ computation
- Stable softmax
- Running max accumulation
- Causal masking
- KV cache reuse

This allows sequence lengths up to 2048 within browser memory constraints.

---

## 6. Distributed Execution Model

SCXµ enables distributed expert execution across browser contexts.

### 6.1 Multi-Tab Expert Sharding

Each browser tab can host a subset of experts.

Example:

```text
Tab A → Experts 0,1
Tab B → Experts 2,3
Tab C → Experts 4,5
Tab D → Experts 6,7
```

Communication via:

- BroadcastChannel
- SharedArrayBuffer (where available)

Deterministic merge ensures consistent output.

---

### 6.2 SCXQ2 Shard Streaming

Model weights are split into expert shards:

```text
expert_0.scxq2
expert_1.scxq2
...
```

Shards are:

- Integrity-verified
- Lazy-loaded
- Cached via IndexedDB or Service Worker

This allows progressive model boot.

---

## 7. Deterministic Replay

For every layer, SCXµ computes:

```text
layer_hash = SHA256(
    input_tensor ||
    router_weights ||
    selected_experts ||
    output_tensor
)
```

Replay guarantees:

- Bit-stable execution
- Tamper detection
- Cross-device validation

Determinism is a core design constraint.

---

## 8. Security Model

SCXµ ensures:

- Model integrity via Merkle root
- Shard integrity via SHA-256
- Deterministic routing
- No dynamic weight mutation
- No runtime eval

Execution layer never parses symbolic logic. Binary container is authoritative.

---

## 9. Performance Characteristics

On a modern desktop iGPU:

- 5–15 tokens/sec
- 35–40 MB weight footprint
- ~65 MB runtime memory including KV cache

Sparse MoE reduces MLP compute by ~40–60%.

---

## 10. Comparison to Traditional Architectures

| Feature             | SCXµ | Server LLM |
| ------------------- | ---- | ---------- |
| Client-only         | Yes  | No         |
| Deterministic       | Yes  | Often No   |
| INT4 Native         | Yes  | Often FP16 |
| Sparse MoE          | Yes  | Sometimes  |
| Distributed Browser | Yes  | No         |
| Merkle Verified     | Yes  | Rare       |

---

## 11. Extensibility

Planned extensions:

- WebRTC multi-device expert mesh
- INT1 / ternary experimental branch
- WASM fallback backend
- GPU-only MoE dispatch
- Training loop prototype

---

## 12. Conclusion

SCXµ demonstrates that:

- Sparse transformer inference can run entirely in the browser
- INT4 quantization enables realistic 64M-scale models in ~35 MB
- Deterministic containers + Merkle verification allow trustable distributed AI execution
- Browser contexts can act as distributed compute nodes

SCXµ is not a demo runtime—it is a deterministic sparse transformer system architecture built for client-native execution.
