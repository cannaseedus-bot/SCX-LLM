# SCX4-µ64-8E Binary Layout Specification
Version 1.1 (authoritative runtime mapping)

SCX4 is a deterministic, Merkle-verifiable, hardware-mappable binary container.
Execution payload is binary-only (no JSON payload sections).

## 0. Endianness

- Integer fields: **Little Endian**.
- Floating-point fields: IEEE-754 FP16 / FP32.

## 1. File Structure Overview

```text
[ File Header ]
[ Section Directory ]
[ Embedding Section ]
[ Attention Section ]
[ Expert Section ]
[ Router Section ]
[ LayerNorm Section ]
[ Quantization Tables ]
[ Optional SCXQ2 Lanes ]
[ Merkle Tree Nodes ]
```

## 2. File Header (fixed 128 bytes)

| Offset | Size | Type    | Field |
|---|---:|---|---|
| 0x00 | 4  | char[4] | `"SCX4"` |
| 0x04 | 1  | u8  | major |
| 0x05 | 1  | u8  | minor |
| 0x06 | 2  | u16 | flags |
| 0x08 | 4  | u32 | file_size |
| 0x0C | 4  | u32 | n_layers |
| 0x10 | 4  | u32 | hidden_size |
| 0x14 | 4  | u32 | n_heads |
| 0x18 | 4  | u32 | head_dim |
| 0x1C | 4  | u32 | mlp_dim |
| 0x20 | 4  | u32 | n_experts |
| 0x24 | 4  | u32 | top_k |
| 0x28 | 4  | u32 | vocab_size |
| 0x2C | 4  | u32 | seq_len |
| 0x30 | 4  | u32 | block_size |
| 0x34 | 4  | u32 | section_count |
| 0x38 | 32 | bytes | global_merkle_root |
| 0x58 | 72 | reserved | zero |

## 3. Flags bitmask

- `0x0001` INT4 weights
- `0x0002` MoE enabled
- `0x0004` KV INT8
- `0x0008` SCXQ2 compressed
- `0x0010` Shared attention

## 4. Section Directory

Immediately after the 128-byte header.

```c
struct SectionEntry {
  u32 type;
  u32 offset;
  u32 size;
  u32 merkle_root_offset;
}
```

- Size: 16 bytes per entry.
- **Required alignment:** every section offset is 64-byte aligned.
- **Required safety:** no overlaps, no out-of-range slices.

Section type IDs:

- `0x01` EMBEDDING
- `0x02` ATTENTION
- `0x03` EXPERT
- `0x04` ROUTER
- `0x05` LAYERNORM
- `0x06` QUANT_TABLE
- `0x07` SCXQ2_LANE
- `0x08` MERKLE_TREE

## 5. INT4 Quant Block (canonical)

```c
struct QuantBlock {
  float16 scale;
  int8    zero_point;
  uint8   reserved;
  uint8   packed_weights[block_size / 2];
}
```

Default `block_size = 128`:

- 128 weights
- 64 packed bytes
- + 2-byte scale + 1-byte zp + 1-byte reserved
- Total = 68 bytes per block

Canonical decode:

```text
high = (byte >> 4) & 0xF
low  = byte & 0xF
if nibble >= 8 -> nibble -= 16
real = (nibble - zero_point) * scale
```

## 6. Embedding Section

```text
QuantBlock × ceil((vocab_size * hidden_size) / block_size)
```

Flattened row-major `[token][hidden]`.

## 7. Attention Section

When shared attention is enabled, section stores unique per-layer projection blocks in sequence:

```text
Layer0_Q, Layer0_K, Layer0_V, Layer0_O, Layer1_Q, ...
```

Each matrix flattened row-major `[out_dim][in_dim]`, quantized as QuantBlocks.

## 8. Expert Section

Experts are contiguous:

```text
for expert in n_experts:
  for layer in n_layers:
    MLP_up   [mlp_dim][hidden]
    MLP_down [hidden][mlp_dim]
```

All INT4 quantized; router/gating weights are not stored here.

## 9. Router Section

Router matrices are FP16:

```text
[layer][hidden][expert]
```

Shape per layer: `hidden_size * n_experts`.

## 10. LayerNorm Section

Per layer:

- `gamma: float16[hidden]`
- `beta: float16[hidden]`

Plus final layer norm.

## 11. Quantization Table Section (optional)

Optional dynamic scales, e.g.:

```text
float16 global_scales[n_layers]
```

## 12. SCXQ2 Lane Section (optional)

If flag `0x0008` set, compressed lanes may be used:

```c
struct LaneChunk {
  u32 id;
  u32 offset;
  u32 size;
  u32 hash;
}
```

Loader reconstructs canonical QuantBlocks before execution.

## 13. Merkle Tree Section

- Leaf hash = `SHA256(section_bytes)`.
- Tree built over section hashes.
- Header root (`0x38..0x58`) must match computed root.

## 14. Runtime mapping contract

Typical fixed mapping:

- GPU buffer 0: embeddings
- GPU buffer 1: attention
- GPU buffer 2: experts
- GPU buffer 3: router
- GPU buffer 4: layernorm

Offsets are pointer arithmetic from section table.

## 15. Deterministic MoE contract

```text
scores = softmax(x · W_router)
select top_k (tie-break lower index)
output = Σ(score_i * expert_i_output)
```

## 16. Loader integrity requirements

Loader MUST reject on:

1. invalid magic,
2. incompatible version,
3. file size mismatch,
4. invalid section ranges,
5. overlapping sections,
6. misaligned section offsets,
7. Merkle root mismatch.

## 17. Execution contract

Execution stack must:

- not mutate weight buffers,
- not reorder quant blocks,
- not introduce format-dependent dynamic reinterpretation.

Binary layout is the source of truth.
