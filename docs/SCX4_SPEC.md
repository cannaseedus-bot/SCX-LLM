# SCX4 Model Container Specification
Version 1.0

## 1. Overview

SCX4 is a deterministic binary container format for INT4 sparse transformer models.

It supports:

- Block-wise INT4 quantization
- Shared attention blocks
- MoE expert partitioning
- Router layers (FP16)
- Optional SCXQ2 sharded streaming
- Merkle root integrity verification

## 2. Endianness

- All integers are Little Endian.
- All floats use IEEE-754.

## 3. Header (128 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0x00 | 4 | "SCX4" |
| 0x04 | 1 | Major |
| 0x05 | 1 | Minor |
| 0x06 | 2 | Flags |
| 0x08 | 4 | File Size |
| 0x0C | 4 | Layers |
| 0x10 | 4 | Hidden Size |
| 0x14 | 4 | Heads |
| 0x18 | 4 | Head Dim |
| 0x1C | 4 | MLP Dim |
| 0x20 | 4 | Experts |
| 0x24 | 4 | Top-K |
| 0x28 | 4 | Vocab |
| 0x2C | 4 | Seq Len |
| 0x30 | 4 | Block Size |
| 0x34 | 4 | Section Count |
| 0x38 | 32 | Global Merkle Root |

Remaining bytes are reserved.

## 4. Section Directory

Each entry is 16 bytes:

| Field | Type |
|-------|------|
| type | u32 |
| offset | u32 |
| size | u32 |
| merkle_offset | u32 |

Section types:

| ID | Meaning |
|----|---------|
| 0x01 | Embedding |
| 0x02 | Shared Attention |
| 0x03 | Expert Blocks |
| 0x04 | Router |
| 0x05 | LayerNorm |
| 0x09 | Tokenizer |

## 5. INT4 Quant Block

Each block contains:

```text
float16 scale
int8 zero_point
uint8 reserved
uint8 packed_weights[block_size / 2]
```

Decode rule:

```text
v = nibble
if v >= 8 → v -= 16
real = (v - zero_point) * scale
```

## 6. MoE Execution Rule

Router computes:

```text
scores = softmax(x · W_router)
select top_k (tie-break lower index)
```

Output:

```text
Σ(score_i × expert_i(x))
```

## 7. Integrity

Each section is hashed with SHA-256. A Merkle tree is built over section hashes and the root is stored in the header.

Loaders MUST verify:

- Header integrity
- Section bounds
- Merkle root

## 8. Determinism

Execution must be bit-stable given:

- Same input
- Same weights
- Same routing
