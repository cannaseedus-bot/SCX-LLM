# SCX-TP v1.1 Specification (SCXT)

SCXT is a deterministic tensor-pack execution container. SCX symbols are projection/UI only; binary opcodes are authoritative.

## Canonical binary layout (little endian)

```text
0x00  4B   magic = "SCXT"
0x04  1B   version_major
0x05  1B   version_minor
0x06  2B   flags
0x08 16B   dims = [d0,d1,d2,d3] (4 x uint32)
0x18  4B   cell_count
0x1C  4B   layer_mask
0x20  nB   binary layer (uint8 opcodes)
...   nB   text layer (uint16 codepoints)
...   nB   weight layer (float32)
...  32B   merkle_root
```

- Binary payload contains no JSON.
- Human metadata must be sidecar (`.meta.json`).

## Deterministic opcode table (5-bit)

| Opcode | Name | Projection |
|---:|---|---|
| 0x00 | NOP | - |
| 0x01 | ADD | ⊕ |
| 0x02 | MUL | ⊗ |
| 0x03 | WMIX | ⚖ |
| 0x04 | ENC | 🔐 |
| 0x05 | DEC | 🔓 |
| 0x06 | SLICE | ✂ |
| 0x07 | ACT_TANH | - |
| 0x08 | ACT_STEP | - |
| 0x09 | ACT_NORM | - |
| 0x0A | SEARCH | - |
| 0x0B | ANALYZE | - |
| 0x0C | LOAD | - |
| 0x0D | SAVE | - |
| 0x0E | CMP | - |
| 0x0F | HALT | - |

## 4D indexing rule (canonical)

To map `(w,x,y,z)` to a linear address with dims `[Dw,Dx,Dy,Dz]`:

```text
index = (((w * Dx) + x) * Dy + y) * Dz + z
```

No 8-bit coordinate packing is allowed for canonical addressing.

## WebGPU/SIMD lowering contract

- Runtime executes numeric opcodes and tensor addresses only.
- SCX symbolic forms are compile-time/UI projection.
- Kernels are compiled once and dispatched with typed buffers.

## Integrity and determinism

- Merkle root is computed over execution payload sections.
- Same binary + same input must yield same opcode path and output trace hashes.
