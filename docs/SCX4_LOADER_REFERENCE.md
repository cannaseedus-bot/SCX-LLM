# SCX4-µ Deterministic Loader Reference

This document captures strict loading behavior for SCX4 binaries and a tiny test-file layout.

## 1) Deterministic loader pseudocode

```text
function load_scx4(file_bytes):
    if file_bytes[0:4] != "SCX4":
        error("Invalid magic")

    major = u8(file_bytes, 0x04)
    minor = u8(file_bytes, 0x05)
    flags = u16_le(file_bytes, 0x06)

    file_size = u32_le(file_bytes, 0x08)
    if file_size != len(file_bytes):
        error("File size mismatch")

    n_layers     = u32_le(file_bytes, 0x0C)
    hidden_size  = u32_le(file_bytes, 0x10)
    n_heads      = u32_le(file_bytes, 0x14)
    head_dim     = u32_le(file_bytes, 0x18)
    mlp_dim      = u32_le(file_bytes, 0x1C)
    n_experts    = u32_le(file_bytes, 0x20)
    top_k        = u32_le(file_bytes, 0x24)
    vocab_size   = u32_le(file_bytes, 0x28)
    seq_len      = u32_le(file_bytes, 0x2C)
    block_size   = u32_le(file_bytes, 0x30)
    section_cnt  = u32_le(file_bytes, 0x34)
    global_root  = file_bytes[0x38:0x58]

    directory_offset = 128
    sections = []

    for i in range(section_cnt):
        base = directory_offset + i * 16
        type   = u32_le(file_bytes, base + 0)
        offset = u32_le(file_bytes, base + 4)
        size   = u32_le(file_bytes, base + 8)
        root_o = u32_le(file_bytes, base + 12)

        if offset + size > file_size:
            error("Invalid section bounds")

        sections.append({type, offset, size, merkle_offset: root_o})

    leaf_hashes = []
    for sec in sections:
        data = file_bytes[sec.offset:sec.offset + sec.size]
        leaf_hashes.append(SHA256(data))

    computed_root = build_merkle_root(leaf_hashes)
    if computed_root != global_root:
        error("Merkle root mismatch")

    embedding = find_section(sections, 0x01)
    attention = find_section(sections, 0x02)
    experts   = find_section(sections, 0x03)
    router    = find_section(sections, 0x04)
    layernorm = find_section(sections, 0x05)

    return {
      header,
      sections,
      gpu: {
        embedding: gpu_upload(embedding),
        attention: gpu_upload(attention),
        experts: gpu_upload(experts),
        router: gpu_upload(router),
        layernorm: gpu_upload(layernorm),
      }
    }
```

## 2) Minimal `.scx4` hex anatomy (tiny example)

The following illustrates a tiny one-layer model layout for test fixtures.

### Header start

```text
53 43 58 34        ; "SCX4"
01                 ; major
00                 ; minor
01 00              ; flags
60 00 00 00        ; file size (example)
01 00 00 00        ; layers
04 00 00 00        ; hidden
01 00 00 00        ; heads
04 00 00 00        ; head_dim
08 00 00 00        ; mlp_dim
00 00 00 00        ; experts
00 00 00 00        ; top_k
04 00 00 00        ; vocab
10 00 00 00        ; seq_len
80 00 00 00        ; block_size
01 00 00 00        ; section_count
<32 bytes merkle root>
```

### Section directory entry

```text
01 00 00 00        ; section type: embedding
80 00 00 00        ; section offset
20 00 00 00        ; section size
00 00 00 00        ; merkle offset
```

### Quant block sketch

```text
00 3C              ; fp16 scale (example)
00                 ; zero_point
00                 ; reserved
11 22 33 44 ...    ; packed int4 nibbles
```

## 3) Browser/WebGPU implementation pointers

The reference implementation in this repo is:

- `demo/scx4-loader.js` for strict header + section parsing and GPU uploads.
- `src/runtime/scx4-loader.ts` for runtime-level deterministic parsing helpers.
