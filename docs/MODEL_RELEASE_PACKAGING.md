# Model Release Packaging Format

Recommended release structure:

```text
release/
  scxmu-64m-v1/
    model.scx4
    shards/
      expert_0.scxq2
      ...
    tokenizer/
      vocab.json
      merges.txt
    manifest.json
    SHA256SUMS
    signature.asc
```

## `manifest.json`

```json
{
  "name": "scxmu-64m",
  "version": "1.0.0",
  "architecture": "16-layer sparse MoE",
  "hidden": 640,
  "experts": 8,
  "top_k": 2,
  "quantization": "INT4",
  "container": "SCX4",
  "sharded": true,
  "merkle_root": "<hex>",
  "tokenizer": "BPE"
}
```

## Integrity file

`SHA256SUMS`:

```text
<hash>  model.scx4
<hash>  shards/expert_0.scxq2
...
```

An ASCII-armored detached signature (`signature.asc`) is optional but recommended.
