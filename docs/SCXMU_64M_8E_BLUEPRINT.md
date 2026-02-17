# SCXµ-64M-8E Blueprint

Production-oriented browser configuration for a deterministic sparse runtime.

## Target

- **Name:** SCXµ-64M-8E
- **Parameter budget:** ~61.8M
- **Quantization:** INT4 weights (block-wise)
- **Routing:** MoE Top-2 over 8 experts (deterministic tie-break by lower index)

## Core dimensions

- Layers: 16
- Hidden size: 640
- Heads: 10
- Head dim: 64
- MLP dim: 2560
- Experts: 8
- Top-K: 2
- Vocab: 32,000
- Seq len: 2,048
- Shared attention blocks: 8 (reused across 16 layers)

## Parameter budget (approx)

- Embedding: `32000 * 640 = 20.5M`
- Shared attention: `640 * 640 * 4 * 8 = 13.1M`
- Expert MLP total: `((640*2560)+(2560*640)) * 8 = 26.2M`
- Router + LayerNorm + misc: `~2.0M`

Total: `~61.8M`

## Storage and runtime envelope

- INT4 core weight payload: `~30.9MB`
- With quant scales + metadata: `~35MB`
- Typical runtime memory (desktop browser):
  - weights: `~35MB`
  - KV cache (INT8 @ 2k): `~20MB`
  - activations/work buffers: `~10MB`
  - total: `~65MB`

## Execution contract

For each token:

1. Shared attention block (grouped reuse)
2. Router projection (FP16)
3. Select Top-2 experts
4. Run two expert MLP paths
5. Weighted merge + residual

All execution must remain deterministic for identical inputs/weights.
