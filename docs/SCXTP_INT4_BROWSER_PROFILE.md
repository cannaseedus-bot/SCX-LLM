# SCX-TP-INT4 Browser GPT Profile (v1.0)

This profile defines a hardware-realistic browser execution target for SCX-TP-INT4.

## Design goals

- Deterministic execution
- WebGPU-first lowering
- WASM SIMD fallback possible
- Merkle-verifiable binary payloads
- SCX as projection/control layer only (no symbol interpretation in runtime)

## Runtime constraints

- INT4 packed weights (2 values per byte)
- FP16/FP32 accumulators
- KV cache in INT8
- No server requirement for inference loop
- Browser memory envelope suitable for desktop iGPU

## Canonical INT4 representation

- signed range: `[-8, 7]`
- high nibble = first value, low nibble = second value
- decode rule:

```text
n = nibble
if n >= 8 then n -= 16
```

## Block quantization (required)

```c
struct QuantBlock {
  float16 scale;
  int8    zero_point;
  uint8   reserved;
  uint8   packed_weights[block_size/2];
}
```

Default `block_size = 128`.

## Layer packing order

Per transformer layer, tensors are stored in a deterministic, fixed order:

1. `Q_proj`
2. `K_proj`
3. `V_proj`
4. `O_proj`
5. `MLP_up`
6. `MLP_gate`
7. `MLP_down`
8. `LayerNorm1` (FP16)
9. `LayerNorm2` (FP16)

## Attention numerical contract

1. INT4 weights dequantized to FP16/FP32
2. QKᵀ computed in FP16/FP32 accumulation
3. softmax in FP16/FP32
4. V projection and output projection (INT4 weights, FP accumulation)

Softmax must not be quantized.

## SCX projection boundary

SCX graph/operators are compiled into dispatch opcodes and kernel launches.
Runtime does not parse symbolic SCX terms during execution.

## Generation loop contract

Model forward executes on GPU; sampling stays CPU-side:

```js
while (!eos) {
  const logits = await model.forward(tokens);
  const next = sample(logits, temperature);
  tokens.push(next);
}
```

## Optional SCXQ2 lanes

SCXQ2 compressed lanes may be used as transport/storage representation.
Canonical execution representation remains deterministic INT4 quant blocks.
