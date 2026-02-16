# SCXµ — Sparse Browser Transformer Runtime

SCXµ is a deterministic, browser-native sparse transformer runtime built on:

- SCX4 (INT4 model container format)
- SCXQ2 (sharded streaming format)
- WebGPU execution
- Sparse Mixture-of-Experts (MoE)
- Deterministic replay verification

It targets a 16-layer, 64M parameter, INT4-quantized sparse GPT deployment workflow in the browser.

---

## 🔥 Features

- ⚡ INT4 quantized weights (block-wise)
- 🧠 16-layer transformer (hidden=640)
- 🧩 8 experts, Top-2 routing
- 🌊 SCXQ2 shard streaming
- 🌐 Distributed expert tabs
- 🔒 Merkle-verified model integrity
- ♻ Deterministic replay hashing
- 🚀 WebGPU Flash-style attention (runtime scaffolding)

---

## 🧠 Architecture Overview

```text
SCX4 Container
├─ Embedding (INT4)
├─ Shared Attention Blocks (INT4)
├─ Expert MLP Blocks (INT4)
├─ Router (FP16)
├─ LayerNorm (FP16)
└─ Tokenizer (BPE)
```

Execution pipeline:

```text
Token → Embedding
For each layer:
  Rotary
  Flash Attention
  Router (Top-2)
  Expert MLP Dispatch
  Residual
KV Cache maintained
Output → Sampling
```

---

## 📦 Installation

```bash
npm install
npm run dev
```

Build model artifacts:

```bash
npm run build:model
```

Verify model artifacts:

```bash
npm run verify
```

---

## 📁 Repository Layout

See [`docs/SCXMU_ARCHITECTURE.md`](docs/SCXMU_ARCHITECTURE.md).

---

## 🧪 Testing

```bash
npm test
```

Includes:

- SCX4 conformance tests
- Quantization tests
- Merkle integrity tests
- Determinism tests

---

## 🛡 Security

- All SCX4 sections are SHA256 Merkle-verified.
- Shards are integrity-checked.
- Deterministic routing supports replay safety.

---

## 🚀 Roadmap

- WebRTC multi-device expert mesh
- INT1 experimental branch
- WASM fallback path
- Service worker shard caching

---

## 📄 License

MIT
