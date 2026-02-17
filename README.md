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


## 🧾 SCX4 spec binary vs repository `.b64` text asset

`SCX4` is still a **binary container format** by specification (`magic`, fixed header, section directory, and binary section payloads).

In this repository, the demo model is stored as `demo/model.scx4.b64` for GitHub-friendly review/diff behavior (text instead of a committed raw binary blob).

What this means in practice:

- **Spec/runtime truth**: SCX4 remains binary on disk/in memory for parsing and execution.
- **Repo storage convenience**: the demo checks in a Base64 text wrapper of the same bytes.
- **Demo loader behavior**: `demo/scx4-loader.js` fetches `.b64`, decodes it back to the original bytes, then parses it exactly as SCX4 binary.
- **Build/verify tooling**: `scripts/build-scx4.js` and `scripts/verify-scx4.js` operate on true binary `.scx4` files.

So the `.b64` file is only a source-control transport representation, not a format change to SCX4 itself.

---

## 📁 Repository Layout

See [`docs/SCXMU_ARCHITECTURE.md`](docs/SCXMU_ARCHITECTURE.md).

For deterministic SCX4 loader behavior and tiny-file hex anatomy, see [`docs/SCX4_LOADER_REFERENCE.md`](docs/SCX4_LOADER_REFERENCE.md).

For the production browser model sizing/constraints, see [`docs/SCXMU_64M_8E_BLUEPRINT.md`](docs/SCXMU_64M_8E_BLUEPRINT.md).

For SCXµ routing-layer semantics and deterministic MoE contracts, see [`docs/SCXMU_MOE_LAYER.md`](docs/SCXMU_MOE_LAYER.md).

For browser execution constraints and canonical INT4 runtime rules, see [`docs/SCXTP_INT4_BROWSER_PROFILE.md`](docs/SCXTP_INT4_BROWSER_PROFILE.md).

For SCX-TP opcode/addressing/binary contracts, see [`docs/SCXT_SPEC.md`](docs/SCXT_SPEC.md) and [`docs/scxt.schema.json`](docs/scxt.schema.json).

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
