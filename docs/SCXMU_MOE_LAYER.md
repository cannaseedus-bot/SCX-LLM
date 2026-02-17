# SCXµ as the Deterministic MoE Layer

SCXµ is the routing intelligence layer on top of SCX-TP INT4 weights.

```text
SCXµ = μRouter + μExperts + μDispatch
```

- **μRouter**: lightweight router projection (FP16)
- **μExperts**: expert MLP blocks in INT4
- **μDispatch**: deterministic top-k selection and weighted combine

## High-level flow

```text
Input
  -> Embedding
  -> Router (FP16)
  -> Top-K select (deterministic ties)
  -> Expert dispatch (INT4 expert buffers)
  -> Weighted merge
  -> Next layer
```

## Runtime opcode surface

The execution layer uses fixed opcodes:

- `0x20` ROUTE
- `0x21` SELECT
- `0x22` DISPATCH
- `0x23` COMBINE

(Defined in `src/distributed/tab-coordinator.ts`.)

## Deterministic top-k contract

Given router scores:

1. sort by score descending,
2. break ties by lower expert index,
3. take first `k` experts.

This ensures stable replay and hashable routing traces.

## Deterministic combine contract

Given selected experts and scores:

```text
output = Σ(score_i * expert_i_output)
```

Merge order is deterministic: expert ids sorted ascending.

## Replay/verifiability contract

Two hashes are recorded:

1. **Route hash**

```text
SHA256(input_tensor || router_weights || layer_id)
```

2. **Layer hash**

```text
SHA256(input_tensor || router_weights || selected_experts || output_tensor)
```

Both are exposed by `src/runtime/replay-verifier.ts` and can be stored as a replay trace.
