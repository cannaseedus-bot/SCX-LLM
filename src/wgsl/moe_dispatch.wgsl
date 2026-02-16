const MAX_EXPERTS: u32 = 64u;

struct MoeParams {
  hidden_size: u32,
  n_experts: u32,
  top_k: u32,
  token_offset: u32,
};

@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<storage, read> router_scores: array<f32>;
@group(0) @binding(2) var<storage, read> expert_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> final_out: array<f32>;
@group(0) @binding(5) var<uniform> params: MoeParams;

fn compute_expert(base: u32, i: u32, token_offset: u32) -> f32 {
  let hidden_size = params.hidden_size;
  var acc: f32 = 0.0;

  for (var col: u32 = 0u; col < hidden_size; col = col + 1u) {
    let idx = base + i * hidden_size + col;
    acc = acc + input_vec[token_offset + col] * expert_weights[idx];
  }

  return acc;
}

fn run_expert(expert_id: u32, weight: f32, token_offset: u32) {
  let base = expert_offsets[expert_id];

  for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
    let val = compute_expert(base, i, token_offset);
    final_out[token_offset + i] = final_out[token_offset + i] + weight * val;
  }
}

@compute @workgroup_size(1)
fn moe_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > 0u) {
    return;
  }

  let n_experts = min(params.n_experts, MAX_EXPERTS);
  if (n_experts == 0u) {
    return;
  }

  var best1: u32 = 0u;
  var best2: u32 = 0u;
  var s1: f32 = -1e30;
  var s2: f32 = -1e30;

  for (var i: u32 = 0u; i < n_experts; i = i + 1u) {
    let s = router_scores[i];

    if (s > s1) {
      best2 = best1;
      s2 = s1;
      best1 = i;
      s1 = s;
    } else if (s > s2) {
      best2 = i;
      s2 = s;
    }
  }

  run_expert(best1, s1, params.token_offset);
  if (params.top_k > 1u && n_experts > 1u) {
    run_expert(best2, s2, params.token_offset);
  }
}
