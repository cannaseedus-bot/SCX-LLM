struct MoeParams {
  hidden_size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

struct AttentionParams {
  seq_len: u32,
  head_dim: u32,
  n_heads: u32,
  current_pos: u32,
  scale: f32,
  _apad0: f32,
  _apad1: f32,
  _apad2: f32,
};

@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<storage, read> expert0: array<f32>;
@group(0) @binding(2) var<storage, read> expert1: array<f32>;
@group(0) @binding(3) var<storage, read> router: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(5) var<uniform> moe_params: MoeParams;

@compute @workgroup_size(64)
fn moe_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= moe_params.hidden_size) {
    return;
  }

  var weight = expert0[idx];
  if (router[1] > router[0]) {
    weight = expert1[idx];
  }

  output_vec[idx] = input_vec[idx] * weight;
}

@group(1) @binding(0) var<storage, read> Q: array<f32>;
@group(1) @binding(1) var<storage, read> K_cache: array<f32>;
@group(1) @binding(2) var<storage, read> V_cache: array<f32>;
@group(1) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(1) @binding(4) var<uniform> attn_params: AttentionParams;

@compute @workgroup_size(1)
fn attention(@builtin(global_invocation_id) gid: vec3<u32>) {
  let head = gid.x;
  if (head >= attn_params.n_heads) {
    return;
  }

  let D = attn_params.head_dim;
  let S = attn_params.seq_len;

  var max_score: f32 = -1e9;
  var scores: array<f32, 2048>;

  for (var t: u32 = 0u; t < S; t = t + 1u) {
    var dot: f32 = 0.0;
    for (var i: u32 = 0u; i < D; i = i + 1u) {
      let q = Q[head * D + i];
      let k = K_cache[(head * S + t) * D + i];
      dot = dot + q * k;
    }

    dot = dot * attn_params.scale;
    if (t > attn_params.current_pos) {
      dot = -1e9;
    }

    scores[t] = dot;
    if (dot > max_score) {
      max_score = dot;
    }
  }

  var sum: f32 = 0.0;
  for (var t: u32 = 0u; t < S; t = t + 1u) {
    scores[t] = exp(scores[t] - max_score);
    sum = sum + scores[t];
  }

  for (var i: u32 = 0u; i < D; i = i + 1u) {
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < S; t = t + 1u) {
      let v = V_cache[(head * S + t) * D + i];
      acc = acc + (scores[t] / max(sum, 1e-12)) * v;
    }

    OUT[head * D + i] = acc;
  }
}

@group(2) @binding(0) var<storage, read_write> QK: array<f32>;
@group(2) @binding(1) var<storage, read> sin_cache: array<f32>;
@group(2) @binding(2) var<storage, read> cos_cache: array<f32>;

@compute @workgroup_size(64)
fn rotary(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pair = idx / 2u;
  let is_even = (idx % 2u) == 0u;

  let sin_theta = sin_cache[pair];
  let cos_theta = cos_cache[pair];

  let even = QK[pair * 2u];
  let odd = QK[pair * 2u + 1u];

  if (is_even) {
    QK[pair * 2u] = even * cos_theta - odd * sin_theta;
  } else {
    QK[pair * 2u + 1u] = even * sin_theta + odd * cos_theta;
  }
}
