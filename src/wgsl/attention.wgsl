struct AttentionParams {
  seq_len: u32,
  head_dim: u32,
  n_heads: u32,
  current_pos: u32,
  scale: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: AttentionParams;

@compute @workgroup_size(1)
fn attention(@builtin(global_invocation_id) gid: vec3<u32>) {
  let head = gid.x;
  if (head >= P.n_heads) {
    return;
  }

  let D = P.head_dim;
  let S = P.seq_len;

  var max_score: f32 = -1e9;
  var scores: array<f32, 2048>;

  for (var t: u32 = 0u; t < S; t = t + 1u) {
    var dot: f32 = 0.0;

    for (var i: u32 = 0u; i < D; i = i + 1u) {
      let q = Q[head * D + i];
      let k = K_cache[(head * S + t) * D + i];
      dot = dot + q * k;
    }

    dot = dot * P.scale;

    if (t > P.current_pos) {
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

  for (var t: u32 = 0u; t < S; t = t + 1u) {
    scores[t] = scores[t] / max(sum, 1e-12);
  }

  for (var i: u32 = 0u; i < D; i = i + 1u) {
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < S; t = t + 1u) {
      let v = V_cache[(head * S + t) * D + i];
      acc = acc + scores[t] * v;
    }

    OUT[head * D + i] = acc;
  }
}
