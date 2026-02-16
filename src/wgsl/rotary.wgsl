@group(0) @binding(0) var<storage, read_write> QK: array<f32>;
@group(0) @binding(1) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>;

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
