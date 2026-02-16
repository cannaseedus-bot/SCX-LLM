struct MoeParams {
  hidden_size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<storage, read> expert0: array<f32>;
@group(0) @binding(2) var<storage, read> expert1: array<f32>;
@group(0) @binding(3) var<storage, read> router: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(5) var<uniform> params: MoeParams;

@compute @workgroup_size(64)
fn moe_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.hidden_size) {
    return;
  }

  let score0 = router[0];
  let score1 = router[1];

  var weight = expert0[idx];
  if (score1 > score0) {
    weight = expert1[idx];
  }

  output_vec[idx] = input_vec[idx] * weight;
}
