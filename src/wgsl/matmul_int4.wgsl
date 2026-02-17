const MAX_HIDDEN_SIZE: u32 = 4096u;

struct TransformerBlockParams {
  hidden_size: u32,
  block_size: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(4) var<uniform> params: TransformerBlockParams;

fn dequant_int4(byte_val: u32, high: bool, scale: f32, zp: i32) -> f32 {
  var v: i32;
  if (high) {
    v = i32((byte_val >> 4u) & 0xFu);
  } else {
    v = i32(byte_val & 0xFu);
  }

  if (v >= 8) {
    v = v - 16;
  }

  return f32(v - zp) * scale;
}

@compute @workgroup_size(64)
fn transformer_block(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let hidden_size = params.hidden_size;
  let block_size = max(params.block_size, 2u);

  if (row >= hidden_size || hidden_size > MAX_HIDDEN_SIZE) {
    return;
  }

  var acc: f32 = 0.0;

  for (var col: u32 = 0u; col < hidden_size; col = col + 1u) {
    let idx = row * hidden_size + col;
    let packed = packed_weights[idx / 2u];
    let is_high = (idx % 2u) == 0u;
    let scale = scales[idx / block_size];

    let w = dequant_int4(packed, is_high, scale, 0);
    acc = acc + input_vec[col] * w;
  }

  let x = acc;
  let gelu = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
  output_vec[row] = gelu + input_vec[row];
}
