@group(0) @binding(1)
var<storage, read> in_f32: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_f32: array<f32>;

const K_SIZE: u32 = 5u;
const HALF: u32 = 2u;
const WEIGHT: f32 = 273.0;
const GAU5x5: array<f32, 25u> = array<f32, 25u>(
  1.0,  4.0,  7.0,  4.0,  1.0,
  4.0, 16.0, 26.0, 16.0,  4.0,
  7.0, 26.0, 41.0, 26.0,  7.0,
  4.0, 16.0, 26.0, 16.0,  4.0,
  1.0,  4.0,  7.0,  4.0,  1.0,
);

fn idx(x: i32, y: i32, w: i32, h: i32) -> u32 {
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  return u32(cy * w + cx);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w: u32 = params.width;
  let h: u32 = params.height;

  let x: u32 = gid.x;
  let y: u32 = gid.y;
  if (x >= w || y >= h) {
    return;
  }

  let w_i: i32 = i32(w);
  let h_i: i32 = i32(h);
  let x_i: i32 = i32(x);
  let y_i: i32 = i32(y);

  var acc: f32 = 0.0;
  for (var ky: u32 = 0u; ky < K_SIZE; ky++) {
    for (var kx: u32 = 0u; kx < K_SIZE; kx++) {
      let ox: i32 = x_i + i32(kx) - i32(HALF);
      let oy: i32 = y_i + i32(ky) - i32(HALF);
      let src_idx: u32 = idx(ox, oy, w_i, h_i);
      let v: f32 = in_f32[src_idx];
      let k: f32 = GAU5x5[ky * K_SIZE + kx];
      acc += v * k;
    }
  }

  let out_index: u32 = y * w + x;
  out_f32[out_index] = acc / WEIGHT;
}
