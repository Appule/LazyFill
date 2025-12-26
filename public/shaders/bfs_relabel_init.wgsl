struct Params {
  width: f32, height: f32, sigma: f32, parity: f32, 
  strength: f32, min_x: f32, min_y: f32, pad: f32 
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> label: array<u32>;
@group(0) @binding(2) var<storage, read_write> dist: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 【修正】絶対座標計算
  let gx = gid.x + u32(params.min_x);
  let gy = gid.y + u32(params.min_y);

  let w = u32(params.width);
  if (gx >= w || gy >= u32(params.height)) { return; }
  
  let idx = gy * w + gx;

  if (label[idx] == 1u) {
    dist[idx] = 0u;
  } else {
    dist[idx] = 1000000u;
  }
}