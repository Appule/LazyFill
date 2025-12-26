struct Params {
  width: f32, height: f32, sigma: f32, parity: f32, 
  strength: f32, min_x: f32, min_y: f32, pad: f32 
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> intensity: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read_write> h: array<u32>;
@group(0) @binding(4) var<storage, read_write> caps: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow: array<f32>;
@group(0) @binding(6) var<storage, read> dist: array<i32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 【修正】絶対座標計算
  let gx = gid.x + u32(params.min_x);
  let gy = gid.y + u32(params.min_y);

  let w = u32(params.width);
  let h_dim = u32(params.height);

  if (gx >= w || gy >= h_dim) { return; }

  let idx = gy * w + gx;

  // Capacity計算 (グローバル座標で隣接画素を参照)
  let val = intensity[idx];
  let inv_s = 1.0 / (2.0 * params.sigma * params.sigma);
  
  var cr = 0.0;
  if (gx + 1u < w) {
    let d = val + intensity[idx + 1u];
    cr = exp(-(d) * inv_s) + 0.000001; 
  }
  
  var cd = 0.0;
  if (gy + 1u < h_dim) {
    let d = val + intensity[idx + w];
    cd = exp(-(d) * inv_s) + 0.000001;
  }
  caps[idx * 2u] = cr;
  caps[idx * 2u + 1u] = cd;

  flow[idx * 2u] = 0.0;
  flow[idx * 2u + 1u] = 0.0;

  let l = label[idx];
  if (l == 1u) { 
    h[idx] = 0u;
  } else {
    h[idx] = u32(max(0, dist[idx]));
  }
}