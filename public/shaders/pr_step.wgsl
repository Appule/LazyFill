struct Params {
  width: f32, height: f32, sigma: f32, parity: f32, 
  strength: f32, min_x: f32, min_y: f32, pad: f32 
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> label: array<u32>;
@group(0) @binding(2) var<storage, read_write> h: array<u32>;
@group(0) @binding(3) var<storage, read> caps: array<f32>;
@group(0) @binding(4) var<storage, read_write> flow: array<f32>;

const V_MAX: u32 = 1000000u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 【修正】絶対座標計算
  let gx = gid.x + u32(params.min_x);
  let gy = gid.y + u32(params.min_y);

  let w = u32(params.width);
  let h_dim = u32(params.height);
  
  if (gx >= w || gy >= h_dim) { return; }

  // 【修正】パリティチェックは絶対座標で行う
  let parity = u32(params.parity);
  if ((gx + gy) % 2u != parity) { return; }

  let idx = gy * w + gx;
  let l = label[idx];
  if (l == 1u) { return; } 

  var exc: f32 = 0.0;
  if (l == 2u) { exc += params.strength; }

  if (gx > 0u) { exc += flow[(idx - 1u) * 2u]; }
  if (gy > 0u) { exc += flow[(idx - w) * 2u + 1u]; }
  exc -= flow[idx * 2u];
  exc -= flow[idx * 2u + 1u];

  if (exc <= 0.00001) { return; }

  let my_h = h[idx];
  if (my_h >= V_MAX) { return; }

  var min_h = V_MAX;

  // Right
  if (gx + 1u < w) {
    let n_idx = idx + 1u;
    let cap = caps[idx * 2u];
    let f = flow[idx * 2u];
    let res = cap - f;
    if (res > 0.00001) {
      min_h = min(min_h, h[n_idx]);
      if (my_h == h[n_idx] + 1u) {
        let push = min(exc, res);
        flow[idx * 2u] += push; 
        exc -= push;
        if (exc <= 0.00001) { return; }
      }
    }
  }
  // Down
  if (gy + 1u < h_dim) {
    let n_idx = idx + w;
    let cap = caps[idx * 2u + 1u];
    let f = flow[idx * 2u + 1u];
    let res = cap - f;
    if (res > 0.00001) {
      min_h = min(min_h, h[n_idx]);
      if (my_h == h[n_idx] + 1u) {
        let push = min(exc, res);
        flow[idx * 2u + 1u] += push; 
        exc -= push;
        if (exc <= 0.00001) { return; }
      }
    }
  }
  // Left
  if (gx > 0u) {
    let n_idx = idx - 1u;
    let cap = caps[n_idx * 2u];
    let f = flow[n_idx * 2u];
    let res = cap + f;
    if (res > 0.00001) {
      min_h = min(min_h, h[n_idx]);
      if (my_h == h[n_idx] + 1u) {
        let push = min(exc, res);
        flow[n_idx * 2u] -= push; 
        exc -= push;
        if (exc <= 0.00001) { return; }
      }
    }
  }
  // Up
  if (gy > 0u) {
    let n_idx = idx - w;
    let cap = caps[n_idx * 2u + 1u];
    let f = flow[n_idx * 2u + 1u];
    let res = cap + f;
    if (res > 0.00001) {
      min_h = min(min_h, h[n_idx]);
      if (my_h == h[n_idx] + 1u) {
        let push = min(exc, res);
        flow[n_idx * 2u + 1u] -= push; 
        exc -= push;
        if (exc <= 0.00001) { return; }
      }
    }
  }

  if (min_h >= my_h && min_h < V_MAX) {
    h[idx] = min_h + 1u;
  }
}