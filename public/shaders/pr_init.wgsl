struct Params { width: f32, height: f32, sigma: f32, parity: f32, strength: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> intensity: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read_write> h: array<u32>;
@group(0) @binding(4) var<storage, read_write> caps: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow: array<f32>;
@group(0) @binding(6) var<storage, read> dist: array<i32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let idx = gid.y * w + gid.x;
  if (gid.x >= w || gid.y >= u32(params.height)) { return; }

  // 1. Capacity (Float)
  let val = intensity[idx];
  let inv_s = 1.0 / (2.0 * params.sigma * params.sigma);
  
  // Right
  var cr = 0.0;
  if (gid.x + 1u < w) {
    let d = val + intensity[idx + 1u];
    cr = exp(-(d) * inv_s) + 0.000001; 
  }
  // Down
  var cd = 0.0;
  if (gid.y + 1u < u32(params.height)) {
    let d = val + intensity[idx + w];
    cd = exp(-(d) * inv_s) + 0.000001;
  }
  caps[idx * 2u] = cr;
  caps[idx * 2u + 1u] = cd;

  // 2. Flow Init
  flow[idx * 2u] = 0.0;
  flow[idx * 2u + 1u] = 0.0;

  // 3. Height Init
  let l = label[idx];
  if (l == 1u) { // Sink
    h[idx] = 0u;
  } else {
    // Source & Unknown
    h[idx] = u32(max(0, dist[idx]));
  }
}