struct Params { width: f32, height: f32, sigma: f32, parity: f32, strength: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> caps: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read> dist_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> dist_out: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let h_dim = u32(params.height);
  let idx = gid.y * w + gid.x;
  if (gid.x >= w || gid.y >= h_dim) { return; }

  var best = dist_in[idx];

  // Look at neighbors. If Neighbor is closer to Sink, and I can push to Neighbor,
  // then my distance = Neighbor + 1.
  
  // Can I push to Right (idx+1)? (Residual: Cap - Flow > 0)
  if (gid.x + 1u < w) {
    if (caps[idx * 2u] - flow[idx * 2u] > 0.0001) {
      let nd = dist_in[idx + 1u];
      if (nd != 1000000u) { best = min(best, nd + 1u); }
    }
  }
  // Can I push to Down (idx+w)?
  if (gid.y + 1u < h_dim) {
    if (caps[idx * 2u + 1u] - flow[idx * 2u + 1u] > 0.0001) {
      let nd = dist_in[idx + w];
      if (nd != 1000000u) { best = min(best, nd + 1u); }
    }
  }
  // Can I push to Left (idx-1)? (Residual: Cap + Flow > 0)
  if (gid.x > 0u) {
    let n_idx = idx - 1u;
    if (caps[n_idx * 2u] + flow[n_idx * 2u] > 0.0001) {
      let nd = dist_in[n_idx];
      if (nd != 1000000u) { best = min(best, nd + 1u); }
    }
  }
  // Can I push to Up (idx-w)?
  if (gid.y > 0u) {
    let n_idx = idx - w;
    if (caps[n_idx * 2u + 1u] + flow[n_idx * 2u + 1u] > 0.0001) {
      let nd = dist_in[n_idx];
      if (nd != 1000000u) { best = min(best, nd + 1u); }
    }
  }
  dist_out[idx] = best;
}