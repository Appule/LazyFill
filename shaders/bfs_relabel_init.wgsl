struct Params { width: f32, height: f32, sigma: f32, parity: f32, strength: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> label: array<u32>;
@group(0) @binding(2) var<storage, read_write> dist: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let idx = gid.y * w + gid.x;
  if (gid.x >= w || gid.y >= u32(params.height)) { return; }

  // Sink starts at 0, others INF
  if (label[idx] == 1u) {
      dist[idx] = 0u;
  } else {
      dist[idx] = 1000000u;
  }
}