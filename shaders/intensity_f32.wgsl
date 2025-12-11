@group(0) @binding(1)
var<storage, read> in_f32: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_f32: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.length) {
    return;
  }

  out_f32[i] = max(-in_f32[i] * params.scl, 0.0);
}
