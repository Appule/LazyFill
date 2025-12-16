// shaders/jfa_init.wgsl
struct InitUniforms {
  width  : i32,
  height : i32,
  N      : i32,
};

@group(0) @binding(0) var<uniform> uInit : InitUniforms;
@group(0) @binding(1) var<storage, read>  maskBuf    : array<i32>;
@group(0) @binding(2) var<storage, read_write> nearest : array<i32>;
@group(0) @binding(3) var<storage, read_write> labels  : array<i32>;
@group(0) @binding(4) var<storage, read_write> dists   : array<f32>;

const INF : f32 = 1e30;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i : i32 = i32(gid.x);
  if (i >= uInit.N) { return; }

  let m = maskBuf[uInit.width * (i / uInit.width) + (i % uInit.width)]; // mask[i], explicit to be clear
  let b = (m >= 1);
  nearest[i] = select(-1, i, b);
  labels[i]  = select(-1, m, b);
  dists[i]   = select(INF, 0.0, b);
}
