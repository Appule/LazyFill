struct Params {
  length: u32, // ピクセル数
}

@group(0) @binding(0)
var<storage, read> rgba: array<u32>;   // 入力 RGBA packed u32

@group(0) @binding(1)
var<storage, read_write> outRgba: array<u32>; // 出力 RGBA packed u32

@group(0) @binding(2)
var<uniform> params: Params;

fn unpack_rgba(px: u32) -> vec4<u32> {
  let r: u32 = (px >> 0u)  & 0xFFu;
  let g: u32 = (px >> 8u)  & 0xFFu;
  let b: u32 = (px >> 16u) & 0xFFu;
  let a: u32 = (px >> 24u) & 0xFFu;
  return vec4<u32>(r, g, b, a);
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 0u) | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= params.length) {
    return;
  }

  let px: u32 = rgba[i];
  let ch: vec4<u32> = unpack_rgba(px);

  let l: f32 = 0.299 * f32(ch.x) + 0.587 * f32(ch.y) + 0.114 * f32(ch.z);
  let y: u32 = u32(clamp(l, 0.0, 255.0));

  outRgba[i] = pack_rgba(y, y, y, 255u);
}
