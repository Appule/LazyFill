@group(0) @binding(1)
var<storage, read> inputRGBA: array<u32>;

@group(0) @binding(2)
var<storage, read_write> outputLuma: array<f32>;

fn unpack_rgba(px: u32) -> vec4<f32> {
  let r: f32 = f32((px >> 0u)  & 0xFFu) / 255.0;
  let g: f32 = f32((px >> 8u)  & 0xFFu) / 255.0;
  let b: f32 = f32((px >> 16u) & 0xFFu) / 255.0;
  let a: f32 = f32((px >> 24u) & 0xFFu) / 255.0;
  return vec4<f32>(r, g, b, a);
}

fn luminance(r: f32, g: f32, b: f32) -> f32 {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

fn blendWithWhite(c: f32, a: f32) -> f32 {
  return c * a + (1.0 - a) * 1.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  // 範囲外チェック
  if (x >= params.width || y >= params.height) {
    return;
  }

  // 1次元インデックスを計算
  let i = y * params.width + x;

  let px: u32 = inputRGBA[i];
  let ch: vec4<f32> = unpack_rgba(px);

  let r_blend: f32 = blendWithWhite(ch.r, ch.a);
  let g_blend: f32 = blendWithWhite(ch.g, ch.a);
  let b_blend: f32 = blendWithWhite(ch.b, ch.a);

  outputLuma[i] = luminance(r_blend, g_blend, b_blend);
}