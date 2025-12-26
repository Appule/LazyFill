@group(0) @binding(1)
var<storage, read> inputLuma: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outputRGBA: array<u32>;

fn luma_to_u8(l: f32) -> u32 {
  let clamped = clamp(l, 0.0, 1.0);
  let scaled  = clamped * 255.0;
  let rounded = u32(clamp(floor(scaled + 0.5), 0.0, 255.0));
  return rounded;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  if (x >= params.width || y >= params.height) {
    return;
  }

  let i = y * params.width + x;

  let l = inputLuma[i];
  let v = luma_to_u8(l);

  // R=G=B=v, A=255
  let r: u32 = v;
  let g: u32 = v << 8u;
  let b: u32 = v << 16u;
  let a: u32 = 255u << 24u;

  outputRGBA[i] = (r | g | b | a);
}