struct Params {
  width: f32, height: f32, step: f32, pad1: f32,
  min_x: f32, min_y: f32, pad2: f32, pad3: f32
}
struct PixelData { sx: i32, sy: i32, label: u32, pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> jfa_out: array<PixelData>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 【修正】オフセットを加算して画像の絶対座標を計算
  let gx = gid.x + u32(params.min_x);
  let gy = gid.y + u32(params.min_y);

  let w = u32(params.width);
  let h = u32(params.height);

  // 範囲外チェックは絶対座標で行う
  if (gx >= w || gy >= h) { return; }

  let idx = gy * w + gx;
  let l = labels[idx];

  if (l != 0u) {
    jfa_out[idx] = PixelData(i32(gx), i32(gy), l, 0u);
  } else {
    jfa_out[idx] = PixelData(-10000, -10000, 0u, 0u);
  }
}