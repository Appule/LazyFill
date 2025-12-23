struct Params { width: f32, height: f32, step: f32, pad: f32 }
struct PixelData { sx: i32, sy: i32, label: u32, pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> jfa_out: array<PixelData>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let h = u32(params.height);
  if (gid.x >= w || gid.y >= h) { return; }

  let idx = gid.y * w + gid.x;
  let l = labels[idx];

  // Label 1 or 2 is a Seed
  if (l != 0u) {
    // 自分自身がシード
    jfa_out[idx] = PixelData(i32(gid.x), i32(gid.y), l, 0u);
  } else {
    // Unknown: 遠くの座標で初期化 (ここでは -1 を無効値として扱う)
    // または十分に遠い座標 (例: -10000)
    jfa_out[idx] = PixelData(-10000, -10000, 0u, 0u);
  }
}