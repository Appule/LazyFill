struct Params { width: f32, height: f32, step: f32, pad: f32 }
struct PixelData { sx: i32, sy: i32, label: u32, pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> jfa_in: array<PixelData>;
@group(0) @binding(2) var<storage, read_write> label_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> dist_out: array<i32>;

fn manhattan(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
  return abs(x1 - x2) + abs(y1 - y2);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let h = u32(params.height);
  if (gid.x >= w || gid.y >= h) { return; }

  let idx = gid.y * w + gid.x;
  let data = jfa_in[idx];

  label_out[idx] = data.label;

  if (data.label != 0u) {
    let d = manhattan(i32(gid.x), i32(gid.y), data.sx, data.sy);
    dist_out[idx] = d;
  } else {
    dist_out[idx] = 2000000000; 
  }
}