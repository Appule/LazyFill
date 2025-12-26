struct Params {
  width: f32, height: f32, step: f32, pad1: f32,
  min_x: f32, min_y: f32, pad2: f32, pad3: f32
}
struct PixelData { sx: i32, sy: i32, label: u32, pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> jfa_in: array<PixelData>;
@group(0) @binding(2) var<storage, read_write> jfa_out: array<PixelData>;

fn get_idx(x: u32, y: u32, w: u32) -> u32 { return y * w + x; }

fn manhattan(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
  return abs(x1 - x2) + abs(y1 - y2);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 【修正】オフセット加算
  let gx = gid.x + u32(params.min_x);
  let gy = gid.y + u32(params.min_y);

  let w = u32(params.width);
  let h = u32(params.height);

  if (gx >= w || gy >= h) { return; }

  let idx = get_idx(gx, gy, w);
  let k = i32(params.step);
  let my_x = i32(gx);
  let my_y = i32(gy);

  var best_seed = jfa_in[idx];
  var best_dist = manhattan(my_x, my_y, best_seed.sx, best_seed.sy);

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }

      let nx = my_x + dx * k;
      let ny = my_y + dy * k;

      if (nx >= 0 && ny >= 0 && nx < i32(w) && ny < i32(h)) {
        let n_idx = get_idx(u32(nx), u32(ny), w);
        let n_data = jfa_in[n_idx];

        if (n_data.label != 0u) {
          let d = manhattan(my_x, my_y, n_data.sx, n_data.sy);
          if (d < best_dist) {
            best_dist = d;
            best_seed = n_data;
          }
        }
      }
    }
  }
  jfa_out[idx] = best_seed;
}