struct Params {
  width: f32,
  height: f32,
  border_thickness: f32,
  pad: f32
}

struct Color {
  r: f32, g: f32, b: f32, a: f32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> markers: array<i32>;
@group(0) @binding(2) var<storage, read> palette: array<Color>;
@group(0) @binding(3) var<storage, read_write> outputBuffer: array<u32>;

fn get_idx(x: u32, y: u32, w: u32) -> u32 {
  return y * w + x;
}

fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  let ur = u32(clamp(r * 255.0, 0.0, 255.0));
  let ug = u32(clamp(g * 255.0, 0.0, 255.0));
  let ub = u32(clamp(b * 255.0, 0.0, 255.0));
  let ua = u32(clamp(a * 255.0, 0.0, 255.0));
  return (ua << 24u) | (ub << 16u) | (ug << 8u) | ur;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let h = u32(params.height);
  
  if (gid.x >= w || gid.y >= h) { return; }
  
  let idx = get_idx(gid.x, gid.y, w);
  let my_id = markers[idx];

  // Case 1: 自分がマーカーなら、その色をそのまま出力 (つぶさない)
  if (my_id != 0) {
    let c = palette[my_id];
    outputBuffer[idx] = pack_color(c.r, c.g, c.b, 1.0); // 不透明
    return;
  }

  // Case 2: 自分が背景(0)なら、周囲にマーカーがあるか探す (Outer Border)
  var found_dist1 = false; // 直近(白枠用)
  var found_dist2 = false; // その外側(黒枠用)

  let x = i32(gid.x);
  let y = i32(gid.y);
  let iw = i32(w);
  let ih = i32(h);

  // 5x5範囲を探索 (-2 ~ +2)
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) { continue; } // 自分はスキップ

      let nx = x + dx;
      let ny = y + dy;

      // 画面内チェック
      if (nx >= 0 && nx < iw && ny >= 0 && ny < ih) {
        let n_idx = get_idx(u32(nx), u32(ny), w);
        let n_id = markers[n_idx];

        // 周囲にマーカーが存在するか？
        if (n_id != 0) {
          // チェビシェフ距離 (max(abs(dx), abs(dy))) を使用して正方形の枠を作る
          let dist = max(abs(dx), abs(dy));
          
          if (dist == 1) { 
            found_dist1 = true; 
            // 距離1が見つかれば、距離2の判定を待つ必要はないが、
            // breakするとループ終了処理が複雑になるのでフラグだけ立てる
          }
          else if (dist == 2) { 
            found_dist2 = true; 
          }
        }
      }
    }
  }

  // 色決定: 優先順位は 近距離(白) > 遠距離(黒) > 背景(透明)
  if (found_dist1) {
    // マーカーの直近 (外側1px) -> 白
    outputBuffer[idx] = pack_color(1.0, 1.0, 1.0, 1.0);
  } else if (found_dist2) {
    // そのさらに外側 (外側2px) -> 黒
    outputBuffer[idx] = pack_color(0.0, 0.0, 0.0, 1.0);
  } else {
    // 何もなし -> 透明
    outputBuffer[idx] = 0u;
  }
}