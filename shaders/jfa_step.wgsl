// shaders/jfa_step.wgsl
struct StepUniforms {
  width  : i32,
  height : i32,
  N      : i32,
  jump   : i32,
  dirLen : i32,
};

@group(0) @binding(0) var<uniform> uStep : StepUniforms;

// Read (current state)
@group(0) @binding(1) var<storage, read>       curNearest : array<i32>;
@group(0) @binding(3) var<storage, read>       curDists   : array<i32>;

// Write (next state)
@group(0) @binding(4) var<storage, read_write> nextNearest : array<i32>;

// directions
@group(0) @binding(6) var<storage, read> directions : array<i32>; // packed as [dx,dy, dx,dy, ...]

// next distance
@group(0) @binding(7) var<storage, read_write> nextDists   : array<i32>;

fn clampi(v:i32, minv:i32, maxv:i32) -> i32 {
  return max(minv, min(v, maxv));
}

fn idxToXY(i:i32, width:i32) -> vec2<i32> {
  let x = i % width;
  let y = i / width;
  return vec2<i32>(x, y);
}

fn manhattan(i:i32, seedIdx:i32, width:i32) -> i32 {
  if (seedIdx < 0) { return 100000; }
  let xy  = idxToXY(i, width);
  let sxy = idxToXY(seedIdx, width);
  let dx = abs(xy.x - sxy.x);
  let dy = abs(xy.y - sxy.y);
  return dx + dy;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i : i32 = i32(gid.x);
  if (i >= uStep.N) { return; }

  // Start with current best
  var bestSeed  : i32 = curNearest[i];
  var bestDist  : i32 = curDists[i];

  let xy = idxToXY(i, uStep.width);

  for (var k:i32 = 0; k < uStep.dirLen; k = k + 1) {
    let dx = directions[2*k]   * uStep.jump;
    let dy = directions[2*k+1] * uStep.jump;
    let nx = clampi(xy.x + dx, 0, uStep.width - 1);
    let ny = clampi(xy.y + dy, 0, uStep.height - 1);
    let nIdx = ny * uStep.width + nx;

    let candSeed = curNearest[nIdx];
    if (candSeed == 0 || candSeed == 2) { continue; }
    let candDist  = manhattan(i, candSeed, uStep.width);

    if (candDist < bestDist) {
      bestDist  = candDist;
      bestSeed  = candSeed;
    }
  }

  // Commit to next buffers
  nextNearest[i] = bestSeed;
  nextDists[i]   = bestDist;
}
