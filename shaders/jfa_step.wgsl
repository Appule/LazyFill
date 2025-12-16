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
@group(0) @binding(2) var<storage, read>       curLabels  : array<i32>;
@group(0) @binding(3) var<storage, read>       curDists   : array<f32>;

// Write (next state)
@group(0) @binding(4) var<storage, read_write> nextNearest : array<i32>;
@group(0) @binding(5) var<storage, read_write> nextLabels  : array<i32>;

// directions
@group(0) @binding(6) var<storage, read> directions : array<i32>; // packed as [dx,dy, dx,dy, ...]

// next distance
@group(0) @binding(7) var<storage, read_write> nextDists   : array<f32>;

fn clampi(v:i32, minv:i32, maxv:i32) -> i32 {
  return max(minv, min(v, maxv));
}

fn idxToXY(i:i32, width:i32) -> vec2<i32> {
  let x = i % width;
  let y = i / width;
  return vec2<i32>(x, y);
}

fn manhattan(i:i32, seedIdx:i32, width:i32) -> f32 {
  if (seedIdx < 0) { return 1e30; }
  let xy  = idxToXY(i, width);
  let sxy = idxToXY(seedIdx, width);
  let dx = abs(xy.x - sxy.x);
  let dy = abs(xy.y - sxy.y);
  return f32(dx + dy);
}

fn tieBreak(currLabel:i32, currSeed:i32, candLabel:i32, candSeed:i32) -> bool {
  if (candLabel < currLabel) { return true; }
  if (candLabel > currLabel) { return false; }
  // same label
  return candSeed < currSeed;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i : i32 = i32(gid.x);
  if (i >= uStep.N) { return; }

  // Start with current best
  var bestSeed  : i32 = curNearest[i];
  var bestLabel : i32 = curLabels[i];
  var bestDist  : f32 = curDists[i];

  let xy = idxToXY(i, uStep.width);

  for (var k:i32 = 0; k < uStep.dirLen; k = k + 1) {
    let dx = directions[2*k]   * uStep.jump;
    let dy = directions[2*k+1] * uStep.jump;
    let nx = clampi(xy.x + dx, 0, uStep.width - 1);
    let ny = clampi(xy.y + dy, 0, uStep.height - 1);
    let nIdx = ny * uStep.width + nx;

    let candSeed = curNearest[nIdx];
    if (candSeed < 0) { continue; }
    let candLabel = curLabels[candSeed];  // label of that seed
    let candDist  = manhattan(i, candSeed, uStep.width);

    if (candDist < bestDist) {
      bestDist  = candDist;
      bestSeed  = candSeed;
      bestLabel = candLabel;
    } else if (candDist == bestDist && candSeed != bestSeed) {
      if (tieBreak(bestLabel, bestSeed, candLabel, candSeed)) {
        bestDist  = candDist;
        bestSeed  = candSeed;
        bestLabel = candLabel;
      }
    }
  }

  // Commit to next buffers
  nextNearest[i] = bestSeed;
  nextLabels[i]  = bestLabel;
  nextDists[i]   = bestDist;
}
