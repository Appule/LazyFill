// Jump Flooding Algorithm (CPU, JavaScript)
// Grid coordinates: (x, y) with flatten index i = y * width + x

/**
 * Create initial maps from seed labels.
 * - seedLabels: Int16Array | Int32Array, length width*height
 *   values: -1 (none), 0 (source), 1 (sink), or 0..K-1 for multi-class.
 */
function initJFA(width, height, seedLabels) {
  const N = width * height;
  const nearestSeedIndex = new Int32Array(N);
  const labelMap = new Int32Array(N);
  const distanceMap = new Float32Array(N);

  // Initialize with "infinite" distance
  const INF = Number.POSITIVE_INFINITY;

  for (let i = 0; i < N; i++) {
    const lbl = seedLabels[i];
    if (lbl >= 0) {
      // This pixel is a seed: nearest is itself
      nearestSeedIndex[i] = i;
      labelMap[i] = lbl;
      distanceMap[i] = 0.0;
    } else {
      nearestSeedIndex[i] = -1;
      labelMap[i] = -1;
      distanceMap[i] = INF;
    }
  }
  
  return { nearestSeedIndex, labelMap, distanceMap };
}

/**
 * Utility: convert index to (x, y).
 */
function idxToXY(i, width) {
  return { x: i % width, y: (i / width) | 0 };
}

/**
 * Utility: clamp coordinate to grid bounds.
 */
function clamp(v, min, max) {
  if (v < min) return min;
  if (v > max) return max;
  return v;
}

/**
 * Compute squared Euclidean distance between pixel (x,y) and seed index sIdx.
 * nearestSeedIndex holds sIdx; if sIdx < 0, return +INF.
 */
function distToSeed(i, sIdx, width) {
  if (sIdx < 0) return Number.POSITIVE_INFINITY;
  const { x, y } = idxToXY(i, width);
  const { x: sx, y: sy } = idxToXY(sIdx, width);
  const dx = Math.abs(x - sx);
  const dy = Math.abs(y - sy);
  return dx + dy;
}

/**
 * Jump directions (8-neighborhood). Expand to 16 for more accuracy if desired.
 */
const directions8 = [
  { dx: 1, dy: 0 }, // E
  { dx: -1, dy: 0 }, // W
  { dx: 0, dy: 1 }, // S
  { dx: 0, dy: -1 }, // N
  { dx: 1, dy: 1 }, // SE
  { dx: 1, dy: -1 }, // NE
  { dx: -1, dy: 1 }, // SW
  { dx: -1, dy: -1 }, // NW
];

/**
 * Tie-break rule: decide if candidate replaces current when distances are equal.
 * Customize to prefer source over sink, etc.
 * Here: prefer lower label id; then prefer lower seed index as deterministic fallback.
 */
function tieBreak(currentLabel, currentSeedIdx, candidateLabel, candidateSeedIdx) {
  if (candidateLabel < currentLabel) return true;
  if (candidateLabel > currentLabel) return false;
  // same label: prefer smaller seed index for determinism
  return candidateSeedIdx < currentSeedIdx;
}

/**
 * Run Jump Flooding to assign nearest seed to every pixel.
 * @param {number} width
 * @param {number} height
 * @param {Int32Array|Int16Array} seedLabels  length width*height
 * @returns {{ nearestSeedIndex, labelMap, distanceMap }}
 */
export function runJumpFlooding(width, height, seedLabels) {
  const directions = directions8;

  const N = width * height;
  const INF = Number.POSITIVE_INFINITY;

  // Initialize state
  let { nearestSeedIndex, labelMap, distanceMap } = initJFA(width, height, seedLabels);

  // Determine initial jump length: next power of two >= max(width, height)
  const maxDim = Math.max(width, height);
  let jump = 1;
  while (jump < maxDim) jump <<= 1;

  const stats = { iterations: [] };

  // JFA main loop: jump -> jump/2 -> ... -> 1
  while (jump >= 1) {
    // For each pixel, probe neighbors at current jump length
    for (let i = 0; i < N; i++) {
      const { x, y } = idxToXY(i, width);

      // Current best
      let bestSeedIdx = nearestSeedIndex[i];
      let bestLabel = labelMap[i];
      let bestDist = distanceMap[i];

      // Probe directions
      for (let k = 0; k < directions.length; k++) {
        const dx = directions[k].dx * jump;
        const dy = directions[k].dy * jump;
        const nx = clamp(x + dx, 0, width - 1);
        const ny = clamp(y + dy, 0, height - 1);
        const nIdx = ny * width + nx;

        // Candidate comes from neighbor's current nearest seed
        const candSeedIdx = nearestSeedIndex[nIdx];
        if (candSeedIdx < 0) continue; // neighbor has no seed info

        const candLabel = seedLabels[candSeedIdx]; // label of that seed
        const candDist = distToSeed(i, candSeedIdx, width);

        // Update if strictly better, or tie-break
        if (candDist < bestDist) {
          bestDist = candDist;
          bestSeedIdx = candSeedIdx;
          bestLabel = candLabel;
        } else if (candDist === bestDist && candSeedIdx !== bestSeedIdx) {
          if (tieBreak(bestLabel, bestSeedIdx, candLabel, candSeedIdx)) {
            bestDist = candDist;
            bestSeedIdx = candSeedIdx;
            bestLabel = candLabel;
          }
        }
      }

      // Update Arrays
      nearestSeedIndex[i] = bestSeedIdx;
      labelMap[i] = bestLabel;
      distanceMap[i] = bestDist;
    }

    stats.iterations.push({ jump });

    // Halve jump length
    jump >>= 1;
  }

  return { nearestSeedIndex, labelMap, distanceMap, stats };
}
