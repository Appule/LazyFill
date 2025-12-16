// Minimal scalar image graph cut (push-relabel) for 4-neighborhood

export function graphCutScalar({
  width,
  height,
  image,        // Float64Array length=width*height, intensities normalized [0,1]
  nearestSeedIndex, labelMap, distanceMap, // from JFA
  sigma = 0.1,  // for Cs(p,q)
  sourceSeeds,  // Array of {x,y}
  sinkSeeds,    // Array of {x,y}
  bias = 0.01   // small terminal bias for non-seeds
}) {
  const N = width * height;
  const idx = (x, y) => y * width + x;

  // Residual capacities
  const capRight = new Float64Array(N); // (x,y)->(x+1,y)
  const capLeft = new Float64Array(N); // (x+1,y)->(x,y)
  const capDown = new Float64Array(N); // (x,y)->(x,y+1)
  const capUp = new Float64Array(N); // (x,y+1)->(x,y)
  const capSrc = new Float64Array(N); // source->node
  const capSnk = new Float64Array(N); // node->sink

  // Build neighbor capacities
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = idx(x, y);
      if (x + 1 < width) {
        const j = idx(x + 1, y);
        const c = Math.exp(-Math.pow(image[i] - image[j], 2) / (2 * sigma * sigma));
        capRight[i] = c;
        capLeft[j] = c; // symmetric
      }
      if (y + 1 < height) {
        const j = idx(x, y + 1);
        const c = Math.exp(-Math.pow(image[i] - image[j], 2) / (2 * sigma * sigma));
        capDown[i] = c;
        capUp[j] = c; // symmetric
      }
    }
  }

  // Terminal capacities: hard seeds as "infinite", others small bias
  const INF = 1e9;
  const seedSrc = new Uint8Array(N);
  const seedSnk = new Uint8Array(N);
  sourceSeeds.forEach(({ x, y }) => { seedSrc[idx(x, y)] = 1; });
  sinkSeeds.forEach(({ x, y }) => { seedSnk[idx(x, y)] = 1; });

  for (let i = 0; i < N; i++) {
    if (seedSrc[i]) capSrc[i] = INF;
    else if (seedSnk[i]) capSrc[i] = 0;
    else capSrc[i] = bias;

    if (seedSnk[i]) capSnk[i] = INF;
    else if (seedSrc[i]) capSnk[i] = 0;
    else capSnk[i] = bias;
  }

  // Push-Relabel
  const h = new Int32Array(N);
  const excess = new Float64Array(N);
  const active = new Uint8Array(N);

  // Initialize: saturate source edges
  let sourceCount = 0;
  for (let i = 0; i < N; i++) {
    if (capSrc[i] > 0) {
      const f = capSrc[i]; // send flow from source to i
      capSrc[i] -= f;      // residual source->i reduced
      excess[i] += f;      // node gains excess
      sourceCount++;
    }
  }

  initHeights(h, distanceMap, nearestSeedIndex, labelMap);

  // Active queue: all nodes with positive excess except sinks/seeds
  const queue = [];
  for (let i = 0; i < N; i++) {
    if (excess[i] > 0 && !seedSnk[i]) { active[i] = 1; queue.push(i); }
    if (seedSrc[i]) { discharge(i); }
  }

  function initHeights(h, distanceMap, nearestSeedIndex, labelMap) {
    const N = distanceMap.length;

    for (let i = 0; i < N; i++) {
      const seedIdx = nearestSeedIndex[i];
      const label = labelMap[seedIdx]; // 0=background, 1=foreground

      if (label === 1) {
        // 最近傍 seed が背景なら、その距離を高さに
        h[i] = distanceMap[i];
      } else {
        // 前景 seed に属するなら高い値を与える
        h[i] = N - distanceMap[i];
      }
    }
  }

  function relabel(i) {
    // min neighbor height + 1 among residual-capacity-positive arcs
    let minH = Infinity;
    // Neighbors
    // right
    if (capRight[i] > 0) {
      const x = i % width, y = (i / width) | 0;
      if (x + 1 < width) minH = Math.min(minH, h[i + 1]);
    }
    // left
    if (capLeft[i] > 0) {
      const x = i % width;
      if (x - 1 >= 0) minH = Math.min(minH, h[i - 1]);
    }
    // down
    if (capDown[i] > 0) {
      const y = (i / width) | 0;
      if (y + 1 < height) minH = Math.min(minH, h[i + width]);
    }
    // up
    if (capUp[i] > 0) {
      const y = (i / width) | 0;
      if (y - 1 >= 0) minH = Math.min(minH, h[i - width]);
    }
    // to sink (node->sink residual)
    if (capSnk[i] > 0) {
      // sink height = 0
      minH = Math.min(minH, 0);
    }
    if (minH < Infinity) h[i] = minH + 1;
  }

  function pushDir(from, to, capArrFromTo) {
    if (capArrFromTo[from] <= 0 || excess[from] <= 0) return false;
    if (h[from] !== h[to] + 1) return false; // admissible?
    const send = Math.min(capArrFromTo[from], excess[from]);
    capArrFromTo[from] -= send; // reduce residual forward
    // increase residual backward in matching array
    // We need to find the matching reverse capacity array
    // Map forward array to reverse:
    // right -> left at 'to', left -> right at 'to', down -> up at 'to', up -> down at 'to'
    if (to === from + 1) capLeft[to] += send;
    else if (to === from - 1) capRight[to] += send;
    else if (to === from + width) capUp[to] += send;
    else if (to === from - width) capDown[to] += send;

    excess[from] -= send;
    excess[to] += send;
    return true;
  }

  function discharge(i) {
    // Try push in 4 dirs and to sink; if no push possible, relabel
    let pushed = false;
    const x = i % width;
    const y = (i / width) | 0;
    // push to sink if admissible (sink height=0)
    if (capSnk[i] > 0 && excess[i] > 0 && h[i] === 1) {
      const send = Math.min(capSnk[i], excess[i]);
      capSnk[i] -= send;
      excess[i] -= send;
      pushed = true;
    }
    // right
    if (x + 1 < width && capRight[i] > 0) {
      pushed = pushDir(i, i + 1, capRight) || pushed;
    }
    // left
    if (x - 1 >= 0 && capLeft[i] > 0) {
      pushed = pushDir(i, i - 1, capLeft) || pushed;
    }
    // down
    if (y + 1 < height && capDown[i] > 0) {
      pushed = pushDir(i, i + width, capDown) || pushed;
    }
    // up
    if (y - 1 >= 0 && capUp[i] > 0) {
      pushed = pushDir(i, i - width, capUp) || pushed;
    }
    if (!pushed) relabel(i);
    return pushed;
  }

  while (queue.length > 0) {
    const i = queue.shift();
    active[i] = 0;
    if (seedSrc[i] || seedSnk[i]) continue; // keep seeds fixed
    while (excess[i] > 1e-12) {
      const pushed = discharge(i);
      if (!pushed) break; // after relabel, loop will retry; break helps avoid infinite loops
    }
    if (excess[i] > 1e-12) { // still active
      if (!active[i]) { active[i] = 1; queue.push(i); }
    }
    // neighbors that may have become active
    const x = i % width, y = (i / width) | 0;
    const neigh = [];
    if (x + 1 < width) neigh.push(i + 1);
    if (x - 1 >= 0) neigh.push(i - 1);
    if (y + 1 < height) neigh.push(i + width);
    if (y - 1 >= 0) neigh.push(i - width);
    for (const j of neigh) {
      if (excess[j] > 1e-12 && !active[j]) { active[j] = 1; queue.push(j); }
    }
  }

  // BFS in residual graph from source to mark foreground
  const visited = new Uint8Array(N);
  const q = [];
  // Initialize BFS from source seeds
  for (let i = 0; i < N; i++) if (seedSrc[i]) { visited[i] = 1; q.push(i); }
  while (q.length) {
    const u = q.shift();
    const x = u % width, y = (u / width) | 0;
    // Residual arcs to neighbors
    if (x + 1 < width && capRight[u] > 1e-12 && !visited[u + 1]) { visited[u + 1] = 1; q.push(u + 1); }
    if (x - 1 >= 0 && capLeft[u] > 1e-12 && !visited[u - 1]) { visited[u - 1] = 1; q.push(u - 1); }
    if (y + 1 < height && capDown[u] > 1e-12 && !visited[u + width]) { visited[u + width] = 1; q.push(u + width); }
    if (y - 1 >= 0 && capUp[u] > 1e-12 && !visited[u - width]) { visited[u - width] = 1; q.push(u - width); }
    // Residual from source to u isn't explicit; seeds already enqueued
  }

  // Mask: visited=true is source-side (foreground)
  const mask = new Uint8Array(N);
  for (let i = 0; i < N; i++) mask[i] = visited[i] ? 1 : 0;
  return { mask };
}
