// webgpuJFA.js
import {
  initWebGPU,
  createBuffer,
  loadWGSL,
  createPipeline,
  runKernel,
  readBuffer,
  createBindGroupWithBindings,
} from './wgpuUtils.js';

/**
 * Run Jump Flooding on GPU to compute nearestSeedIndex, labelMap, distanceMap.
 * @param {number} width
 * @param {number} height
 * @param {Int32Array} mask - values: 2(fg), 1(bg), -1(undefined)
 * @returns {Promise<{ nearestSeedIndex:Int32Array, labelMap:Int32Array, distanceMap:Float32Array }>}
 */
export async function runJumpFloodingWebGPU(width, height, mask) {
  const { adapter, device } = await initWebGPU();
  const N = width * height;

  // Directions (8-neighborhood). Packed as vec2<i32>[] in WGSL storage
  const directions = [
    { dx: 1, dy: 0 },   // E
    { dx: -1, dy: 0 },  // W
    { dx: 0, dy: 1 },   // S
    { dx: 0, dy: -1 },  // N
    { dx: 1, dy: 1 },   // SE
    { dx: 1, dy: -1 },  // NE
    { dx: -1, dy: 1 },  // SW
    { dx: -1, dy: -1 }, // NW
  ];
  const dirArray = new Int32Array(directions.flatMap(d => [d.dx, d.dy]));

  // WGSL shaders
  const initWGSL = await loadWGSL('./shaders/jfa_init.wgsl');
  const stepWGSL = await loadWGSL('./shaders/jfa_step.wgsl');

  const initPipeline = createPipeline(device, initWGSL, 'main');
  const stepPipeline = createPipeline(device, stepWGSL, 'main');

  // Buffers
  const maskBuf = createBuffer(device, mask, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const dirBuf = createBuffer(device, dirArray, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const uniformInit = createBuffer(device, new Int32Array([width, height, N]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Double buffers for nearest/label/distance (ping-pong per iteration)
  const nearestA = createBuffer(device, new Int32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const nearestB = createBuffer(device, new Int32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  const labelA = createBuffer(device, new Int32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const labelB = createBuffer(device, new Int32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  const distA = createBuffer(device, new Float32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const distB = createBuffer(device, new Float32Array(N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // For step kernel uniforms: width,height,N, jump, directionsLen
  const stepUniform = createBuffer(device, new Int32Array([width, height, N, 1, directions.length]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Init pass: fill nearest/label/dist from mask (seeds initialize themselves; others inf)
  const initBind = createBindGroupWithBindings(device, initPipeline, {
    0: uniformInit,
    1: maskBuf,
    2: nearestA,
    3: labelA,
    4: distA,
  });

  const workgroupSize = 256; // threads per group
  const dispatch = Math.ceil(N / workgroupSize);
  runKernel(device, [initPipeline], [initBind], [[dispatch, 1, 1]]);

  // Compute initial jump length: next power of two >= max(width,height)
  const maxDim = Math.max(width, height);
  let jump = 1;
  while (jump < maxDim) jump <<= 1;

  // Ping-pong state
  let curNearest = nearestA, curLabel = labelA, curDist = distA;
  let nextNearest = nearestB, nextLabel = labelB, nextDist = distB;

  while (jump >= 1) {
    // Update uniform: width,height,N,jump,dirLen
    const u = new Int32Array([width, height, N, jump, directions.length]);
    device.queue.writeBuffer(stepUniform, 0, u.buffer, u.byteOffset, u.byteLength);

    // Bind group for this iteration (read from cur*, write to next*)
    const stepBind = createBindGroupWithBindings(device, stepPipeline, {
      0: stepUniform,
      1: curNearest,
      2: curLabel,
      3: curDist,
      4: nextNearest,
      5: nextLabel,
      6: dirBuf,
      7: nextDist,
      // directions at binding 6 via dirBind handled by separate bind group below
    });

    // Dispatch: we need two bind groups; some runtimes require single bind group set per slot.
    // Use runKernel with both bind groups in order (pipeline is same).
    runKernel(device, [stepPipeline], [stepBind], [[dispatch, 1, 1]]);

    // Swap ping-pong buffers
    [curNearest, nextNearest] = [nextNearest, curNearest];
    [curLabel, nextLabel] = [nextLabel, curLabel];
    [curDist, nextDist] = [nextDist, curDist];

    // Halve jump
    jump >>= 1;
  }

  // Read back results
  const nearestSeedIndex = await readBuffer(device, curNearest, Int32Array, N);
  const labelMap = await readBuffer(device, curLabel, Int32Array, N);
  const distanceMap = await readBuffer(device, curDist, Float32Array, N);

  return { nearestSeedIndex, labelMap, distanceMap };
}
