import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings } from './wgpuUtils.js';

export async function runPushRelabelWebGPU(imageWidth, imageHeight, intensityData, labelMapData, distanceMapData, options = {}) {
  const { device } = await initWebGPU();

  const strength = options.strength || 0.8;
  const sigma = options.sigma || 0.1;

  // --- WGSL Load ---
  const initCode = await loadWGSL('./shaders/pr_init.wgsl');
  const stepCode = await loadWGSL('./shaders/pr_step.wgsl');
  const bfsRelabelInitCode = await loadWGSL('./shaders/bfs_relabel_init.wgsl');
  const bfsRelabelStepCode = await loadWGSL('./shaders/bfs_relabel_step.wgsl');

  // --- Pipelines ---
  const pipelines = {
    init: createPipeline(device, initCode, 'main'),
    step: createPipeline(device, stepCode, 'main'),
    bfsRelabelInit: createPipeline(device, bfsRelabelInitCode, 'main'),
    bfsRelabelStep: createPipeline(device, bfsRelabelStepCode, 'main'),
  };

  const numPixels = imageWidth * imageHeight;
  const workgroups = [Math.ceil(imageWidth / 16), Math.ceil(imageHeight / 16), 1];

  // --- Buffers ---
  // Params: [width, height, sigma, parity, strength, 0, 0, 0]
  const paramsArray = new Float32Array([imageWidth, imageHeight, sigma, 0.0, strength, 0.0, 0.0, 0.0]);
  const paramsBuffer = createBuffer(device, paramsArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const intensityBuffer = createBuffer(device, new Float32Array(intensityData), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const labelBuffer = createBuffer(device, labelMapData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const hBuffer = createBuffer(device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // Edges: Float32 for Cap and Flow
  const capBuffer = createBuffer(device, new Float32Array(numPixels * 2), GPUBufferUsage.STORAGE);
  const flowBuffer = createBuffer(device, new Float32Array(numPixels * 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // BFS Dist/Mask Buffers (Ping-Pong capable)
  const distBufferA = createBuffer(device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const distBufferB = createBuffer(device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // --- BindGroups ---
  // 1. Init
  const bgInit = createBindGroupWithBindings(device, pipelines.init, {
    0: paramsBuffer, 1: intensityBuffer, 2: labelBuffer, 3: hBuffer, 4: capBuffer, 5: flowBuffer,
    6: createBuffer(device, distanceMapData, GPUBufferUsage.STORAGE) // JFA Dist
  });

  // 2. Step (Push-Relabel)
  const bgStep = createBindGroupWithBindings(device, pipelines.step, {
    0: paramsBuffer, 1: labelBuffer, 2: hBuffer, 3: capBuffer, 4: flowBuffer
  });

  // 3. BFS Relabel (Global Relabeling: Sink -> Source)
  // Re-uses hBuffer as output distance
  const bgBfsRelabelInit = createBindGroupWithBindings(device, pipelines.bfsRelabelInit, {
    0: paramsBuffer, 1: labelBuffer, 2: distBufferA // A initialized
  });
  // Step A->B: Write B, then we will copy B to h later? 
  // Optimization: Write directly to B, then B->A. Finally Copy A to h.
  const bgBfsRelabelStepAB = createBindGroupWithBindings(device, pipelines.bfsRelabelStep, {
    0: paramsBuffer, 1: capBuffer, 2: flowBuffer, 3: distBufferA, 4: distBufferB
  });
  const bgBfsRelabelStepBA = createBindGroupWithBindings(device, pipelines.bfsRelabelStep, {
    0: paramsBuffer, 1: capBuffer, 2: flowBuffer, 3: distBufferB, 4: distBufferA
  });

  // --- Execution Logic ---
  console.log("Initializing...");
  runKernel(device, [pipelines.init], [bgInit], [workgroups]);

  const MAX_ITER = 3000;
  const BFS_FREQ = 1000; // 1000回ごとにGlobal Relabeling
  const BFS_DIAMETER = Math.max(imageWidth, imageHeight);

  console.log(`Starting Loop (Strength=${strength})...`);

  for (let i = 0; i < MAX_ITER; i++) {

    // --- Periodic BFS (Global Relabeling) ---
    // 高さが不正確になると収束が遅くなるため、定期的に正確な距離にリセットする
    if (i > 0 && i % BFS_FREQ === 0) {
      // 1. Init: Sink=0, Others=INF
      runKernel(device, [pipelines.bfsRelabelInit], [bgBfsRelabelInit], [workgroups]);

      // 2. Propagate (Sink <- Source)
      for (let k = 0; k < BFS_DIAMETER; k += 2) {
        runKernel(device, [pipelines.bfsRelabelStep], [bgBfsRelabelStepAB], [workgroups]);
        runKernel(device, [pipelines.bfsRelabelStep], [bgBfsRelabelStepBA], [workgroups]);
      }

      // 3. Update H: Copy distBufferA (result) to hBuffer
      // 専用カーネルを作るのがベストですが、ここではcopyBufferToBufferを使います
      // ※hBuffer(u32)とdistBufferA(u32)はサイズ同じ
      const byteSize = numPixels * 4;
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(distBufferA, 0, hBuffer, 0, byteSize);
      device.queue.submit([commandEncoder.finish()]);
    }

    // --- Push-Relabel Step (Checkerboard) ---
    const parity = i % 2;
    device.queue.writeBuffer(paramsBuffer, 12, new Float32Array([parity]));
    runKernel(device, [pipelines.step], [bgStep], [workgroups]);
  }

  // --- Readback ---
  const finalHeights = await readBuffer(device, hBuffer, Uint32Array, numPixels);

  // --- Process Results (CPU側で判定) ---
  const segmentation = new Uint8Array(numPixels);
  const normalizedHeights = new Float32Array(numPixels);
  
  // 閾値: 画像のサイズ(パスの最大長)の2倍以上なら、シンクに到達できない(ソース側)とみなす
  const threshold = 2 * (imageWidth + imageHeight);

  for (let i = 0; i < numPixels; i++) {
    const h = finalHeights[i];

    // 1. Segmentation Check
    // ソースノード(Label 2) または 高さが閾値以上の場所を前景(1)とする
    // ※Label 2は初期化でhが高い値になっているはずですが、念のためOR条件推奨
    if (labelMapData[i] === 2 || h >= threshold) {
      segmentation[i] = 1;
    } else {
      segmentation[i] = 0;
    }

    // 2. Normalize Heights
    // 0.0 ~ 1.0 (以上) の範囲に正規化して出力
    normalizedHeights[i] = h / threshold;
  }

  // 戻り値から excess を削除し、heightsは正規化したものを返す
  return {
    segmentation: segmentation,
    heights: normalizedHeights
  };
}