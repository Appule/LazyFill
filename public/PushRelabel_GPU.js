import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings } from './wgpuUtils.js';

export async function runPushRelabelWebGPU(imageWidth, imageHeight, intensityData, labelMapData, distanceMapData, options = {}, bbInfo = null) {
  const { device } = await initWebGPU();
  const maxIter = options.maxIter || 5000;
  const bfsFreq = options.bfsFreq || 500;
  const strength = options.strength || 0.95;
  const sigma = options.sigma || 0.3;

  // BB情報の展開 (指定がない場合は画像全体)
  const { minX, minY, width: bbW, height: bbH } = bbInfo || { minX: 0, minY: 0, width: imageWidth, height: imageHeight };

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
  // 【重要】WorkgroupはBBのサイズに合わせて計算する (画像全体ではなく計算領域のみ起動)
  const workgroups = [Math.ceil(bbW / 16), Math.ceil(bbH / 16), 1];

  // --- Buffers ---
  // Params: [width, height, sigma, parity, strength, minX, minY, 0]
  // WGSL側で struct Params { width: f32, height: f32, sigma: f32, parity: f32, strength: f32, minX: f32, minY: f32 } と定義されている想定
  const paramsArray = new Float32Array([
    imageWidth, imageHeight, sigma, 0.0,
    strength, minX, minY, 0.0
  ]);
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
  const bgBfsRelabelInit = createBindGroupWithBindings(device, pipelines.bfsRelabelInit, {
    0: paramsBuffer, 1: labelBuffer, 2: distBufferA
  });
  const bgBfsRelabelStepAB = createBindGroupWithBindings(device, pipelines.bfsRelabelStep, {
    0: paramsBuffer, 1: capBuffer, 2: flowBuffer, 3: distBufferA, 4: distBufferB
  });
  const bgBfsRelabelStepBA = createBindGroupWithBindings(device, pipelines.bfsRelabelStep, {
    0: paramsBuffer, 1: capBuffer, 2: flowBuffer, 3: distBufferB, 4: distBufferA
  });

  // --- Execution Logic ---
  console.log("Initializing PR...");
  runKernel(device, [pipelines.init], [bgInit], [workgroups]);

  // BFS_DIAMETER は画像全体ではなく、BBの対角線サイズ程度で十分だが、安全のためBBの最大辺をとる
  const BFS_DIAMETER = Math.max(bbW, bbH);

  for (let i = 0; i < maxIter; i++) {

    // --- Periodic BFS (Global Relabeling) ---
    if (i > 0 && i % bfsFreq === 0) {
      // 1. Init
      runKernel(device, [pipelines.bfsRelabelInit], [bgBfsRelabelInit], [workgroups]);

      // 2. Propagate
      for (let k = 0; k < BFS_DIAMETER; k += 2) {
        runKernel(device, [pipelines.bfsRelabelStep], [bgBfsRelabelStepAB], [workgroups]);
        runKernel(device, [pipelines.bfsRelabelStep], [bgBfsRelabelStepBA], [workgroups]);
      }

      // 3. Update H: Copy distBufferA to hBuffer
      // ここは BB内だけでよいが、copyBufferToBufferは連続領域のみ。
      // 全体をコピーするか、あるいはCompute ShaderでBB内コピーをするのが最適だが、頻度が低いので全体コピーで妥協
      const byteSize = numPixels * 4;
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(distBufferA, 0, hBuffer, 0, byteSize);
      device.queue.submit([commandEncoder.finish()]);
    }

    // --- Push-Relabel Step (Checkerboard) ---
    const parity = i % 2;
    // Update parity uniform (offset 12 bytes = float index 3)
    device.queue.writeBuffer(paramsBuffer, 12, new Float32Array([parity]));
    runKernel(device, [pipelines.step], [bgStep], [workgroups]);
  }

  // --- Readback ---
  const finalHeights = await readBuffer(device, hBuffer, Uint32Array, numPixels);

  // --- Process Results (CPU側で判定) ---
  const segmentation = new Uint8Array(numPixels);
  const normalizedHeights = new Float32Array(numPixels);

  // 閾値: BBのサイズ基準で判定
  const threshold = 2 * (bbW + bbH);

  // BB内のみループして結果を構築（外側は0のまま）
  for (let y = minY; y < minY + bbH; y++) {
    const rowOffset = y * imageWidth;
    for (let x = minX; x < minX + bbW; x++) {
      const i = rowOffset + x;
      const h = finalHeights[i];

      if (h >= threshold) {
        segmentation[i] = 1;
      } else {
        segmentation[i] = 0;
      }
      normalizedHeights[i] = h / threshold;
    }
  }

  return {
    segmentation: segmentation,
    heights: normalizedHeights
  };
}