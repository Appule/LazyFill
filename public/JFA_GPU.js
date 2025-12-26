import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings } from './wgpuUtils.js';

/**
 * Jump Flooding Algorithmを実行し、最近傍ラベルマップと距離マップを計算する
 * @param {number} width 画像幅
 * @param {number} height 画像高さ
 * @param {ArrayLike} labels ラベルデータ
 * @param {object} bbInfo { minX, minY, width, height }
 */
export async function runJumpFloodingWebGPU(width, height, labels, bbInfo = null) {
  const { device } = await initWebGPU();

  // BB情報の展開
  const { minX, minY, width: bbW, height: bbH } = bbInfo || { minX: 0, minY: 0, width: width, height: height };

  // --- 1. WGSLロード ---
  const initWgsl = await loadWGSL('./shaders/jfa_init.wgsl');
  const stepWgsl = await loadWGSL('./shaders/jfa_step.wgsl');
  const finalWgsl = await loadWGSL('./shaders/jfa_final.wgsl');

  // --- 2. パイプライン作成 ---
  const pipelines = {
    init: createPipeline(device, initWgsl, 'main'),
    step: createPipeline(device, stepWgsl, 'main'),
    final: createPipeline(device, finalWgsl, 'main'),
  };

  const numPixels = width * height;
  // WorkgroupはBBサイズに合わせる
  const workgroups = [Math.ceil(bbW / 16), Math.ceil(bbH / 16), 1];

  // --- 3. バッファ作成 ---
  // Params: [width, height, step_size, padding, minX, minY, 0, 0]
  // 配列サイズを8 (32bytes) に拡張して minX, minY を格納
  const paramsArray = new Float32Array([
    width, height, 0, 0,
    minX, minY, 0, 0
  ]);
  const paramsBuffer = createBuffer(device, paramsArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Input Labels
  const labelInputBuffer = createBuffer(device, new Uint32Array(labels), GPUBufferUsage.STORAGE);

  // JFA Ping-Pong Buffers
  const jfaBufferA = createBuffer(device, new Int32Array(numPixels * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const jfaBufferB = createBuffer(device, new Int32Array(numPixels * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Output Buffers
  const nearestLabelBuffer = createBuffer(device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const distanceBuffer = createBuffer(device, new Int32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // --- 4. BindGroups作成 ---
  const bgInit = createBindGroupWithBindings(device, pipelines.init, {
    0: paramsBuffer, 1: labelInputBuffer, 2: jfaBufferA
  });

  const bgStepAB = createBindGroupWithBindings(device, pipelines.step, {
    0: paramsBuffer, 1: jfaBufferA, 2: jfaBufferB
  });
  const bgStepBA = createBindGroupWithBindings(device, pipelines.step, {
    0: paramsBuffer, 1: jfaBufferB, 2: jfaBufferA
  });

  const bgFinalA = createBindGroupWithBindings(device, pipelines.final, {
    0: paramsBuffer, 1: jfaBufferA, 2: nearestLabelBuffer, 3: distanceBuffer
  });
  const bgFinalB = createBindGroupWithBindings(device, pipelines.final, {
    0: paramsBuffer, 1: jfaBufferB, 2: nearestLabelBuffer, 3: distanceBuffer
  });

  // --- 5. 実行 ---
  console.log("JFA: Initializing...");
  runKernel(device, [pipelines.init], [bgInit], [workgroups]);

  // JFA Loop
  // ステップサイズ計算: BBの最大辺に基づく
  const maxDim = Math.max(bbW, bbH);

  // maxDimに最も近い2の累乗の半分から開始
  let stepSize = 1 << (Math.ceil(Math.log2(maxDim)) - 1);
  if (stepSize >= maxDim) stepSize /= 2;

  let currentBufferIsA = true;

  console.log(`JFA: Starting loop (max_step=${stepSize} for BB=${bbW}x${bbH})...`);

  while (stepSize >= 1) {
    // Uniform更新 (Step Size) - offset 8 bytes (index 2)
    device.queue.writeBuffer(paramsBuffer, 8, new Float32Array([stepSize]));

    if (currentBufferIsA) {
      runKernel(device, [pipelines.step], [bgStepAB], [workgroups]);
    } else {
      runKernel(device, [pipelines.step], [bgStepBA], [workgroups]);
    }

    currentBufferIsA = !currentBufferIsA;
    stepSize /= 2;
  }

  // --- 6. 最終変換 ---
  console.log("JFA: Finalizing...");
  const finalBg = currentBufferIsA ? bgFinalA : bgFinalB;
  runKernel(device, [pipelines.final], [finalBg], [workgroups]);

  // --- 7. 読み取り ---
  const labelResult32 = await readBuffer(device, nearestLabelBuffer, Uint32Array, numPixels);
  const distResult = await readBuffer(device, distanceBuffer, Int32Array, numPixels);

  return {
    nearestLabelMap: new Uint8Array(labelResult32),
    distanceMap: distResult
  };
}