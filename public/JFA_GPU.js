import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings } from './wgpuUtils.js';

/**
 * Jump Flooding Algorithmを実行し、最近傍ラベルマップと距離マップを計算する
 * @param {number} width 画像幅
 * @param {number} height 画像高さ
 * @param {ArrayLike} labels ラベルデータ (0:Unknown, 1:BG, 2:FG などを想定)
 * @returns {Promise<{nearestLabelMap: Uint8Array, distanceMap: Float32Array}>}
 */
export async function runJumpFloodingWebGPU(width, height, labels) {
  const { device } = await initWebGPU();

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
  const workgroups = [Math.ceil(width / 16), Math.ceil(height / 16), 1];

  // --- 3. バッファ作成 ---
  // Params: [width, height, step_size, padding]
  const paramsArray = new Float32Array([width, height, 0, 0]);
  const paramsBuffer = createBuffer(device, paramsArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Input Labels
  const labelInputBuffer = createBuffer(device, new Uint32Array(labels), GPUBufferUsage.STORAGE);

  // JFA Ping-Pong Buffers
  // 各ピクセル: struct { seed_x: i32, seed_y: i32, label: u32, padding: u32 }
  // Int32Array (4 elements per pixel)
  const jfaBufferA = createBuffer(device, new Int32Array(numPixels * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const jfaBufferB = createBuffer(device, new Int32Array(numPixels * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Output Buffers
  const nearestLabelBuffer = createBuffer(device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const distanceBuffer = createBuffer(device, new Int32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // --- 4. BindGroups作成 ---
  // Init
  const bgInit = createBindGroupWithBindings(device, pipelines.init, {
    0: paramsBuffer,
    1: labelInputBuffer,
    2: jfaBufferA // Init結果をAに書き込む
  });

  // Step BindGroups (Ping-Pong)
  // A -> B
  const bgStepAB = createBindGroupWithBindings(device, pipelines.step, {
    0: paramsBuffer,
    1: jfaBufferA, // Read
    2: jfaBufferB  // Write
  });
  // B -> A
  const bgStepBA = createBindGroupWithBindings(device, pipelines.step, {
    0: paramsBuffer,
    1: jfaBufferB, // Read
    2: jfaBufferA  // Write
  });

  // Finalize BindGroups
  // 最終結果がAにある場合
  const bgFinalA = createBindGroupWithBindings(device, pipelines.final, {
    0: paramsBuffer,
    1: jfaBufferA,       // Input Seed Map
    2: nearestLabelBuffer,
    3: distanceBuffer
  });
  // 最終結果がBにある場合
  const bgFinalB = createBindGroupWithBindings(device, pipelines.final, {
    0: paramsBuffer,
    1: jfaBufferB,
    2: nearestLabelBuffer,
    3: distanceBuffer
  });


  // --- 5. 実行 ---
  console.log("JFA: Initializing...");
  runKernel(device, [pipelines.init], [bgInit], [workgroups]);

  // JFA Loop
  // ステップサイズ: N/2, N/4, ..., 1
  const maxDim = Math.max(width, height);
  // 2の累乗に合わせる場合の開始ステップ数計算
  // 例: 500px -> nextPoT=512 -> start=256
  let stepSize = 1 << (Math.ceil(Math.log2(maxDim)) - 1);
  // 画像サイズより大きいステップは不要なので調整
  if (stepSize >= maxDim) stepSize /= 2;

  let currentBufferIsA = true; // 現在の有効なデータが入っているバッファ

  console.log(`JFA: Starting loop (max_step=${stepSize})...`);

  while (stepSize >= 1) {
    // Uniform更新 (Step Size)
    device.queue.writeBuffer(paramsBuffer, 8, new Float32Array([stepSize])); // offset 8 bytes (3rd float)

    if (currentBufferIsA) {
      runKernel(device, [pipelines.step], [bgStepAB], [workgroups]);
    } else {
      runKernel(device, [pipelines.step], [bgStepBA], [workgroups]);
    }

    // Swap and Halve
    currentBufferIsA = !currentBufferIsA;
    stepSize /= 2;
  }

  // --- 6. 最終変換 (Seed Map -> Label & Distance) ---
  console.log("JFA: Finalizing...");
  const finalBg = currentBufferIsA ? bgFinalA : bgFinalB;
  runKernel(device, [pipelines.final], [finalBg], [workgroups]);

  // --- 7. 読み取り ---
  const labelResult32 = await readBuffer(device, nearestLabelBuffer, Uint32Array, numPixels);
  const distResult = await readBuffer(device, distanceBuffer, Int32Array, numPixels);

  const labelResult8 = new Uint8Array(labelResult32);

  return {
    nearestLabelMap: labelResult8,
    distanceMap: distResult
  };
}