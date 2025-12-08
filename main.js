import { addEdge, buildGridGraph, createGraphBuffers, buildCSR, createParamsBuffer } from './graphBuilder.js';
import { loadWGSL, createPipeline, createBindGroupWithBindings, runKernel, readBuffer } from './wgpuUtils.js';

/**
 * 指定回数だけ UpdateDual -> UpdatePrimal を繰り返す
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipelineDual
 * @param {GPUComputePipeline} pipelinePrimal
 * @param {GPUBindGroup} bindGroupDual
 * @param {GPUBindGroup} bindGroupPrimal
 * @param {number} N - ノード数
 * @param {number} M - エッジ数
 * @param {number} iterations - 繰り返し回数
 */
export async function runIterations(device, pipelineDual, pipelinePrimal, bindGroupDual, bindGroupPrimal, N, M, iterations) {
  for (let it = 0; it < iterations; it++) {
    // UpdateDual（エッジごと）: M が 0 のときはスキップ
    if (M > 0) {
      runKernel(device, pipelineDual, bindGroupDual, Math.ceil(M / 256));
      // GPU の処理完了を待つ
      await device.queue.onSubmittedWorkDone();
    }

    // UpdatePrimal（ノードごと）
    runKernel(device, pipelinePrimal, bindGroupPrimal, Math.ceil(N / 256));
    await device.queue.onSubmittedWorkDone();
  }
}


export async function initGraphCut() {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // --- グラフ準備（例: N=16, no edges） ---
  const { N, edges } = buildGridGraph(10, 10);
  const M = edges.length;
  const initData = {
    d_u: new Float32Array([...Array(N).keys()]),
    d_u_bar: new Float32Array([...Array(N).keys()])
  };
  const buffers = createGraphBuffers(device, N, M, initData);

  // --- params uniform buffer を生成 ---
  const paramsValues = {
    tau: 0.1,
    sigma: 0.1,
    theta: 0.5,
    N: N,
    M: M,
    source: 0, // 例: source node index
    sink: N - 1 // 例: sink node index
  };
  const paramsBuffer = createParamsBuffer(device, paramsValues);

  // --- WGSL 読み込み & パイプライン作成 ---
  const wgslCode = await loadWGSL('./graphCut.wgsl');

  // ここでは entryPoint を指定してパイプラインを作る（updateDual / updatePrimal）
  const pipelineDual = createPipeline(device, wgslCode, 'updateDual');
  const pipelinePrimal = createPipeline(device, wgslCode, 'updatePrimal');

  // --- bindGroup を作る（binding 番号は WGSL に合わせる） ---
  const bindingMapDual = {
    // updateDual が参照する binding のみ
    1: buffers.d_u_bar,   // WGSL binding(1)
    2: buffers.d_p,       // binding(2)
    3: buffers.d_cap,     // binding(3)
    4: buffers.head,      // binding(4)
    5: buffers.tail,      // binding(5)
    10: paramsBuffer      // binding(10)
  };

  const bindingMapPrimal = {
    // updatePrimal が参照する binding のみ
    0: buffers.d_u,           // binding(0)
    1: buffers.d_u_bar,       // binding(1)
    2: buffers.d_p,           // binding(2)
    6: buffers.row_ptr,       // binding(6)
    // 7: buffers.col_idx,       // binding(7)
    8: buffers.rev_edge,      // binding(8)
    9: buffers.csr_edge_idx,  // binding(9)
    10: paramsBuffer
  };

  const bindGroupDual = createBindGroupWithBindings(device, pipelineDual, bindingMapDual);
  const bindGroupPrimal = createBindGroupWithBindings(device, pipelinePrimal, bindingMapPrimal);


  const iterations = 10;
  await runIterations(device, pipelineDual, pipelinePrimal, bindGroupDual, bindGroupPrimal, N, M, iterations);

  // ループ後に結果を読み出す
  const result = await readBuffer(device, buffers.d_u, Float32Array, N);
  console.log("最終 d_u:", result);
}

export async function testGraphConstruction() {
  const { N, edges } = buildGridGraph(10, 10);
  const M = edges.length;

  const { row_ptr, col_idx, rev_edge } = buildCSR(N, edges);

  console.log("ノード数 N:", N);
  console.log("エッジ数 M:", M);
  console.log("row_ptr:", row_ptr);
  console.log("col_idx:", col_idx);
  console.log("rev_edge:", rev_edge);
}

export async function testWebGPUComputation() {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const N = 16; // 小さめのテスト用ノード数
  const M = 0;  // エッジは不要

  // d_u を初期化 (0,1,2,...)
  const initData = { d_u: new Float32Array([...Array(N).keys()]) };
  const buffers = createGraphBuffers(device, N, M, initData);

  // WGSL カーネル
  const wgslCode = `
    @group(0) @binding(0) var<storage, read_write> d_u : array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
        let i = GlobalInvocationID.x;
        if (i < arrayLength(&d_u)) {
            d_u[i] *= 2.0;
        }
    }
  `;

  const pipeline = createPipeline(device, wgslCode);
  const bindGroup = createBindGroup(device, pipeline, { d_u: buffers.d_u });

  runKernel(device, pipeline, bindGroup, Math.ceil(N / 64));

  // デバッグ: d_u を CPU に戻す
  const result = await readBuffer(device, buffers.d_u, Float32Array, N);
  console.log("計算結果 d_u:", result);
}