/**
 * WebGPU を初期化してアダプタとデバイスを返します。
 * @async
 * @function initWebGPU
 * @throws {Error} WebGPU がサポートされていない場合、または初期化に失敗した場合
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice}>}
 */
export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU がサポートされていません");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("GPU アダプタの取得に失敗しました");
  }

  const device = await adapter.requestDevice();
  if (!device) {
    throw new Error("GPU デバイスの取得に失敗しました");
  }

  return { adapter, device };
}

/**
 * GPU バッファを生成する共通関数
 * @param {GPUDevice} device - WebGPU のデバイスオブジェクト
 * @param {TypedArray} data - 初期化するデータ配列 (Float32Array, Uint32Array など)
 * @param {number} [usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST] - バッファの利用用途 (STORAGE, COPY_SRC, COPY_DST など)
 * @returns {GPUBuffer} - 初期化済みの GPUBuffer
 */
export function createBuffer(device, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
  const size = Math.max(4, data.byteLength);
  const buffer = device.createBuffer({
    size,
    usage,
    mappedAtCreation: true,
  });
  const writeArray = new data.constructor(buffer.getMappedRange());
  writeArray.set(data);
  buffer.unmap();
  return buffer;
}

/**
 * パスからwgslコードを取得
 * @param {string} path - WGSL シェーダーコードのパス
 */
export async function loadWGSL(path) {
  const response = await fetch(path);
  return await response.text();
}

/**
 * パイプライン生成関数
 * @param {GPUDevice} device - WebGPU デバイス
 * @param {string} wgslCode - WGSL シェーダコード
 * @param {string} entryPoint - エントリーポイント名
 * @returns {GPUComputePipeline}
 */
export function createPipeline(device, wgslCode, entryPoint = 'main') {
  const shaderModule = device.createShaderModule({ code: wgslCode });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint }
  });
}

/**
 * カーネル実行関数
 * @param {GPUDevice} device - WebGPU デバイス
 * @param {GPUComputePipeline} pipeline - コンピュートパイプライン
 * @param {GPUBindGroup} bindGroup - バインドグループ
 * @param {Array} workgroups - dispatchWorkgroups 数 [x, y, z]
 */
export function runKernel(device, pipeline, bindGroup, workgroups) {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroups);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * デバッグ用: GPUBuffer の内容を CPU に戻す関数
 * @param {GPUDevice} device - WebGPU デバイス
 * @param {GPUBuffer} buffer - 読み出したい GPUBuffer
 * @param {TypedArrayConstructor} ArrayType - Float32Array, Uint32Array など
 * @param {number} length - 要素数
 * @returns {Promise<TypedArray>} - CPU 側にコピーされた配列
 */
export async function readBuffer(device, buffer, ArrayType, length) {
  const readBuffer = device.createBuffer({
    size: length * ArrayType.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, readBuffer.size);
  device.queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const array = new ArrayType(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  return array;
}

/**
 * バインドグループ生成関数（柔軟版）
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {Object} bindingMap - { [bindingNumber]: GPUBuffer } の形で渡す
 * @returns {GPUBindGroup}
 */
export function createBindGroupWithBindings(device, pipeline, bindingMap) {
  const entries = Object.keys(bindingMap).map(k => {
    const binding = Number(k);
    const buffer = bindingMap[k];
    return {
      binding,
      resource: { buffer }
    };
  });

  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
}
