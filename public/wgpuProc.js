import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings, } from './wgpuUtils.js';

export async function imageProc(rgbaFlat, width, height) {
  const { adapter, device } = await initWebGPU();

  if (rgbaFlat.length % 4 !== 0) {
    throw new Error('RGBA 配列の長さは4の倍数である必要があります');
  }
  const pixelCount = rgbaFlat.length / 4;

  // 入力を u32 配列にパック
  const rgbaPacked = new Uint32Array(pixelCount);
  for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
    rgbaPacked[i] =
      (rgbaFlat[p] << 0) |
      (rgbaFlat[p + 1] << 8) |
      (rgbaFlat[p + 2] << 16) |
      (rgbaFlat[p + 3] << 24);
  }

  // 出力用バッファ (RGBA packed u32)
  const outPacked = new Uint32Array(pixelCount);

  const pingBuffer = createBuffer(device, rgbaPacked, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const pongBuffer = createBuffer(device, outPacked, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, width, true);      // width
  view.setUint32(4, height, true);     // height
  view.setUint32(8, pixelCount, true); // length
  view.setFloat32(12, 1.0, true);      // LoG scale
  const paramsBuffer = createBuffer(device, paramsData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const wgslHeader = await loadWGSL('shaders/imageProc/header.wgsl');
  const wgslCodeGray = await loadWGSL('shaders/imageProc/rgbaToGray_f32.wgsl');
  const wgslCodeGau = await loadWGSL('shaders/imageProc/gaussian_f32.wgsl');
  const wgslCodeLap = await loadWGSL('shaders/imageProc/laplacian_f32.wgsl');
  const wgslCodeInt = await loadWGSL('shaders/imageProc/intensity_f32.wgsl');
  const wgslCodePack = await loadWGSL('shaders/imageProc/f32ToPacked_u32.wgsl');
  const pipelines = [
    createPipeline(device, wgslHeader + wgslCodeGray, 'main'),
    createPipeline(device, wgslHeader + wgslCodeGau, 'main'),
    createPipeline(device, wgslHeader + wgslCodeLap, 'main'),
    createPipeline(device, wgslHeader + wgslCodeInt, 'main'),
    createPipeline(device, wgslHeader + wgslCodePack, 'main'),
  ];

  const bindGroups = [
    createBindGroupWithBindings(device, pipelines[0], {
      0: paramsBuffer, 1: pingBuffer, 2: pongBuffer,
    }),
    createBindGroupWithBindings(device, pipelines[1], {
      0: paramsBuffer, 1: pongBuffer, 2: pingBuffer,
    }),
    createBindGroupWithBindings(device, pipelines[2], {
      0: paramsBuffer, 1: pingBuffer, 2: pongBuffer,
    }),
    createBindGroupWithBindings(device, pipelines[3], {
      0: paramsBuffer, 1: pongBuffer, 2: pingBuffer,
    }),
    createBindGroupWithBindings(device, pipelines[4], {
      0: paramsBuffer, 1: pingBuffer, 2: pongBuffer,
    }),
  ];

  const workgroups = [
    [Math.ceil(width / 16), Math.ceil(height / 16), 1],
    [Math.ceil(width / 16), Math.ceil(height / 16), 1],
    [Math.ceil(width / 16), Math.ceil(height / 16), 1],
    [Math.ceil(width / 16), Math.ceil(height / 16), 1],
    [Math.ceil(width / 16), Math.ceil(height / 16), 1],
  ];
  await runKernel(device, pipelines, bindGroups, workgroups);

  // 結果読み出し
  const outU32 = await readBuffer(device, pongBuffer, Uint32Array, pixelCount);

  // RGBA形式に展開
  const outU8 = new Uint8Array(pixelCount * 4);
  for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
    const px = outU32[i];
    outU8[p] = (px >> 0) & 0xFF;
    outU8[p + 1] = (px >> 8) & 0xFF;
    outU8[p + 2] = (px >> 16) & 0xFF;
    outU8[p + 3] = (px >> 24) & 0xFF;
  }

  return outU8;
}
