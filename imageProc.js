import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings, } from './wgpuUtils.js';

export async function rgbaToGrayscale(rgbaFlat, wgslPath = 'shaders/grayscale.wgsl') {
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

  const rgbaBuffer = createBuffer(device, rgbaPacked, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outBuffer = createBuffer(device, outPacked, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const paramsData = new Uint32Array([pixelCount]);
  const paramsBuffer = createBuffer(device, paramsData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const wgslCode = await loadWGSL(wgslPath);
  const pipeline = createPipeline(device, wgslCode, 'main');

  const bindGroup = createBindGroupWithBindings(device, pipeline, {
    0: rgbaBuffer,
    1: outBuffer,
    2: paramsBuffer,
  });

  const workgroupSize = 256;
  const workgroups = Math.ceil(pixelCount / workgroupSize);
  runKernel(device, pipeline, bindGroup, [workgroups, 1, 1]);

  // 結果読み出し
  const outU32 = await readBuffer(device, outBuffer, Uint32Array, pixelCount);

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
