// public/MarkerRenderer.js
import { initWebGPU, createBuffer, loadWGSL, createPipeline, runKernel, readBuffer, createBindGroupWithBindings } from './wgpuUtils.js';

export class MarkerRenderer {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.buffers = {};
  }

  async init() {
    const { device } = await initWebGPU();
    this.device = device;
    const code = await loadWGSL('./shaders/marker_render.wgsl');
    this.pipeline = createPipeline(device, code, 'main');
  }

  /**
   * マーカーバッファから画像データを生成する
   * @param {number} width 
   * @param {number} height 
   * @param {Int32Array} markerBuffer 
   * @param {Object} labels AppState.labels
   * @returns {Uint8ClampedArray} RGBAデータ
   */
  async render(width, height, markerBuffer, labels) {
    if (!this.device) await this.init();

    const numPixels = width * height;

    // 1. カラーパレットの作成 (Structure of Array ではなく Array of Structs)
    // shader: struct Color { r: f32, g: f32, b: f32, a: f32 }
    // 最大IDまで確保
    const maxId = Math.max(...Object.keys(labels).map(Number));
    // Float32Array: 1要素あたり4 floats
    const paletteData = new Float32Array((maxId + 1) * 4);

    for (const [idStr, conf] of Object.entries(labels)) {
      const id = Number(idStr);
      const offset = id * 4;
      // 色を 0.0 - 1.0 に正規化
      paletteData[offset + 0] = conf.r / 255.0;
      paletteData[offset + 1] = conf.g / 255.0;
      paletteData[offset + 2] = conf.b / 255.0;
      paletteData[offset + 3] = conf.a; // 設定値をそのまま使うが、シェーダーで1.0に上書き可
    }

    // 2. バッファ作成
    // Params
    const paramsData = new Float32Array([width, height, 1.0, 0.0]);
    const paramsBuf = createBuffer(this.device, paramsData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Marker Input
    const markerBuf = createBuffer(this.device, markerBuffer, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    // Palette Input
    const paletteBuf = createBuffer(this.device, paletteData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    // Output (u32 packed RGBA)
    // 読み取り用なので COPY_SRC
    const outputBuf = createBuffer(this.device, new Uint32Array(numPixels), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // 3. BindGroup
    const bindGroup = createBindGroupWithBindings(this.device, this.pipeline, {
      0: paramsBuf,
      1: markerBuf,
      2: paletteBuf,
      3: outputBuf
    });

    // 4. 実行
    const workgroups = [Math.ceil(width / 16), Math.ceil(height / 16), 1];
    runKernel(this.device, [this.pipeline], [bindGroup], [workgroups]);

    // 5. 読み取り
    // outputBufは u32配列だが、ImageData作成用に Uint8ClampedArray として読み取る
    // WebGPUのバッファ読み取りヘルパーが ArrayBuffer を返す前提
    const resultBuffer = await readBuffer(this.device, outputBuf, Uint8Array, numPixels * 4);

    // リソース解放 (実際のアプリではバッファを再利用したほうが高速ですが、簡略化のため毎回解放)
    paramsBuf.destroy();
    markerBuf.destroy();
    paletteBuf.destroy();
    outputBuf.destroy();

    return new Uint8ClampedArray(resultBuffer);
  }
}