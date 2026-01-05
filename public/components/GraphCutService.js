import { extractNormalizedR } from '../flatDataUtils.js';
import { imageProc } from '../wgpuProc.js';
import { runJumpFloodingWebGPU } from '../JFA_GPU.js';
import { runPushRelabelWebGPU } from '../PushRelabel_GPU.js';

export const GraphCutService = {
  async execAutoMark(state, bbThreshold, padding) {
    if (!state.isImageLoaded) return;

    // エッジ検出
    const imageProcResult = await imageProc(state.inputData, state.width, state.height);
    const normalizedR = extractNormalizedR(imageProcResult);

    const { width, height, markerBuffer } = state;
    const numPixels = width * height;
    const targetId = 2;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let hasContent = false;

    // BB計算
    for (let i = 0; i < numPixels; i++) {
      if (normalizedR[i] > bbThreshold) {
        const x = i % width;
        const y = Math.floor(i / width);
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
        hasContent = true;
      }
    }

    if (!hasContent) return;

    const pMinX = Math.max(0, minX - padding);
    const pMaxX = Math.min(width - 1, maxX + padding);
    const pMinY = Math.max(0, minY - padding);
    const pMaxY = Math.min(height - 1, maxY + padding);

    const writeMarker = (idx) => {
      if (markerBuffer[idx] === 0) {
        markerBuffer[idx] = targetId;
        state.updatePixelCount(targetId, 1);
      }
    };

    // 4方向スキャン
    for (let x = pMinX; x <= pMaxX; x++) {
      let hitLine = false;
      for (let y = pMinY; y <= pMaxY; y++) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    for (let x = pMinX; x <= pMaxX; x++) {
      let hitLine = false;
      for (let y = pMaxY; y >= pMinY; y--) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    for (let y = pMinY; y <= pMaxY; y++) {
      let hitLine = false;
      for (let x = pMinX; x <= pMaxX; x++) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    for (let y = pMinY; y <= pMaxY; y++) {
      let hitLine = false;
      for (let x = pMaxX; x >= pMinX; x--) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
  },

  async run(state, params) {
    if (!state.isImageLoaded) return null;

    const imageProcResult = await imageProc(state.inputData, state.width, state.height);
    const normalizedR = extractNormalizedR(imageProcResult);
    const { width, height } = state;
    const numPixels = width * height;

    const objectIds = Object.keys(state.labels)
      .map(Number).filter(id => id >= 2).sort((a, b) => b - a);

    const finalLabelMap = new Uint8Array(numPixels).fill(0);
    if (objectIds.length === 0) return finalLabelMap;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let hasContent = false;
    for (let i = 0; i < numPixels; i++) {
      if (normalizedR[i] > params.bbThreshold) {
        const x = i % width;
        const y = Math.floor(i / width);
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
        hasContent = true;
      }
    }

    if (!hasContent) return finalLabelMap;

    const pad = params.padding;
    const pMinX = Math.max(0, minX - pad);
    const pMaxX = Math.min(width - 1, maxX + pad);
    const pMinY = Math.max(0, minY - pad);
    const pMaxY = Math.min(height - 1, maxY + pad);
    const bbWidth = pMaxX - pMinX + 1;
    const bbHeight = pMaxY - pMinY + 1;
    const bbInfo = { minX: pMinX, minY: pMinY, width: bbWidth, height: bbHeight };

    for (const targetId of objectIds) {
      if ((state.labelPixelCounts[targetId] || 0) <= 0) continue;

      const tempMarker = new Int32Array(numPixels);

      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        for (let x = pMinX; x <= pMaxX; x++) {
          const idx = rowOffset + x;
          const uid = state.markerBuffer[idx];
          if (uid === targetId) tempMarker[idx] = 2;
          else if (uid !== 0) tempMarker[idx] = 1;
          else tempMarker[idx] = 0;
        }
      }

      // Sink on BB edges
      for (let x = pMinX; x <= pMaxX; x++) {
        const topIdx = pMinY * width + x;
        const botIdx = pMaxY * width + x;
        if (tempMarker[topIdx] === 0) tempMarker[topIdx] = 1;
        if (tempMarker[botIdx] === 0) tempMarker[botIdx] = 1;
      }
      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        const leftIdx = rowOffset + pMinX;
        const rightIdx = rowOffset + pMaxX;
        if (tempMarker[leftIdx] === 0) tempMarker[leftIdx] = 1;
        if (tempMarker[rightIdx] === 0) tempMarker[rightIdx] = 1;
      }

      const { distanceMap } = await runJumpFloodingWebGPU(width, height, tempMarker, bbInfo);
      const bfsFreq = params.bfsNum > 0 ? Math.floor(params.maxIter / params.bfsNum) : params.maxIter;

      const prResult = await runPushRelabelWebGPU(
        width, height, normalizedR, tempMarker, distanceMap,
        { strength: params.strength, sigma: params.sigma, maxIter: params.maxIter, bfsFreq },
        bbInfo
      );

      const seg = prResult.segmentation;
      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        for (let x = pMinX; x <= pMaxX; x++) {
          const i = rowOffset + x;
          if (seg[i] === 1) finalLabelMap[i] = targetId;
        }
      }
    }

    return finalLabelMap;
  }
};