import { getImageDataFromFileInput, drawFlatArrayToCanvas, drawFloatArrayToCanvas } from './fileIO.js';
import { extractNormalizedR, drawCirclesOnCanvas } from './flatDataUtils.js';
import { imageProc } from './wgpuProc.js';
import { runJumpFloodingWebGPU } from './JFA_GPU.js';
import { runPushRelabelWebGPU } from './PushRelabel_GPU.js';


export async function main() {
  const fileInput = document.getElementById('fileInput');
  const canvasInput = document.getElementById('canvasInput');
  const canvasMarker = document.getElementById('canvasMarker');
  const canvasOutput = document.getElementById('canvasOutput');
  const ctxMarker = canvasMarker.getContext('2d', { willReadFrequently: true });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    const img = new Image();
    img.onload = () => {
      canvasInput.width = img.width;
      canvasInput.height = img.height;
      canvasMarker.width = img.width;
      canvasMarker.height = img.height;
      const ctxInput = canvasInput.getContext('2d');
      ctxInput.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(file);
  });

  let drawing = false;
  let currentLabel = 'fg'; // 'fg' = 前景, 'bg' = 背景

  canvasMarker.addEventListener('mousedown', () => drawing = true);
  canvasMarker.addEventListener('mouseup', () => drawing = false);
  canvasMarker.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    ctxMarker.fillStyle = currentLabel === 'fg' ? 'rgba(0, 255, 0, 1)' : 'rgba(255, 0, 0, 1)';
    ctxMarker.beginPath();
    ctxMarker.arc(e.offsetX, e.offsetY, 5, 0, 2 * Math.PI);
    ctxMarker.fill();
  });

  document.getElementById('btnFg').addEventListener('click', () => currentLabel = 'fg');
  document.getElementById('btnBg').addEventListener('click', () => currentLabel = 'bg');
  document.getElementById('btnRun').addEventListener('click', runGrabCut);

  async function runGrabCut() {
    const { data, width, height } = await getImageDataFromFileInput(fileInput); // rgba
    if (!data) return;
    const imageProcResult = await imageProc(data, width, height); // rgba <= rgba

    const normalizedR = extractNormalizedR(imageProcResult); // flat <= rgba
    // drawCirclesOnCanvas(canvasOutput, normalizedR, width, height); // draw <= flat

    // markerキャンバスから前景／背景ヒントを反映
    let markerData = ctxMarker.getImageData(0, 0, canvasMarker.width, canvasMarker.height).data;
    const mask = new Int32Array(width * height).fill(-1);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let idx = y * width + x;
        let r = markerData[idx * 4], g = markerData[idx * 4 + 1];
        if (g > 200) {
          mask[idx] = 2; // 緑 → 前景
        }
        else if (r > 200) {
          mask[idx] = 1; // 赤 → 背景
        }
        else mask[idx] = 0; // その他 → 無視
      }
    }

    const { nearestSeedIndex, labelMap, distanceMap } = await runJumpFloodingWebGPU(width, height, mask);

    const { segmentation, heights } = await runPushRelabelWebGPU(
      width, height, normalizedR,
      labelMap, distanceMap,
      { strength: 0.8, sigma: 0.1 }
    );
    
    drawFloatArrayToCanvas(canvasOutput, segmentation, width, height);
  }
}
