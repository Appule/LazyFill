import { getImageDataFromFileInput, convertToGrayscale, downloadBufferAsImage } from './fileIO.js';
import { extractNormalizedR } from './flatDataUtils.js';
import { imageProc } from './wgpuProc.js';
import { runJumpFloodingWebGPU } from './JFA_GPU.js';
import { runPushRelabelWebGPU } from './PushRelabel_GPU.js';

// ==========================================
// 1. STATE
// ==========================================
class AppState {
  constructor() {
    this.width = 0;
    this.height = 0;

    this.inputData = null;       // RGBA
    this.markerBuffer = null;    // Int32Array
    this.latestSegmentation = null; // Uint8Array or similar
    this.labels = {
      0: { r: 0, g: 0, b: 0, a: 0.0, hex: '#000000' },      // Eraser (Alpha 0)
      1: { r: 0, g: 0, b: 255, a: 0.6, hex: '#0000ff' },    // BG
      2: { r: 255, g: 0, b: 0, a: 0.6, hex: '#ff0000' }     // Default Object
    };

    this.currentLabelId = 2;
    this.brushSize = 2;
    this.isImageLoaded = false;
  }

  reset(width, height, inputData) {
    this.width = width;
    this.height = height;
    this.inputData = inputData;
    this.markerBuffer = new Int32Array(width * height).fill(0);
    this.latestSegmentation = null;
    this.isImageLoaded = true;
  }

  addLabel() {
    const ids = Object.keys(this.labels).map(Number);
    const newId = Math.max(...ids) + 1;

    const r = Math.floor(Math.random() * 200);
    const g = Math.floor(Math.random() * 200);
    const b = Math.floor(Math.random() * 200);
    const hex = "#" + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');

    this.labels[newId] = { r, g, b, a: 0.6, hex };
    return newId;
  }

  removeLabel(id) {
    if (id <= 1) return; // Cannot delete BG or Eraser
    delete this.labels[id];

    // Clear markers of this ID
    for (let i = 0; i < this.markerBuffer.length; i++) {
      if (this.markerBuffer[i] === id) {
        this.markerBuffer[i] = 0; // Reset to undefined
      }
    }

    if (this.currentLabelId === id) {
      this.currentLabelId = 1;
    }
  }

  updateLabelColor(id, hex, alpha) {
    if (!this.labels[id]) return;
    if (hex) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);

      const current = this.labels[id];
      this.labels[id] = {
        r, g, b, hex,
        a: (alpha !== undefined) ? parseFloat(alpha) : current.a
      };
    } else {
      const current = this.labels[id];
      this.labels[id] = {
        r: current.r, g: current.g, b: current.b, hex: current.hex,
        a: (alpha !== undefined) ? parseFloat(alpha) : current.a
      };
    }
  }

  getColor(id) {
    return this.labels[id] || this.labels[0];
  }
}

// ==========================================
// 2. VIEW
// ==========================================
class AppView {
  constructor(state, handlers) {
    this.state = state;
    this.handlers = handlers;

    this.transform = {
      scale: 1.0,
      x: 0,
      y: 0
    };
    this.isPanning = false;
    this.lastMousePos = { x: 0, y: 0 };

    this.els = {
      viewport: document.getElementById('viewport'),
      canvasContainer: document.getElementById('canvasContainer'),
      dropMessage: document.getElementById('drop-message'),

      // Top Bar
      btnRun: document.getElementById('btnRun'),
      btnDownloadImg: document.getElementById('btnDownloadImg'),
      btnDownloadMask: document.getElementById('btnDownloadMask'),
      btnToggleParams: document.getElementById('btnToggleParams'),
      panelParams: document.getElementById('panel-params'),

      // Params
      inputs: {
        bb: document.getElementById('inpBB'),
        padding: document.getElementById('inpPadding'),
        sigma: document.getElementById('inpSigma'),
        maxIter: document.getElementById('inpMaxIter'),
        bfsNum: document.getElementById('inpBfsNum'),
        strength: document.getElementById('inpStrength'),
        brush: document.getElementById('inpBrushSize'),
      },

      // Tools & Palette
      chkDynamic: document.getElementById('chkDynamic'),
      dispBrush: document.getElementById('dispBrushSize'),
      chkHideMarker: document.getElementById('chkHideMarker'),
      palette: document.getElementById('paletteContainer'),
      colorPicker: document.getElementById('colorPicker'),
      alphaInput: document.getElementById('alphaInput'),
      currentLabelName: document.getElementById('currentLabelName'),
      btnAddLabel: document.getElementById('btnAddLabel'),
      btnDeleteLabel: document.getElementById('btnDeleteLabel'),
      btnClear: document.getElementById('btnClearMarkers'),

      canvases: {
        input: document.getElementById('canvasInput'),
        marker: document.getElementById('canvasMarker'),
        output: document.getElementById('canvasOutput'),
      },
      ctx: {
        input: document.getElementById('canvasInput').getContext('2d'),
        marker: document.getElementById('canvasMarker').getContext('2d'),
        output: document.getElementById('canvasOutput').getContext('2d'),
      }
    };

    this.bindEvents();
  }

  bindEvents() {
    // --- Drag & Drop (Viewport) ---
    const vp = this.els.viewport;

    ['dragenter', 'dragover'].forEach(eventName => {
      vp.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        vp.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      vp.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        vp.classList.remove('drag-over');
      }, false);
    });

    vp.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        this.handlers.onFileLoad(files[0]);
      }
    });

    // Paste from Clipboard support
    window.addEventListener('paste', (e) => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          const file = item.getAsFile();
          this.handlers.onFileLoad(file);
          break;
        }
      }
    });

    // --- Top Bar Actions ---
    this.els.btnRun.addEventListener('click', () => this.handlers.onRun());
    this.els.btnDownloadImg.addEventListener('click', () => this.handlers.onDownloadImage());
    this.els.btnDownloadMask.addEventListener('click', () => this.handlers.onDownloadMask());

    // Toggle Params Panel
    this.els.btnToggleParams.addEventListener('click', () => {
      this.els.panelParams.classList.toggle('hidden');
    });

    // --- Tools & Palette Actions ---
    this.els.btnAddLabel.addEventListener('click', () => this.handlers.onAddLabel());
    this.els.btnDeleteLabel.addEventListener('click', () => this.handlers.onDeleteLabel());

    // Clear Confirmation
    this.els.btnClear.addEventListener('click', () => {
      if (confirm("Are you sure you want to clear all markers?")) {
        this.handlers.onClearMarkers();
        this.handlers.onRun();
      }
    });

    // --- Color & Alpha Editing ---
    this.els.colorPicker.addEventListener('input', (e) => this.handlers.onColorChange(e.target.value));
    this.els.colorPicker.addEventListener('change', () => this.handlers.onColorEditEnd());
    this.els.alphaInput.addEventListener('input', (e) => this.handlers.onAlphaChange(e.target.value));
    this.els.alphaInput.addEventListener('change', () => this.handlers.onColorEditEnd());

    // Brush Size
    this.els.inputs.brush.addEventListener('input', (e) => {
      const size = parseInt(e.target.value);
      this.state.brushSize = size;
      this.els.dispBrush.textContent = size;
    });

    // Hide Markers
    this.els.chkHideMarker.addEventListener('change', (e) => {
      this.els.canvases.marker.style.opacity = e.target.checked ? '0' : '1'; // 1 because we handle alpha in canvas
      if (!e.target.checked) this.redrawMarkerCanvas();
    });


    // --- Canvas Navigation & Drawing ---

    // Wheel Zoom
    vp.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.handleZoom(e);
    }, { passive: false });

    // Mouse Events
    vp.addEventListener('mousedown', (e) => {
      // Prevent default drag behavior
      e.preventDefault();

      if (e.button === 1) { // Middle
        this.isPanning = true;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        vp.style.cursor = 'grab';
      } else if (e.button === 0) { // Left
        this.drawing = true;
        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, e.shiftKey);
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isPanning) {
        const dx = e.clientX - this.lastMousePos.x;
        const dy = e.clientY - this.lastMousePos.y;
        this.transform.x += dx;
        this.transform.y += dy;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        this.updateTransform();
      } else if (this.drawing) {
        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, e.shiftKey);
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (this.isPanning && e.button === 1) {
        this.isPanning = false;
        vp.style.cursor = 'default';
      } else if (this.drawing && e.button === 0) {
        this.drawing = false;
        this.handlers.onDrawEnd();
      }
    });

    vp.addEventListener('contextmenu', e => e.preventDefault());
  }

  // --- Transform Logic (FullScreen Aware) ---
  getCanvasCoordinates(e) {
    if (!this.state.isImageLoaded) return null;
    const rect = this.els.canvasContainer.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    const actualScale = rect.width / this.state.width;
    const x = Math.floor(relX / actualScale);
    const y = Math.floor(relY / actualScale);
    return { x, y };
  }

  handleZoom(e) {
    if (!this.state.isImageLoaded) return;
    const zoomSensitivity = 0.001;
    const delta = -e.deltaY * zoomSensitivity;
    let newScale = this.transform.scale + this.transform.scale * delta;
    newScale = Math.max(0.1, Math.min(newScale, 20.0));

    const rectViewport = this.els.viewport.getBoundingClientRect();
    const vpMouseX = e.clientX - rectViewport.left;
    const vpMouseY = e.clientY - rectViewport.top;
    const oldX = this.transform.x;
    const oldY = this.transform.y;
    const scaleRatio = newScale / this.transform.scale;

    this.transform.x = vpMouseX - (vpMouseX - oldX) * scaleRatio;
    this.transform.y = vpMouseY - (vpMouseY - oldY) * scaleRatio;
    this.transform.scale = newScale;

    this.updateTransform();
  }

  updateTransform() {
    const { x, y, scale } = this.transform;
    this.els.canvasContainer.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  }

  resetView(imgW, imgH) {
    const vpW = this.els.viewport.clientWidth;
    const vpH = this.els.viewport.clientHeight;
    const scale = Math.min(vpW / imgW, vpH / imgH) * 0.9;
    const x = (vpW - imgW * scale) / 2;
    const y = (vpH - imgH * scale) / 2;
    this.transform = { scale, x, y };
    this.updateTransform();

    // Remove drop message once image is loaded
    this.els.dropMessage.style.display = 'none';
  }

  resizeCanvases(w, h) {
    Object.values(this.els.canvases).forEach(c => { c.width = w; c.height = h; });
    this.resetView(w, h);
  }

  drawInputImage(img) {
    this.els.ctx.input.drawImage(img, 0, 0);
    this.els.canvases.output.style.visibility = 'hidden';
  }

  redrawMarkerCanvas() {
    const { width, height, markerBuffer } = this.state;
    if (width === 0) return;
    const ctx = this.els.ctx.marker;
    ctx.clearRect(0, 0, width, height);

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;
    for (let i = 0; i < width * height; i++) {
      const labelId = markerBuffer[i];
      if (labelId !== 0) {
        const c = this.state.getColor(labelId);
        data[i * 4 + 0] = c.r;
        data[i * 4 + 1] = c.g;
        data[i * 4 + 2] = c.b;
        data[i * 4 + 3] = Math.floor(c.a * 255);
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  drawCircle(cx, cy, r, labelId) {
    if (this.els.chkHideMarker.checked) return;
    const c = this.state.getColor(labelId);
    const ctx = this.els.ctx.marker;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    if (labelId === 0) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fill();
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${c.a})`;
      ctx.fill();
    }
    ctx.restore();
  }

  updatePaletteUI() {
    const container = this.els.palette;
    container.innerHTML = '';

    Object.keys(this.state.labels).forEach(key => {
      const id = Number(key);
      if (id === 0) return;

      const c = this.state.labels[id];
      const btn = document.createElement('button');
      btn.className = 'label-btn';
      btn.textContent = id === 1 ? `BG` : `Col${id}`;
      btn.style.backgroundColor = c.hex;
      const brightness = (c.r * 299 + c.g * 587 + c.b * 114) / 1000;
      btn.style.color = brightness > 125 ? 'black' : 'white';

      // Box-shadow selection style (as requested)
      btn.style.border = '1px solid rgba(0,0,0,0.2)';
      if (id === this.state.currentLabelId) {
        btn.style.boxShadow = 'inset 0 0 0 3px black';
        btn.style.fontWeight = 'bold';
      } else {
        btn.style.boxShadow = 'none';
        btn.style.fontWeight = 'normal';
      }
      btn.style.padding = '5px 2px';
      btn.style.whiteSpace = 'nowrap';
      btn.style.overflow = 'hidden';
      btn.style.textOverflow = 'ellipsis';
      btn.style.fontSize = '0.9em';

      btn.addEventListener('click', () => this.handlers.onLabelSelect(id));
      container.appendChild(btn);
    });

    // Sync Controls
    const currId = this.state.currentLabelId;
    const curr = this.state.getColor(currId);
    this.els.colorPicker.value = curr.hex;
    this.els.alphaInput.value = curr.a;
    this.els.currentLabelName.textContent = currId === 1 ? `BG (1)` : `Col ${currId}`;
    this.els.btnDeleteLabel.disabled = (currId === 1);
  }

  drawResult(resultLabelMap) {
    // inputData (グレースケール画像) を参照するために取得
    const { width, height, inputData } = this.state;
    const ctx = this.els.ctx.output;
    const imgData = ctx.createImageData(width, height);
    const d = imgData.data;

    for (let i = 0; i < width * height; i++) {
      const labelId = resultLabelMap[i];

      if (labelId >= 2) {
        const c = this.state.getColor(labelId);
        // 入力画像の輝度を取得 (R成分に格納されている)
        const lum = inputData[i * 4];

        // 乗算合成: 輝度 * (色 / 255)
        // downloadImageと同じ計算式を適用
        d[i * 4 + 0] = Math.floor(lum * (c.r / 255));
        d[i * 4 + 1] = Math.floor(lum * (c.g / 255));
        d[i * 4 + 2] = Math.floor(lum * (c.b / 255));

        // 不透明(255)にして、下にあるcanvasInput(グレースケール)を
        // この「色付き乗算画像」で完全に上書きする
        d[i * 4 + 3] = 255;
      } else {
        // 背景部分は透明のままにし、下のcanvasInputが見えるようにする
        d[i * 4 + 3] = 0;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    this.els.canvases.output.style.visibility = 'visible';
  }
  
  updateDownloadButtons(hasResult) {
    this.els.btnDownloadImg.disabled = !hasResult;
    this.els.btnDownloadMask.disabled = !hasResult;
  }

  getParameters() {
    return {
      bbThreshold: parseFloat(this.els.inputs.bb.value),
      padding: parseInt(this.els.inputs.padding.value),
      sigma: parseFloat(this.els.inputs.sigma.value),
      maxIter: parseInt(this.els.inputs.maxIter.value),
      bfsNum: parseInt(this.els.inputs.bfsNum.value),
      strength: parseFloat(this.els.inputs.strength.value),
      isDynamic: this.els.chkDynamic.checked
    };
  }
}

// ==========================================
// 3. SERVICE
// ==========================================
const GraphCutService = {
  async run(state, params) {
    if (!state.isImageLoaded) return;

    // 1. Preprocess Image
    const imageProcResult = await imageProc(state.inputData, state.width, state.height);
    const normalizedR = extractNormalizedR(imageProcResult);
    const { width, height } = state;
    const numPixels = width * height;

    // 2. Identify Objects (Labels >= 2)
    const objectIds = Object.keys(state.labels).map(Number).filter(id => id >= 2);

    // Result Buffer (Initialize with 0)
    const finalLabelMap = new Uint8Array(numPixels).fill(0);

    // ---------------------------------------------
    // Multi-Label Loop: One-vs-Rest for each Object
    // ---------------------------------------------
    if (objectIds.length === 0) {
      console.log("No object labels defined.");
      return null;
    }

    // Calculate BBox of visual features (for padding)
    // Note: Using BBThreshold on normalizedR to guess object extents
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

    for (const targetId of objectIds) {
      console.log(`Processing Object ID: ${targetId}...`);

      // A. Create Temporary Marker Buffer for Binary Cut
      // Target(2) vs Background(1)
      // - User's Target ID -> 2 (Source)
      // - User's Other IDs -> 1 (Sink)
      // - User's BG (1) -> 1 (Sink)
      // - Padding Area -> 1 (Sink) [If not marked by user]

      const tempMarker = new Int32Array(numPixels);

      for (let i = 0; i < numPixels; i++) {
        const uid = state.markerBuffer[i];
        if (uid === targetId) {
          tempMarker[i] = 2; // Source
        } else if (uid !== 0) {
          tempMarker[i] = 1; // Sink (Explicit BG or Other Objects)
        } else {
          tempMarker[i] = 0; // Unknown
        }
      }

      // B. Apply Auto Padding (Set implicit Sink around BBox)
      if (hasContent) {
        const pad = params.padding;
        // Fill outside of [minX-pad, maxX+pad] x [minY-pad, maxY+pad] with Sink(1)
        // ONLY if user hasn't marked it (tempMarker[i] == 0)
        const pMinX = Math.max(0, minX - pad);
        const pMaxX = Math.min(width - 1, maxX + pad);
        const pMinY = Math.max(0, minY - pad);
        const pMaxY = Math.min(height - 1, maxY + pad);

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            if (x < pMinX || x > pMaxX || y < pMinY || y > pMaxY) {
              const idx = y * width + x;
              if (tempMarker[idx] === 0) {
                tempMarker[idx] = 1; // Implicit Sink
              }
            }
          }
        }
      }

      // C. JFA (Binary)
      const { distanceMap } = await runJumpFloodingWebGPU(width, height, tempMarker);

      // D. Push-Relabel (Binary)
      const bfsFreq = params.bfsNum > 0 ? Math.floor(params.maxIter / params.bfsNum) : params.maxIter;
      const prResult = await runPushRelabelWebGPU(
        width, height, normalizedR, tempMarker, distanceMap,
        {
          strength: params.strength,
          sigma: params.sigma,
          maxIter: params.maxIter,
          bfsFreq: bfsFreq
        }
      );

      // E. Merge Result
      // If segmented as Foreground (1), assign targetId to final map
      const seg = prResult.segmentation; // 0 or 1
      for (let i = 0; i < numPixels; i++) {
        if (seg[i] === 1) {
          finalLabelMap[i] = targetId;
        }
      }
    }

    state.latestSegmentation = finalLabelMap;
    
    return finalLabelMap;
  }
};

// ==========================================
// 4. MAIN CONTROLLER
// ==========================================
export async function main() {
  const state = new AppState();

  const handlers = {
    onFileLoad: (file) => {
      if (!file) return;
      const img = new Image();
      img.onload = () => {
        // Prepare View
        view.resizeCanvases(img.width, img.height);

        getImageDataFromFileInput({ files: [file] }).then(res => {
          convertToGrayscale(res.data);
          state.reset(img.width, img.height, res.data);

          // Draw Grayscale to Input Canvas
          const ctx = view.els.ctx.input;
          const idata = ctx.createImageData(img.width, img.height);
          idata.data.set(res.data);
          ctx.putImageData(idata, 0, 0);

          view.updatePaletteUI();
          view.updateDownloadButtons(false);
        });
      };
      img.src = URL.createObjectURL(file);
    },

    onLabelSelect: (id) => { state.currentLabelId = id; view.updatePaletteUI(); },
    onAddLabel: () => { state.currentLabelId = state.addLabel(); view.updatePaletteUI(); },
    onDeleteLabel: () => { state.removeLabel(state.currentLabelId); view.updatePaletteUI(); view.redrawMarkerCanvas(); },
    onColorChange: (hex) => { state.updateLabelColor(state.currentLabelId, hex); view.updatePaletteUI(); view.redrawMarkerCanvas(); },
    onAlphaChange: (alpha) => { state.updateLabelColor(state.currentLabelId, undefined, alpha); view.redrawMarkerCanvas(); },
    onColorEditEnd: () => {
      const params = view.getParameters();

      if (params.isDynamic) {
        handlers.onRun();
      } else if (state.latestSegmentation) {
        // 自動実行が無効でも、すでに計算結果があるなら
        // 新しい色/透明度で結果表示だけ更新する
        view.drawResult(state.latestSegmentation);

        // ※マスク画像をダウンロードする際の色も変わるため、
        // 計算結果(latestSegmentation)自体は変わらなくても
        // 表示とダウンロード用データのために drawResult は呼んでおくのが親切です
      }
    },
    onClearMarkers: () => { state.markerBuffer.fill(0); view.redrawMarkerCanvas(); },

    onDraw: (cx, cy, shiftKey) => {
      const r = state.brushSize;
      const labelId = shiftKey ? 0 : state.currentLabelId;
      view.drawCircle(cx, cy, r, labelId);
      const { width, height, markerBuffer } = state;
      const r2 = r * r;
      const minX = Math.max(0, cx - r);
      const maxX = Math.min(width - 1, cx + r);
      const minY = Math.max(0, cy - r);
      const maxY = Math.min(height - 1, cy + r);
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) {
            markerBuffer[y * width + x] = labelId;
          }
        }
      }
    },

    onDrawEnd: () => {
      if (view.getParameters().isDynamic) handlers.onRun();
    },

    onRun: async () => {
      const params = view.getParameters();
      const resultMap = await GraphCutService.run(state, params);
      if (resultMap) {
        view.drawResult(resultMap);
        view.updateDownloadButtons(true);
      }
    },

    onDownloadImage: () => {
      if (!state.latestSegmentation) return;
      const { width, height, inputData, latestSegmentation } = state;
      downloadBufferAsImage(width, height, (data) => {
        for (let i = 0; i < width * height; i++) {
          const labelId = latestSegmentation[i];
          const luminance = inputData[i * 4];
          if (labelId >= 2) {
            const c = state.getColor(labelId);
            data[i * 4 + 0] = Math.floor(luminance * (c.r / 255));
            data[i * 4 + 1] = Math.floor(luminance * (c.g / 255));
            data[i * 4 + 2] = Math.floor(luminance * (c.b / 255));
            data[i * 4 + 3] = 255;
          } else {
            data[i * 4 + 0] = 0; data[i * 4 + 1] = 0; data[i * 4 + 2] = 0; data[i * 4 + 3] = 0;
          }
        }
      }, 'result_image.png');
    },

    onDownloadMask: () => {
      if (!state.latestSegmentation) return;
      const { width, height, latestSegmentation } = state;
      downloadBufferAsImage(width, height, (data) => {
        for (let i = 0; i < width * height; i++) {
          const labelId = latestSegmentation[i];
          if (labelId >= 2) {
            const c = state.getColor(labelId);
            data[i * 4 + 0] = c.r;
            data[i * 4 + 1] = c.g;
            data[i * 4 + 2] = c.b;
            data[i * 4 + 3] = Math.floor(c.a * 255);
          } else {
            data[i * 4 + 0] = 0; data[i * 4 + 1] = 0; data[i * 4 + 2] = 0; data[i * 4 + 3] = 0;
          }
        }
      }, 'result_mask.png');
    }
  };

  const view = new AppView(state, handlers);
}