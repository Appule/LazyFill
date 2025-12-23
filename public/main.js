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
      1: { r: 0, g: 0, b: 255, a: 1.0, hex: '#0000ff' },    // BG
      2: { r: 255, g: 0, b: 0, a: 1.0, hex: '#ff0000' }     // Default Object
    };
    this.labelPixelCounts = {};
    this.isMarkerDirty = false;

    this.toolMode = 'brush';
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
    this.isMarkerDirty = false;
    this.isImageLoaded = true;
    this.labelPixelCounts = {};
    Object.keys(this.labels).forEach(k => this.labelPixelCounts[k] = 0);
  }

  updatePixelCount(id, delta) {
    if (!this.labelPixelCounts[id]) this.labelPixelCounts[id] = 0;
    this.labelPixelCounts[id] += delta;
  }

  addLabel() {
    const ids = Object.keys(this.labels).map(Number);
    const newId = Math.max(...ids) + 1;

    const r = Math.floor(Math.random() * 200);
    const g = Math.floor(Math.random() * 200);
    const b = Math.floor(Math.random() * 200);
    const hex = "#" + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');

    this.labels[newId] = { r, g, b, a: 1.0, hex };
    return newId;
  }
  
  removeLabel(id) {
    if (id <= 1) return;
    delete this.labels[id];

    // 削除対象のカウントをリセット
    this.labelPixelCounts[id] = 0;

    for (let i = 0; i < this.markerBuffer.length; i++) {
      if (this.markerBuffer[i] === id) {
        this.markerBuffer[i] = 0;
      }
    }
    if (this.currentLabelId === id) this.currentLabelId = 1;
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
      inpZoomLevel: document.getElementById('inpZoomLevel'),
      btnRun: document.getElementById('btnRun'),
      chkDynamic: document.getElementById('chkDynamic'),
      btnDownloadImg: document.getElementById('btnDownloadImg'),
      chkTransparent: document.getElementById('chkTransparent'),
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
      brushGuide: document.getElementById('brushGuide'),
      toolRadios: document.querySelectorAll('input[name="toolMode"]'),
      dispBrush: document.getElementById('dispBrushSize'),
      chkShowMarker: document.getElementById('chkShowMarker'),
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
    this.els.inpZoomLevel.addEventListener('change', (e) => {
      let percent = parseFloat(e.target.value);
      if (isNaN(percent) || percent <= 0) percent = 100;
      const newScale = percent / 100.0;
      this.setZoomManual(newScale);
    });
    this.els.btnRun.addEventListener('click', () => this.handlers.onRun());
    this.els.btnDownloadImg.addEventListener('click', () => this.handlers.onDownloadImage());
    this.els.btnDownloadMask.addEventListener('click', () => this.handlers.onDownloadMask());

    // Toggle Params Panel
    this.els.btnToggleParams.addEventListener('click', () => {
      this.els.panelParams.classList.toggle('hidden');
    });

    // --- Tools & Palette Actions ---
    // --- Tool Switching ---
    this.els.toolRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        if (e.target.checked) {
          this.state.toolMode = e.target.value;
          this.updateCursor(); // カーソル形状更新
          this.updateBrushGuideVisibility(); // Moveモードならガイド消すなど
        }
      });
    });

    // --- Marker Visibility & Auto Run Logic ---
    this.els.chkShowMarker.addEventListener('change', (e) => {
      const isVisible = e.target.checked;
      this.updateLayerVisibility();

      // 変更があった場合(isMarkerDirty)のみ実行する
      if (!isVisible && this.getParameters().isDynamic && this.state.isMarkerDirty) {
        this.handlers.onRun();
      }
    });

    // --- Buttons ---
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
    this.els.alphaInput.addEventListener('input', (e) => this.handlers.onAlphaChange(e.target.value));

    // Brush Size
    this.els.inputs.brush.addEventListener('input', (e) => {
      const size = parseInt(e.target.value);
      this.state.brushSize = size;
      this.els.dispBrush.textContent = size;
    });

    // Show Marker
    this.els.chkShowMarker.addEventListener('change', () => {
      this.updateLayerVisibility();
    });

    // --- Canvas Navigation & Drawing ---

    // Wheel Zoom
    vp.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.handleZoom(e);
      this.updateBrushGuide(e.clientX, e.clientY);
    }, { passive: false });

    // Mouse Events
    vp.addEventListener('mousedown', (e) => {
      e.preventDefault();

      // マウスボタン判定
      const isLeft = (e.button === 0);
      const isMiddle = (e.button === 1);
      const isRight = (e.button === 2);

      const mode = this.state.toolMode;

      // ■ パン(移動)開始条件:
      // 1. ミドルドラッグ or 右ドラッグ (全モード共通)
      // 2. 左ドラッグ (Moveモード時のみ)
      if (isMiddle || isRight || (isLeft && mode === 'move')) {
        this.isPanning = true;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        vp.style.cursor = 'grabbing';
        return;
      }

      // ■ 描画開始条件:
      // 左ドラッグ AND (Brush or Eraser)
      if (isLeft && (mode === 'brush' || mode === 'eraser')) {
        this.drawing = true;
        // Eraserモードなら消しゴム(labelId=0)として振る舞う
        const isEraser = (mode === 'eraser');

        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, isEraser);
      }
    });

    window.addEventListener('mousemove', (e) => {
      // 1. パン処理
      if (this.isPanning) {
        const dx = e.clientX - this.lastMousePos.x;
        const dy = e.clientY - this.lastMousePos.y;
        this.transform.x += dx;
        this.transform.y += dy;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        this.updateTransform();

        // パン中もガイド位置更新
        this.updateBrushGuide(e.clientX, e.clientY);
        return;
      }

      // 2. 描画処理
      if (this.drawing) {
        const isEraser = (this.state.toolMode === 'eraser');
        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, isEraser);
      }

      // 3. ブラシガイドの更新 (マウス移動時常時)
      this.updateBrushGuide(e.clientX, e.clientY);
    });

    window.addEventListener('mouseup', (e) => {
      // パン終了
      if (this.isPanning) {
        // 解除条件を緩くする(どのボタンが上がっても解除でOK)
        this.isPanning = false;
        this.updateCursor(); // カーソルをツール標準に戻す
      }
      // 描画終了
      else if (this.drawing && e.button === 0) {
        this.drawing = false;
        this.handlers.onDrawEnd();
      }
    });

    vp.addEventListener('contextmenu', e => e.preventDefault());
  }

  setToolMode(mode) {
    // 1. Stateを更新
    this.state.toolMode = mode;

    // 2. UI (ラジオボタン) の見た目を更新
    const radio = Array.from(this.els.toolRadios).find(r => r.value === mode);
    if (radio) {
      radio.checked = true;
    }

    // 3. カーソルとガイドの表示状態を更新
    this.updateCursor();
    this.updateBrushGuideVisibility();
  }

  // --- Cursor & Guide Helpers ---

  updateCursor() {
    const vp = this.els.viewport;
    const mode = this.state.toolMode;
    if (mode === 'move') {
      vp.style.cursor = 'grab';
    } else {
      // Brush/Eraserはガイドが出るのでカーソルは消すか十字にする
      // ここでは十字(crosshair)か、'none'にしてガイドのみにする手が一般的
      vp.style.cursor = 'crosshair';
    }
  }

  updateBrushGuideVisibility() {
    const mode = this.state.toolMode;
    // Moveモードならガイド非表示
    if (mode === 'move') {
      this.els.brushGuide.style.display = 'none';
    } else {
      this.els.brushGuide.style.display = 'block';
    }
  }

  updateBrushGuide(clientX, clientY) {
    // 画像未ロード、またはMoveモードなら更新しない
    if (!this.state.isImageLoaded || this.state.toolMode === 'move') {
      this.els.brushGuide.style.display = 'none';
      return;
    }

    this.els.brushGuide.style.display = 'block';

    // ブラシ半径(px) * 2 * 表示倍率 = 画面上の直径
    const diameter = (this.state.brushSize * 2 - 1) * this.transform.scale;

    const guide = this.els.brushGuide;
    guide.style.width = `${diameter}px`;
    guide.style.height = `${diameter}px`;
    guide.style.left = `${clientX}px`;
    guide.style.top = `${clientY}px`;

    // ※ index.htmlのCSSで transform: translate(-50%, -50%) が
    // 指定されている前提なので、left/topはマウス中心でOK
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

    // ズーム係数
    const ZOOM_FACTOR = 1.1;
    const direction = e.deltaY > 0 ? -1 : 1;
    const factor = direction > 0 ? ZOOM_FACTOR : (1 / ZOOM_FACTOR);

    let newScale = this.transform.scale * factor;

    // 制限 (1% ~ 10000%)
    newScale = Math.max(0.01, Math.min(newScale, 100.0));

    // マウス中心ズーム計算
    const rectViewport = this.els.viewport.getBoundingClientRect();
    const vpMouseX = e.clientX - rectViewport.left;
    const vpMouseY = e.clientY - rectViewport.top;
    const oldX = this.transform.x;
    const oldY = this.transform.y;

    // 実際の倍率比 (クランプされたnewScaleを使うため再計算)
    const scaleRatio = newScale / this.transform.scale;

    this.transform.x = vpMouseX - (vpMouseX - oldX) * scaleRatio;
    this.transform.y = vpMouseY - (vpMouseY - oldY) * scaleRatio;
    this.transform.scale = newScale;

    this.updateTransform();
  }
  
  setZoomManual(newScale) {
    if (!this.state.isImageLoaded) return;

    // 制限
    newScale = Math.max(0.01, Math.min(newScale, 100.0));

    const oldScale = this.transform.scale;
    const scaleRatio = newScale / oldScale;

    // ビューポートの中心を基準にズーム
    const vpW = this.els.viewport.clientWidth;
    const vpH = this.els.viewport.clientHeight;
    const centerX = vpW / 2;
    const centerY = vpH / 2;

    const oldX = this.transform.x;
    const oldY = this.transform.y;

    this.transform.x = centerX - (centerX - oldX) * scaleRatio;
    this.transform.y = centerY - (centerY - oldY) * scaleRatio;
    this.transform.scale = newScale;

    this.updateTransform();
  }

  updateTransform() {
    const { x, y, scale } = this.transform;
    this.els.canvasContainer.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;

    // 入力欄の表示を同期
    if (document.activeElement !== this.els.inpZoomLevel) {
      this.els.inpZoomLevel.value = Math.round(scale * 100);
    }
  }

  resetView(imgW, imgH) {
    const vpW = this.els.viewport.clientWidth;
    const vpH = this.els.viewport.clientHeight;

    // 画面に収まるスケール (90%)
    const scale = Math.min(vpW / imgW, vpH / imgH) * 0.9;

    const x = (vpW - imgW * scale) / 2;
    const y = (vpH - imgH * scale) / 2;

    this.transform = { scale, x, y };
    this.updateTransform();

    this.els.dropMessage.style.display = 'none';

    // 初期値をinputにも反映
    this.els.inpZoomLevel.value = Math.round(scale * 100);
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

  updateMarkerRect(minX, minY, maxX, maxY) {
    const { width, height, markerBuffer } = this.state;

    // 範囲を画像内にクリップ
    minX = Math.max(0, minX);
    minY = Math.max(0, minY);
    maxX = Math.min(width - 1, maxX);
    maxY = Math.min(height - 1, maxY);

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;

    if (w <= 0 || h <= 0) return;

    const ctx = this.els.ctx.marker;
    const imgData = ctx.getImageData(minX, minY, w, h);
    const data = imgData.data;

    // 指定矩形内のみループしてデータ更新
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // markerBuffer上の絶対座標
        const globalX = minX + x;
        const globalY = minY + y;
        const idx = globalY * width + globalX;
        const labelId = markerBuffer[idx];

        // ImageData上のインデックス
        const dataIdx = (y * w + x) * 4;

        if (labelId !== 0) {
          const c = this.state.getColor(labelId);
          data[dataIdx + 0] = c.r;
          data[dataIdx + 1] = c.g;
          data[dataIdx + 2] = c.b;
          data[dataIdx + 3] = Math.floor(c.a * 255);
        } else {
          // 消しゴム(ID:0)部分は透明にする
          data[dataIdx + 0] = 0;
          data[dataIdx + 1] = 0;
          data[dataIdx + 2] = 0;
          data[dataIdx + 3] = 0;
        }
      }
    }

    ctx.putImageData(imgData, minX, minY);
  }

  updateLayerVisibility() {
    const isMarkerVisible = this.els.chkShowMarker.checked;

    // 1. マーカーキャンバスの制御
    // CSS opacityで制御 (1:見える, 0:見えない)
    this.els.canvases.marker.style.opacity = isMarkerVisible ? '1' : '0';

    // 2. マスク(Output)キャンバスの制御
    // マーカーが表示されているなら、マスクは隠す
    // マーカーが隠されているなら、マスクを表示する(ただし結果がある場合のみ)
    if (isMarkerVisible) {
      this.els.canvases.output.style.visibility = 'hidden';
    } else {
      // 結果データが存在するかチェック (State経由またはvisibilityチェック)
      // ここでは簡易的に「データがあれば表示」としたいが、ViewはStateを直接持っているので確認
      if (this.state.latestSegmentation) {
        this.els.canvases.output.style.visibility = 'visible';
      }
    }
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
      btn.textContent = id === 1 ? `背景` : `色 ${id}`;
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
    this.els.currentLabelName.textContent = currId === 1 ? `背景` : `色 ${currId}`;
    this.els.btnDeleteLabel.disabled = (currId === 1);
  }

  drawResult(resultLabelMap) {
    const { width, height, inputData } = this.state; // inputData取得
    const ctx = this.els.ctx.output;
    const imgData = ctx.createImageData(width, height);
    const d = imgData.data;

    // ... (ピクセルデータの構築ループは変更なし) ...
    for (let i = 0; i < width * height; i++) {
      const labelId = resultLabelMap[i];
      if (labelId >= 2) {
        const c = this.state.getColor(labelId);
        const lum = inputData[i * 4];
        d[i * 4 + 0] = Math.floor(lum * (c.r / 255));
        d[i * 4 + 1] = Math.floor(lum * (c.g / 255));
        d[i * 4 + 2] = Math.floor(lum * (c.b / 255));
        d[i * 4 + 3] = 255;
      } else {
        d[i * 4 + 3] = 0;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    this.updateLayerVisibility();
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
    const objectIds = Object.keys(state.labels).map(Number).filter(id => id >= 2).sort((a, b) => b - a);

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
      const count = state.labelPixelCounts[targetId] || 0;
      if (count <= 0) {
        console.log(`Skipping Obj ${targetId} (No markers)`);
        continue;
      }

      console.log(`Processing Obj ${targetId}`);

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

    onLabelSelect: (id) => { state.currentLabelId = id; view.updatePaletteUI(); view.setToolMode('brush'); },
    onAddLabel: () => { state.currentLabelId = state.addLabel(); view.updatePaletteUI(); },
    onDeleteLabel: () => { state.removeLabel(state.currentLabelId); state.isMarkerDirty = true; view.updatePaletteUI(); view.redrawMarkerCanvas(); },
    onColorChange: (hex) => { 
      state.updateLabelColor(state.currentLabelId, hex); view.updatePaletteUI(); view.redrawMarkerCanvas();
      if (state.latestSegmentation) view.drawResult(state.latestSegmentation);
    },
    onAlphaChange: (alpha) => {
      state.updateLabelColor(state.currentLabelId, undefined, alpha); view.redrawMarkerCanvas();
      if (state.latestSegmentation) view.drawResult(state.latestSegmentation);
    },
    onClearMarkers: () => { state.markerBuffer.fill(0); state.isMarkerDirty = true; state.labelPixelCounts = {}; view.redrawMarkerCanvas(); },

    onDraw: (cx, cy, isEraser) => {
      let r = state.brushSize;
      const labelId = isEraser ? 0 : state.currentLabelId;
      const { width, height, markerBuffer } = state;

      // 更新があったかどうかのフラグ
      let changed = false;

      // 更新範囲（バウンディングボックス）の記録用
      let updateMinX = width, updateMinY = height;
      let updateMaxX = 0, updateMaxY = 0;

      // --- 1. ピクセル更新ロジック ---

      if (r === 1) {
        // 【サイズ1の場合】カーソル直下の1ピクセルのみ
        // cx, cy は既にMath.floorされている整数座標
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
          const idx = cy * width + cx;
          const oldId = markerBuffer[idx];

          if (oldId !== labelId) {
            state.updatePixelCount(oldId, -1);
            state.updatePixelCount(labelId, 1);
            markerBuffer[idx] = labelId;
            state.isMarkerDirty = true;

            // 更新範囲は1点
            updateMinX = cx; updateMaxX = cx;
            updateMinY = cy; updateMaxY = cy;
            changed = true;
          }
        }
      } else {
        // 【サイズ2以上の場合】円形範囲
        // 半径計算: r=2なら中心+1pxなど、好みに応じて調整可能ですが、
        // ここでは従来の仕様に準拠しつつ、rを半径として扱います。
        r -= 1;
        const r2 = r * r; // 半径の二乗 (距離判定用)

        // 探索範囲
        const minX = Math.max(0, cx - r);
        const maxX = Math.min(width - 1, cx + r);
        const minY = Math.max(0, cy - r);
        const maxY = Math.min(height - 1, cy + r);

        for (let y = minY; y <= maxY; y++) {
          for (let x = minX; x <= maxX; x++) {
            // 円の内側判定
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) {
              const idx = y * width + x;
              const oldId = markerBuffer[idx];

              if (oldId !== labelId) {
                state.updatePixelCount(oldId, -1);
                state.updatePixelCount(labelId, 1);
                markerBuffer[idx] = labelId;
                state.isMarkerDirty = true;
                changed = true;

                // 更新範囲を拡張
                if (x < updateMinX) updateMinX = x;
                if (x > updateMaxX) updateMaxX = x;
                if (y < updateMinY) updateMinY = y;
                if (y > updateMaxY) updateMaxY = y;
              }
            }
          }
        }
      }

      // --- 2. 描画反映 (同期) ---

      if (changed) {
        // 変更があった矩形領域のみを再描画する
        view.updateMarkerRect(updateMinX, updateMinY, updateMaxX, updateMaxY);
      }
    },

    onDrawEnd: () => {
      const isMarkerVisible = view.els.chkShowMarker.checked;
      if (!isMarkerVisible && view.getParameters().isDynamic && state.isMarkerDirty) {
        handlers.onRun();
      }
    },

    onRun: async () => {
      const params = view.getParameters();
      const resultMap = await GraphCutService.run(state, params);
      if (resultMap) {
        view.drawResult(resultMap);
        view.updateDownloadButtons(true);
        state.isMarkerDirty = false;
      }
    },

    onDownloadImage: () => {
      if (!state.latestSegmentation) return;
      const { width, height, inputData, latestSegmentation } = state;
      const isTransparent = view.els.chkTransparent.checked;
      downloadBufferAsImage(width, height, (data) => {
        for (let i = 0; i < width * height; i++) {
          const labelId = latestSegmentation[i];
          const luminance = inputData[i * 4];
          if (labelId >= 2) {
            // 前景: 乗算合成
            const c = state.getColor(labelId);
            data[i * 4 + 0] = Math.floor(luminance * (c.r / 255));
            data[i * 4 + 1] = Math.floor(luminance * (c.g / 255));
            data[i * 4 + 2] = Math.floor(luminance * (c.b / 255));
            data[i * 4 + 3] = 255;
          } else {
            // 背景: チェックボックスに応じて分岐
            if (isTransparent) {
              // 透過
              data[i * 4 + 0] = 0;
              data[i * 4 + 1] = 0;
              data[i * 4 + 2] = 0;
              data[i * 4 + 3] = 255 - luminance;
            } else {
              // 白背景
              data[i * 4 + 0] = luminance;
              data[i * 4 + 1] = luminance;
              data[i * 4 + 2] = luminance;
              data[i * 4 + 3] = 255;
            }
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