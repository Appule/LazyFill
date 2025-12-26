import { getImageDataFromFileInput, convertToGrayscale, downloadBufferAsImage } from './fileIO.js';
import { extractNormalizedR } from './flatDataUtils.js';
import { imageProc } from './wgpuProc.js';
import { runJumpFloodingWebGPU } from './JFA_GPU.js';
import { runPushRelabelWebGPU } from './PushRelabel_GPU.js';
import { MarkerRenderer } from './MarkerRenderer.js';

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
      0: { r: 0, g: 0, b: 0, a: 0.0, hex: '#000000' },
      1: { r: 0, g: 0, b: 255, a: 1.0, hex: '#0000ff' },
      2: { r: 128, g: 128, b: 128, a: 1.0, hex: '#808080' }
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
    this.markerRenderer = new MarkerRenderer();

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
      btnSave: document.getElementById('btnSave'),
      btnLoad: document.getElementById('btnLoad'),
      inpLoad: document.getElementById('inpLoad'),
      inpZoomLevel: document.getElementById('inpZoomLevel'),
      btnRun: document.getElementById('btnRun'),
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
      chkDynamic: document.getElementById('chkDynamic'),
      chkShowMarker: document.getElementById('chkShowMarker'),
      btnAutoMark: document.getElementById('btnAutoMark'),
      spinner: document.getElementById('loadingSpinner'),
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

    this.init();
    this.bindEvents();
  }

  async init() {
    await this.markerRenderer.init();

    // 【追加】ピクセル描画（ドット絵ライク）を強制するCSSスタイルを適用
    // CSSファイルに書いても良いですが、ここで確実に適用します
    const canvasStyle = 'image-rendering: pixelated; image-rendering: crisp-edges;';
    Object.values(this.els.canvases).forEach(c => {
      c.style = canvasStyle;
    });
  }

  async redrawMarkers() {
    if (!this.state.markerBuffer) return;
    const { width, height, markerBuffer, labels } = this.state;

    // WebGPUレンダラーでピクセルデータを生成 (枠線付き)
    const pixelData = await this.markerRenderer.render(width, height, markerBuffer, labels);
    const imgData = new ImageData(pixelData, width, height);

    // Canvasをクリアして描き直す
    this.els.ctx.marker.clearRect(0, 0, width, height);
    this.els.ctx.marker.putImageData(imgData, 0, 0);
  }

  drawPreviewCircle(cx, cy, radius, colorHex) {
    const ctx = this.els.ctx.marker;
    ctx.fillStyle = colorHex;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  drawPreviewEraser(cx, cy, radius) {
    const ctx = this.els.ctx.marker;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
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
    this.els.btnSave.addEventListener('click', () => this.handlers.onSaveProject());
    this.els.btnLoad.addEventListener('click', () => document.getElementById('inpLoad').click());
    this.els.inpLoad.addEventListener('change', (e) => this.handlers.onLoadProject(e.target.files[0]));
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

    // マーカー表示切替: Viewのメソッドを直接呼ぶ（ご提示のパターン）
    this.els.chkShowMarker.addEventListener('change', () => this.updateLayerVisibility());

    // 背景透過切替: 結果の再描画が必要なため Handler 経由にする
    this.els.chkTransparent.addEventListener('change', () => this.handlers.onToggleTransparent());

    this.els.btnAutoMark.addEventListener('click', () => {
      this.handlers.onAutoMark();
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

  setLoading(isLoading) {
    this.els.spinner.style.display = isLoading ? 'block' : 'none';
    // ボタンの連打防止
    this.els.btnAutoMark.disabled = isLoading;
    this.els.btnRun.disabled = isLoading;
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
    // コンテナとキャンバスのサイズ設定
    this.els.canvasContainer.style.width = w + 'px';
    this.els.canvasContainer.style.height = h + 'px';

    Object.values(this.els.canvases).forEach(c => {
      c.width = w;
      c.height = h;
    });

    // 【修正】画面中央に来るようにオフセットを計算する
    const vw = this.els.viewport.clientWidth;
    const vh = this.els.viewport.clientHeight;

    // (ビューポート - 画像) / 2 で中央位置を算出
    // 画像の方が大きい場合はマイナスになり、正しく中央基準ではみ出します
    const startX = Math.floor((vw - w) / 2);
    const startY = Math.floor((vh - h) / 2);

    // 初期化 (scale: 1.0, x: 中央, y: 中央)
    this.transform = { scale: 1.0, x: startX, y: startY };
    this.updateTransform();
  }

  drawInputImage(img) {
    this.els.ctx.input.drawImage(img, 0, 0);
    this.els.canvases.output.style.visibility = 'hidden';
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
    const hasResult = !!this.state.latestSegmentation;
    const showMarker = document.getElementById('chkShowMarker').checked;

    // 1. Input Canvas (元画像)
    // 結果があれば非表示、なければ表示
    this.els.canvases.input.style.display = hasResult ? 'none' : 'block';
    this.els.canvases.input.style.zIndex = 0;

    // 2. Output Canvas (結果)
    // 結果があれば表示、なければ非表示
    this.els.canvases.output.style.display = hasResult ? 'block' : 'none';
    this.els.canvases.output.style.zIndex = 1;

    // 3. Marker Canvas (マーカー)
    // チェックボックスの状態に従う
    this.els.canvases.marker.style.display = showMarker ? 'block' : 'none';
    // マーカーは常に最前面
    this.els.canvases.marker.style.zIndex = 10;
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

  drawResult(labelMap) {
    const { width, height, labels, inputData } = this.state;
    const imgData = this.els.ctx.output.createImageData(width, height);
    const data = imgData.data;
    const isTransparent = document.getElementById('chkTransparent').checked;

    for (let i = 0; i < width * height; i++) {
      const labelId = labelMap[i];
      // 入力画像の輝度 (グレースケール前提なのでR成分を使用、または平均)
      // inputDataはRGBAなので4倍してアクセス
      const luminance = inputData[i * 4];

      const idx = i * 4;

      if (labelId >= 2) {
        // --- 前景 (マスクあり) ---
        // 黒い線(輝度が低い)は残したい -> 色 * 輝度 (乗算合成)
        const c = labels[labelId];
        // 輝度(0~255) を 0.0~1.0 にして乗算
        const lumRatio = luminance / 255.0;

        data[idx] = c.r * lumRatio;     // R
        data[idx + 1] = c.g * lumRatio; // G
        data[idx + 2] = c.b * lumRatio; // B
        data[idx + 3] = 255;            // A (不透明)
      } else {
        // --- 背景 (マスクなし) ---
        if (isTransparent) {
          // 背景透過ON: アルファ値を 1.0 - 輝度 とする
          // 白い部分(255) -> 透明(0), 黒い線(0) -> 不透明(255)
          data[idx] = 0; // 色は黒にしておく（線用）
          data[idx + 1] = 0;
          data[idx + 2] = 0;
          data[idx + 3] = 255 - luminance;
        } else {
          // 背景透過OFF: 不透明 (元画像をそのまま表示)
          data[idx] = luminance;
          data[idx + 1] = luminance;
          data[idx + 2] = luminance;
          data[idx + 3] = 255;
        }
      }
    }
    this.els.ctx.output.putImageData(imgData, 0, 0);

    // 描画後にレイヤー表示を更新（結果が出たのでInputを隠すため）
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
  async execAutoMark(state, bbThreshold, padding) {
    if (!state.isImageLoaded) return;

    // エッジデータの準備 (重い処理なので非同期)
    const imageProcResult = await imageProc(state.inputData, state.width, state.height);
    const normalizedR = extractNormalizedR(imageProcResult);

    const { width, height, markerBuffer } = state;
    const numPixels = width * height;
    const targetId = 2; // グレーのオブジェクトID固定

    // バウンディングボックス計算
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let hasContent = false;

    // normalizedRを走査して全体BBを求める
    for (let i = 0; i < numPixels; i++) {
      if (normalizedR[i] > bbThreshold) {
        const x = i % width;
        const y = Math.floor(i / width);
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
        hasContent = true;
      }
    }

    if (!hasContent) return; // 線が見つからなければ終了

    // Padding適用
    const pMinX = Math.max(0, minX - padding);
    const pMaxX = Math.min(width - 1, maxX + padding);
    const pMinY = Math.max(0, minY - padding);
    const pMaxY = Math.min(height - 1, maxY + padding);

    // 4方向スキャンで markerBuffer に書き込み
    // ※ 既にマーカーがある場所(markerBuffer[idx] !== 0)は上書きしない
    const writeMarker = (idx) => {
      if (markerBuffer[idx] === 0) {
        markerBuffer[idx] = targetId;
        state.updatePixelCount(targetId, 1);
      }
    };

    // (1) Top -> Bottom
    for (let x = pMinX; x <= pMaxX; x++) {
      let hitLine = false;
      for (let y = pMinY; y <= pMaxY; y++) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    // (2) Bottom -> Top
    for (let x = pMinX; x <= pMaxX; x++) {
      let hitLine = false;
      for (let y = pMaxY; y >= pMinY; y--) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    // (3) Left -> Right
    for (let y = pMinY; y <= pMaxY; y++) {
      let hitLine = false;
      for (let x = pMinX; x <= pMaxX; x++) {
        const idx = y * width + x;
        if (normalizedR[idx] > bbThreshold) hitLine = true;
        else if (hitLine) { writeMarker(idx); break; }
      }
    }
    // (4) Right -> Left
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
    if (!state.isImageLoaded) return;

    const imageProcResult = await imageProc(state.inputData, state.width, state.height);
    const normalizedR = extractNormalizedR(imageProcResult);
    const { width, height } = state;
    const numPixels = width * height;

    const objectIds = Object.keys(state.labels)
      .map(Number)
      .filter(id => id >= 2)
      .sort((a, b) => b - a);

    const finalLabelMap = new Uint8Array(numPixels).fill(0);

    if (objectIds.length === 0) return finalLabelMap; // Return empty map if no objects

    // 1. 全体のBB計算
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

    // 何も描画範囲がない場合、処理しない
    if (!hasContent) return finalLabelMap;

    // パディング適用後のBB (計算領域)
    const pad = params.padding;
    const pMinX = Math.max(0, minX - pad);
    const pMaxX = Math.min(width - 1, maxX + pad);
    const pMinY = Math.max(0, minY - pad);
    const pMaxY = Math.min(height - 1, maxY + pad);

    // BBの幅と高さ
    const bbWidth = pMaxX - pMinX + 1;
    const bbHeight = pMaxY - pMinY + 1;

    // 【追加】BB情報をまとめる
    const bbInfo = { minX: pMinX, minY: pMinY, width: bbWidth, height: bbHeight };

    for (const targetId of objectIds) {
      const count = state.labelPixelCounts[targetId] || 0;
      if (count <= 0) continue;

      console.log(`Processing Obj ${targetId} (Size: ${bbWidth}x${bbHeight}, Offset: ${pMinX},${pMinY})`);

      const tempMarker = new Int32Array(numPixels);

      // CPU最適化: BB内のみ走査して tempMarker を作成
      // 画像全体(numPixels)を舐めると遅いので、必要な矩形部分だけ処理
      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        for (let x = pMinX; x <= pMaxX; x++) {
          const idx = rowOffset + x;
          const uid = state.markerBuffer[idx];

          if (uid === targetId) tempMarker[idx] = 2; // Source
          else if (uid !== 0) tempMarker[idx] = 1;   // Sink (Other objects)
          else tempMarker[idx] = 0;                  // Unknown
        }
      }

      // BB外周にSink設置 (BB内だけ計算すればよくなる)
      // Top & Bottom Edge of BB
      for (let x = pMinX; x <= pMaxX; x++) {
        const topIdx = pMinY * width + x;
        const botIdx = pMaxY * width + x;
        if (tempMarker[topIdx] === 0) tempMarker[topIdx] = 1;
        if (tempMarker[botIdx] === 0) tempMarker[botIdx] = 1;
      }
      // Left & Right Edge of BB
      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        const leftIdx = rowOffset + pMinX;
        const rightIdx = rowOffset + pMaxX;
        if (tempMarker[leftIdx] === 0) tempMarker[leftIdx] = 1;
        if (tempMarker[rightIdx] === 0) tempMarker[rightIdx] = 1;
      }

      // JFAにBB情報を渡す
      const { distanceMap } = await runJumpFloodingWebGPU(width, height, tempMarker, bbInfo);

      const bfsFreq = params.bfsNum > 0 ? Math.floor(params.maxIter / params.bfsNum) : params.maxIter;

      // PushRelabelにBB情報を渡す
      const prResult = await runPushRelabelWebGPU(
        width, height, normalizedR, tempMarker, distanceMap,
        { strength: params.strength, sigma: params.sigma, maxIter: params.maxIter, bfsFreq: bfsFreq },
        bbInfo
      );

      const seg = prResult.segmentation;

      // 結果のマージ (BB内のみ走査してマージ)
      for (let y = pMinY; y <= pMaxY; y++) {
        const rowOffset = y * width;
        for (let x = pMinX; x <= pMaxX; x++) {
          const i = rowOffset + x;
          if (seg[i] === 1) finalLabelMap[i] = targetId;
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

      if (view.els.dropMessage) {
        view.els.dropMessage.style.display = 'none';
      }

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

          view.updateLayerVisibility();
          view.redrawMarkers();
        });
      };
      img.src = URL.createObjectURL(file);
    },

    onAutoMark: async () => {
      if (!state.isImageLoaded) return;

      try {
        const params = view.getParameters();
        await GraphCutService.execAutoMark(state, params.bbThreshold, params.padding);
        state.isMarkerDirty = true;

        await view.redrawMarkers();

        // if chk dynamic, auto run
        if (params.isDynamic) handlers.onRun();

      } catch (e) {
        console.error(e);
        alert("Auto Mark Error: " + e.message);
      }
    },

    onLabelSelect: (id) => { state.currentLabelId = id; view.updatePaletteUI(); view.setToolMode('brush'); },
    onAddLabel: () => { state.currentLabelId = state.addLabel(); view.updatePaletteUI(); },
    onDeleteLabel: () => { state.removeLabel(state.currentLabelId); state.isMarkerDirty = true; view.updatePaletteUI(); view.redrawMarkers(); },
    onColorChange: (hex) => { 
      state.updateLabelColor(state.currentLabelId, hex); view.updatePaletteUI(); view.redrawMarkers();
      if (state.latestSegmentation) view.drawResult(state.latestSegmentation);
    },
    onAlphaChange: (alpha) => {
      state.updateLabelColor(state.currentLabelId, undefined, alpha); view.redrawMarkers();
      if (state.latestSegmentation) view.drawResult(state.latestSegmentation);
    },
    onToggleMarker: () => {
      view.updateLayerVisibility();
    },

    onToggleTransparent: () => {
      if (state.latestSegmentation) {
        view.drawResult(state.latestSegmentation);
      }
    },

    onClearMarkers: () => {
      state.markerBuffer.fill(0);
      state.isMarkerDirty = true;
      state.labelPixelCounts = {};

      // マーカーが消えるので、結果もクリアすべき場合はここで制御
      // 今回はマーカーだけ消す挙動とします
      view.redrawMarkers();
    },

    onDraw: (cx, cy, isEraser) => {
      let r = state.brushSize;
      const labelId = isEraser ? 0 : state.currentLabelId;
      const { width, height, markerBuffer } = state;

      // ViewのContext準備
      const ctx = view.els.ctx.marker;

      // 色の準備 (消しゴムなら透明にするための設定)
      if (isEraser) {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'rgba(0,0,0,1)'; // 色は何でも良い
      } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = state.getColor(labelId).hex;
      }

      // --- ピクセル更新ループ ---
      // サイズ1の場合とサイズ2以上の場合で共通化して記述することも可能ですが、
      // 既存ロジックを維持しつつ、更新があった場所だけ fillRect します。

      if (r === 1) {
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
          const idx = cy * width + cx;
          const oldId = markerBuffer[idx];
          if (oldId !== labelId) {
            // Model Update
            state.updatePixelCount(oldId, -1);
            state.updatePixelCount(labelId, 1);
            markerBuffer[idx] = labelId;
            state.isMarkerDirty = true;

            // 【修正】View Update: バッファが変わったこの1点を描画
            ctx.fillRect(cx, cy, 1, 1);
          }
        }
      } else {
        // 円形範囲
        r -= 1;
        const r2 = r * r;
        const minX = Math.max(0, cx - r);
        const maxX = Math.min(width - 1, cx + r);
        const minY = Math.max(0, cy - r);
        const maxY = Math.min(height - 1, cy + r);

        for (let y = minY; y <= maxY; y++) {
          for (let x = minX; x <= maxX; x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) {
              const idx = y * width + x;
              const oldId = markerBuffer[idx];

              if (oldId !== labelId) {
                // Model Update
                state.updatePixelCount(oldId, -1);
                state.updatePixelCount(labelId, 1);
                markerBuffer[idx] = labelId;
                state.isMarkerDirty = true;

                // 【修正】View Update: バッファが変わったこの1点を描画
                ctx.fillRect(x, y, 1, 1);
              }
            }
          }
        }
      }

      // 描画設定を戻す
      ctx.globalCompositeOperation = 'source-over';
    },

    onDrawEnd: async () => {
      await view.redrawMarkers();
      if (view.getParameters().isDynamic && state.isMarkerDirty) {
        handlers.onRun();
      }
    },

    onRun: async () => {
      const params = view.getParameters();
      view.setLoading(true);
      await new Promise(r => setTimeout(r, 10));

      try {
        const resultMap = await GraphCutService.run(state, params);
        if (resultMap) {
          view.drawResult(resultMap); // 内部で updateLayerVisibility も呼ばれる
          view.updateDownloadButtons(true);
          state.isMarkerDirty = false;
        }
      } catch (e) {
        console.error(e);
        alert("Run Error: " + e.message);
      } finally {
        view.setLoading(false);
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
              data[i * 4 + 0] = 0;
              data[i * 4 + 1] = 0;
              data[i * 4 + 2] = 0;
              data[i * 4 + 3] = 255 - luminance;
            } else {
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
    },

    onSaveProject: () => {
      if (!state.isImageLoaded) {
        alert("No image loaded.");
        return;
      }

      // 1. マーカーをPNG Base64に変換
      // 不可視のCanvasを作成して描画
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = state.width;
      tempCanvas.height = state.height;
      const ctx = tempCanvas.getContext('2d');
      const imgData = ctx.createImageData(state.width, state.height);
      const data = imgData.data;
      const buffer = state.markerBuffer;

      for (let i = 0; i < buffer.length; i++) {
        const id = buffer[i];
        // IDを赤チャンネルに格納 (IDが255以下である前提)
        // IDが0なら透明、それ以外は不透明
        data[i * 4 + 0] = id & 0xFF; // R
        data[i * 4 + 1] = 0;         // G
        data[i * 4 + 2] = 0;         // B
        data[i * 4 + 3] = id > 0 ? 255 : 0; // Alpha
      }
      ctx.putImageData(imgData, 0, 0);
      const markerBase64 = tempCanvas.toDataURL('image/png');

      // 2. JSONデータ構築
      const projectData = {
        version: 1.0,
        width: state.width,
        height: state.height,
        params: view.getParameters(), // 現在のUIパラメータ
        labels: state.labels,         // 色設定など
        markers: markerBase64
      };

      // 3. ダウンロード処理
      const blob = new Blob([JSON.stringify(projectData)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "lazyfill_project.json";
      a.click();
      URL.revokeObjectURL(url);
    },

    onLoadProject: (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);

          // 1. パラメータとラベルの復元 (これは常に実行)
          // ※ view.setParameters のようなメソッドを作るか、個別に設定
          if (json.params) {
            // UIへの反映実装例
            if (json.params.padding) document.getElementById('inpPadding').value = json.params.padding;
            if (json.params.sigma) document.getElementById('inpSigma').value = json.params.sigma;
            // ... 他のパラメータも同様に ...
            // チェックボックス等も復元推奨
          }

          if (json.labels) {
            state.labels = json.labels;
            view.updatePaletteUI();
          }

          // 2. 画像サイズチェック
          const currentW = state.width;
          const currentH = state.height;

          if (state.isImageLoaded && (json.width !== currentW || json.height !== currentH)) {
            alert(`画像サイズが一致しません。\n現在: ${currentW}x${currentH}\n保存データ: ${json.width}x${json.height}\n\nマーカー以外の設定のみ読み込みました。`);
            // マーカー読み込みはスキップ
          } else if (state.isImageLoaded && json.markers) {
            // 3. マーカーの復元 (サイズ一致時)
            const img = new Image();
            img.onload = () => {
              const tempCanvas = document.createElement('canvas');
              tempCanvas.width = currentW;
              tempCanvas.height = currentH;
              const ctx = tempCanvas.getContext('2d');
              ctx.drawImage(img, 0, 0);

              const imgData = ctx.getImageData(0, 0, currentW, currentH);
              const data = imgData.data;

              // バッファに書き戻し
              state.markerBuffer.fill(0);
              state.labelPixelCounts = {}; // カウントリセット

              for (let i = 0; i < state.markerBuffer.length; i++) {
                // 保存時に Rチャネル にIDを入れたので、そこから復元
                const id = data[i * 4 + 0];
                if (id > 0) {
                  state.markerBuffer[i] = id;
                  state.updatePixelCount(id, 1);
                }
              }

              state.isMarkerDirty = true;
              view.redrawMarkers();
              handlers.onRun();
              alert("プロジェクトを読み込みました。");
            };
            img.src = json.markers;
          } else {
            alert("画像がロードされていないか、マーカーデータがありません。設定のみ読み込みました。");
          }

        } catch (err) {
          console.error(err);
          alert("プロジェクトファイルの読み込みに失敗しました。");
        }
      };
      reader.readAsText(file);
    }
  };

  const view = new AppView(state, handlers);
}