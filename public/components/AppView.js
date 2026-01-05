import { MarkerRenderer } from '../MarkerRenderer.js';

export class AppView {
  constructor(state, handlers) {
    this.state = state;
    this.handlers = handlers;
    this.markerRenderer = new MarkerRenderer();

    this.transform = { scale: 1.0, x: 0, y: 0 };
    this.isPanning = false;
    this.lastMousePos = { x: 0, y: 0 };
    this.drawing = false;

    // DOM要素の参照
    this.els = {
      viewport: document.getElementById('viewport'),
      canvasContainer: document.getElementById('canvasContainer'),
      dropMessage: document.getElementById('drop-message'),

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

      inputs: {
        bb: document.getElementById('inpBB'),
        padding: document.getElementById('inpPadding'),
        sigma: document.getElementById('inpSigma'),
        maxIter: document.getElementById('inpMaxIter'),
        bfsNum: document.getElementById('inpBfsNum'),
        strength: document.getElementById('inpStrength'),
        brush: document.getElementById('inpBrushSize'),
      },

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
    const canvasStyle = 'image-rendering: pixelated; image-rendering: crisp-edges;';
    Object.values(this.els.canvases).forEach(c => c.style = canvasStyle);
  }

  // --- Drawing Methods ---

  async redrawMarkers() {
    if (!this.state.markerBuffer) return;
    const { width, height, markerBuffer, labels } = this.state;
    const pixelData = await this.markerRenderer.render(width, height, markerBuffer, labels);
    const imgData = new ImageData(pixelData, width, height);
    this.els.ctx.marker.clearRect(0, 0, width, height);
    this.els.ctx.marker.putImageData(imgData, 0, 0);
  }

  drawInputImage(img) {
    this.els.ctx.input.drawImage(img, 0, 0);
    this.els.canvases.output.style.display = 'none';
  }

  drawResult(labelMap) {
    const { width, height, labels, inputData } = this.state;
    const imgData = this.els.ctx.output.createImageData(width, height);
    const data = imgData.data;
    const isTransparent = this.els.chkTransparent.checked;

    for (let i = 0; i < width * height; i++) {
      const idx = i * 4;
      const labelId = labelMap[i];
      const luminance = inputData[idx];

      if (labelId >= 2) {
        const c = labels[labelId];
        const lumRatio = luminance / 255.0;
        data[idx] = c.r * lumRatio;
        data[idx + 1] = c.g * lumRatio;
        data[idx + 2] = c.b * lumRatio;
        data[idx + 3] = 255;
      } else {
        if (isTransparent) {
          data[idx] = 0; data[idx + 1] = 0; data[idx + 2] = 0;
          data[idx + 3] = 255 - luminance;
        } else {
          data[idx] = luminance; data[idx + 1] = luminance; data[idx + 2] = luminance;
          data[idx + 3] = 255;
        }
      }
    }
    this.els.ctx.output.putImageData(imgData, 0, 0);
    this.updateLayerVisibility();
  }

  updateLayerVisibility() {
    const hasResult = !!this.state.latestSegmentation;
    const showMarker = this.els.chkShowMarker.checked;

    this.els.canvases.input.style.display = hasResult ? 'none' : 'block';
    this.els.canvases.input.style.zIndex = 0;
    this.els.canvases.output.style.display = hasResult ? 'block' : 'none';
    this.els.canvases.output.style.zIndex = 1;
    this.els.canvases.marker.style.display = showMarker ? 'block' : 'none';
    this.els.canvases.marker.style.zIndex = 10;
  }

  // --- UI Updates ---

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

      if (id === this.state.currentLabelId) {
        btn.style.border = '2px solid black';
        btn.style.fontWeight = 'bold';
      } else {
        btn.style.border = '1px solid #ccc';
      }
      btn.addEventListener('click', () => this.handlers.onLabelSelect(id));
      container.appendChild(btn);
    });

    const currId = this.state.currentLabelId;
    const curr = this.state.getColor(currId);
    this.els.colorPicker.value = curr.hex;
    this.els.alphaInput.value = curr.a;
    this.els.currentLabelName.textContent = currId === 1 ? `背景` : `色 ${currId}`;
    this.els.btnDeleteLabel.disabled = (currId === 1);
  }

  updateDownloadButtons(hasResult) {
    this.els.btnDownloadImg.disabled = !hasResult;
    this.els.btnDownloadMask.disabled = !hasResult;
  }

  setLoading(isLoading) {
    this.els.spinner.style.display = isLoading ? 'block' : 'none';
    this.els.btnAutoMark.disabled = isLoading;
    this.els.btnRun.disabled = isLoading;
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

  // --- Transforms & Events ---

  setToolMode(mode) {
    this.state.toolMode = mode;
    const radio = Array.from(this.els.toolRadios).find(r => r.value === mode);
    if (radio) radio.checked = true;
    this.updateCursor();
    this.updateBrushGuideVisibility();
  }

  updateCursor() {
    const vp = this.els.viewport;
    if (this.state.toolMode === 'move') vp.style.cursor = 'grab';
    else vp.style.cursor = 'crosshair';
  }

  updateBrushGuideVisibility() {
    const isMove = this.state.toolMode === 'move';
    this.els.brushGuide.style.display = isMove ? 'none' : 'block';
  }

  updateBrushGuide(clientX, clientY) {
    if (!this.state.isImageLoaded || this.state.toolMode === 'move') {
      this.els.brushGuide.style.display = 'none';
      return;
    }
    this.els.brushGuide.style.display = 'block';
    const diameter = (this.state.brushSize * 2 - 1) * this.transform.scale;
    this.els.brushGuide.style.width = `${diameter}px`;
    this.els.brushGuide.style.height = `${diameter}px`;
    this.els.brushGuide.style.left = `${clientX}px`;
    this.els.brushGuide.style.top = `${clientY}px`;
  }

  resizeCanvases(w, h) {
    this.els.canvasContainer.style.width = w + 'px';
    this.els.canvasContainer.style.height = h + 'px';
    Object.values(this.els.canvases).forEach(c => { c.width = w; c.height = h; });

    const vw = this.els.viewport.clientWidth;
    const vh = this.els.viewport.clientHeight;
    const startX = Math.floor((vw - w) / 2);
    const startY = Math.floor((vh - h) / 2);
    this.transform = { scale: 1.0, x: startX, y: startY };
    this.updateTransform();
  }

  updateTransform() {
    const { x, y, scale } = this.transform;
    this.els.canvasContainer.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    if (document.activeElement !== this.els.inpZoomLevel) {
      this.els.inpZoomLevel.value = Math.round(scale * 100);
    }
  }

  handleZoom(e) {
    if (!this.state.isImageLoaded) return;
    const ZOOM_FACTOR = 1.1;
    const direction = e.deltaY > 0 ? -1 : 1;
    const factor = direction > 0 ? ZOOM_FACTOR : (1 / ZOOM_FACTOR);
    let newScale = Math.max(0.01, Math.min(this.transform.scale * factor, 100.0));

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

  setZoomManual(newScale) {
    if (!this.state.isImageLoaded) return;
    newScale = Math.max(0.01, Math.min(newScale, 100.0));
    const oldScale = this.transform.scale;
    const scaleRatio = newScale / oldScale;
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

  bindEvents() {
    const { viewport } = this.els;

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(evt => {
      viewport.addEventListener(evt, e => { e.preventDefault(); viewport.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
      viewport.addEventListener(evt, e => { e.preventDefault(); viewport.classList.remove('drag-over'); });
    });
    viewport.addEventListener('drop', e => {
      const f = e.dataTransfer.files[0];
      if (f) this.handlers.onFileLoad(f);
    });

    // Paste
    window.addEventListener('paste', e => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          this.handlers.onFileLoad(item.getAsFile());
          break;
        }
      }
    });

    // Buttons
    this.els.btnSave.addEventListener('click', () => this.handlers.onSaveProject());
    this.els.btnLoad.addEventListener('click', () => document.getElementById('inpLoad').click());
    this.els.inpLoad.addEventListener('change', e => this.handlers.onLoadProject(e.target.files[0]));

    this.els.inpZoomLevel.addEventListener('change', e => {
      let p = parseFloat(e.target.value);
      if (isNaN(p) || p <= 0) p = 100;
      this.setZoomManual(p / 100.0);
    });

    this.els.btnRun.addEventListener('click', () => this.handlers.onRun());
    this.els.btnDownloadImg.addEventListener('click', () => this.handlers.onDownloadImage());
    this.els.btnDownloadMask.addEventListener('click', () => this.handlers.onDownloadMask());

    // 【設定パネルの表示切替】
    this.els.btnToggleParams.addEventListener('click', () => {
      // style.cssの.hidden定義が必要です
      this.els.panelParams.classList.toggle('hidden');
    });

    this.els.btnAutoMark.addEventListener('click', () => this.handlers.onAutoMark());

    this.els.toolRadios.forEach(r => {
      r.addEventListener('change', e => {
        if (e.target.checked) this.setToolMode(e.target.value);
      });
    });

    this.els.chkShowMarker.addEventListener('change', () => this.handlers.onToggleMarker());
    this.els.chkTransparent.addEventListener('change', () => this.handlers.onToggleTransparent());

    this.els.btnAddLabel.addEventListener('click', () => this.handlers.onAddLabel());
    this.els.btnDeleteLabel.addEventListener('click', () => this.handlers.onDeleteLabel());
    this.els.btnClear.addEventListener('click', () => {
      if (confirm("Clear all markers?")) {
        this.handlers.onClearMarkers();
        this.handlers.onRun();
      }
    });

    this.els.colorPicker.addEventListener('input', e => this.handlers.onColorChange(e.target.value));
    this.els.alphaInput.addEventListener('input', e => this.handlers.onAlphaChange(e.target.value));

    this.els.inputs.brush.addEventListener('input', e => {
      const s = parseInt(e.target.value);
      this.state.brushSize = s;
      this.els.dispBrush.textContent = s;
    });

    // Mouse
    viewport.addEventListener('wheel', e => {
      e.preventDefault();
      this.handleZoom(e);
      this.updateBrushGuide(e.clientX, e.clientY);
    }, { passive: false });

    viewport.addEventListener('mousedown', e => {
      e.preventDefault();
      const mode = this.state.toolMode;
      const isLeft = e.button === 0;
      if (e.button === 1 || e.button === 2 || (isLeft && mode === 'move')) {
        this.isPanning = true;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        viewport.style.cursor = 'grabbing';
        return;
      }
      if (isLeft && (mode === 'brush' || mode === 'eraser')) {
        this.drawing = true;
        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, mode === 'eraser');
      }
    });

    window.addEventListener('mousemove', e => {
      if (this.isPanning) {
        const dx = e.clientX - this.lastMousePos.x;
        const dy = e.clientY - this.lastMousePos.y;
        this.transform.x += dx;
        this.transform.y += dy;
        this.lastMousePos = { x: e.clientX, y: e.clientY };
        this.updateTransform();
        this.updateBrushGuide(e.clientX, e.clientY);
        return;
      }
      if (this.drawing) {
        const pos = this.getCanvasCoordinates(e);
        if (pos) this.handlers.onDraw(pos.x, pos.y, this.state.toolMode === 'eraser');
      }
      this.updateBrushGuide(e.clientX, e.clientY);
    });

    window.addEventListener('mouseup', e => {
      if (this.isPanning) {
        this.isPanning = false;
        this.updateCursor();
      } else if (this.drawing && e.button === 0) {
        this.drawing = false;
        this.handlers.onDrawEnd();
      }
    });

    viewport.addEventListener('contextmenu', e => e.preventDefault());
  }
}