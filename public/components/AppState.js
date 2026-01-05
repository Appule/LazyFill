export class AppState {
  constructor() {
    this.width = 0;
    this.height = 0;

    this.inputData = null;       // RGBA
    this.markerBuffer = null;    // Int32Array
    this.latestSegmentation = null;
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
    this.imagePath = null;
  }

  reset(width, height, inputData, path = null) {
    this.width = width;
    this.height = height;
    this.inputData = inputData;
    this.markerBuffer = new Int32Array(width * height).fill(0);
    this.latestSegmentation = null;
    this.isMarkerDirty = false;
    this.isImageLoaded = true;
    this.labelPixelCounts = {};
    Object.keys(this.labels).forEach(k => this.labelPixelCounts[k] = 0);
    this.imagePath = path;
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
    const current = this.labels[id];

    if (hex) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      this.labels[id] = { ...current, r, g, b, hex };
    }

    if (alpha !== undefined) {
      this.labels[id].a = parseFloat(alpha);
    }
  }

  getColor(id) {
    return this.labels[id] || this.labels[0];
  }
}