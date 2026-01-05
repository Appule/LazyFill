import { getImageDataFromFileInput, convertToGrayscale, downloadBufferAsImage } from './fileIO.js';
import { AppState } from './components/AppState.js';
import { AppView } from './components/AppView.js';
import { GraphCutService } from './components/GraphCutService.js';

export async function main() {
  const state = new AppState();

  // Electron関連の取得
  let ipcRenderer = null;
  let fs = null;
  let path = null;

  if (window.require) {
    try {
      const electron = window.require('electron');
      ipcRenderer = electron.ipcRenderer;
      fs = window.require('fs');      // ファイル読み込み用
      path = window.require('path');  // パス解析用
    } catch (e) {
      console.warn("Electron modules not found");
    }
  }

  const handlers = {
    onFileLoad: (file) => {
      if (!file) return;

      // --- パス取得ロジック ---
      let filePath = null;
      if (file.path) {
        filePath = file.path;
      }
      if (!filePath && window.require) {
        try {
          const { webUtils } = window.require('electron');
          filePath = webUtils.getPathForFile(file);
        } catch (e) {
          console.warn("Electron webUtils could not be loaded:", e);
        }
      }

      console.log("Detected File Path:", filePath);

      if (view.els.dropMessage) {
        view.els.dropMessage.style.display = 'none';
      }

      const img = new Image();
      img.onload = () => {
        view.resizeCanvases(img.width, img.height);
        getImageDataFromFileInput({ files: [file] }).then(res => {
          convertToGrayscale(res.data);
          state.reset(img.width, img.height, res.data, filePath);

          const ctx = view.els.ctx.input;
          const idata = ctx.createImageData(img.width, img.height);
          idata.data.set(res.data);
          ctx.putImageData(idata, 0, 0);

          view.updatePaletteUI();
          // 【削除】ボタン非活性化の呼び出しを削除
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

    onToggleMarker: () => view.updateLayerVisibility(),

    // 【修正】透過切替: DOMではなくViewの状態変数を更新
    onToggleTransparent: (forceValue = null) => {
      if (forceValue !== null) {
        view.isTransparent = forceValue;
      }
      if (state.latestSegmentation) view.drawResult(state.latestSegmentation);
    },

    onClearMarkers: () => {
      state.markerBuffer.fill(0);
      state.isMarkerDirty = true;
      state.labelPixelCounts = {};
      view.redrawMarkers();
    },

    onDraw: (cx, cy, isEraser) => {
      let r = state.brushSize;
      const labelId = isEraser ? 0 : state.currentLabelId;
      const { width, height, markerBuffer } = state;
      const ctx = view.els.ctx.marker;

      if (isEraser) {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'rgba(0,0,0,1)';
      } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = state.getColor(labelId).hex;
      }

      if (r === 1) {
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
          const idx = cy * width + cx;
          if (markerBuffer[idx] !== labelId) {
            state.updatePixelCount(markerBuffer[idx], -1);
            state.updatePixelCount(labelId, 1);
            markerBuffer[idx] = labelId;
            state.isMarkerDirty = true;
            ctx.fillRect(cx, cy, 1, 1);
          }
        }
      } else {
        r -= 1;
        const r2 = r * r;
        const minX = Math.max(0, cx - r);
        const maxX = Math.min(width - 1, cx + r);
        const minY = Math.max(0, cy - r);
        const maxY = Math.min(height - 1, cy + r);

        for (let y = minY; y <= maxY; y++) {
          for (let x = minX; x <= maxX; x++) {
            if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) {
              const idx = y * width + x;
              if (markerBuffer[idx] !== labelId) {
                state.updatePixelCount(markerBuffer[idx], -1);
                state.updatePixelCount(labelId, 1);
                markerBuffer[idx] = labelId;
                state.isMarkerDirty = true;
                ctx.fillRect(x, y, 1, 1);
              }
            }
          }
        }
      }
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
          state.latestSegmentation = resultMap;
          view.drawResult(resultMap);
          // 【削除】ボタン活性化の呼び出しを削除
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
      // 【修正】DOMではなくViewの状態変数を参照
      const isTransparent = view.isTransparent;

      downloadBufferAsImage(width, height, (data) => {
        for (let i = 0; i < width * height; i++) {
          const labelId = latestSegmentation[i];
          const lum = inputData[i * 4];
          if (labelId >= 2) {
            const c = state.getColor(labelId);
            data[i * 4] = Math.floor(lum * (c.r / 255));
            data[i * 4 + 1] = Math.floor(lum * (c.g / 255));
            data[i * 4 + 2] = Math.floor(lum * (c.b / 255));
            data[i * 4 + 3] = 255;
          } else {
            if (isTransparent) {
              data[i * 4] = 0; data[i * 4 + 1] = 0; data[i * 4 + 2] = 0;
              data[i * 4 + 3] = 255 - lum;
            } else {
              data[i * 4] = lum; data[i * 4 + 1] = lum; data[i * 4 + 2] = lum;
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
            data[i * 4] = c.r; data[i * 4 + 1] = c.g; data[i * 4 + 2] = c.b;
            data[i * 4 + 3] = Math.floor(c.a * 255);
          } else {
            data[i * 4] = 0; data[i * 4 + 1] = 0; data[i * 4 + 2] = 0; data[i * 4 + 3] = 0;
          }
        }
      }, 'result_mask.png');
    },

    onSaveProject: () => {
      if (!state.isImageLoaded) { alert("No image loaded."); return; }

      try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = state.width;
        tempCanvas.height = state.height;
        const ctx = tempCanvas.getContext('2d');
        const imgData = ctx.createImageData(state.width, state.height);
        const data = imgData.data;
        const buffer = state.markerBuffer;

        for (let i = 0; i < buffer.length; i++) {
          const id = buffer[i];
          data[i * 4] = id & 0xFF;
          data[i * 4 + 1] = 0; data[i * 4 + 2] = 0;
          data[i * 4 + 3] = id > 0 ? 255 : 0;
        }
        ctx.putImageData(imgData, 0, 0);
        const markerBase64 = tempCanvas.toDataURL('image/png');

        const projectData = {
          version: 1.0,
          width: state.width,
          height: state.height,
          imagePath: state.imagePath,
          params: view.getParameters(),
          labels: state.labels,
          markers: markerBase64
        };

        const blob = new Blob([JSON.stringify(projectData)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "lazyfill_project.json";
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error(e);
        alert("Save Project Error: " + e.message);
      }
    },

    onLoadProject: (file) => {
      if (!file) return;
      const reader = new FileReader();

      reader.onload = async (e) => {
        try {
          const json = JSON.parse(e.target.result);

          const applyProjectData = () => {
            if (json.params) {
              const p = json.params;
              const setVal = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined) el.value = val; };
              const setCheck = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined) el.checked = val; };
              setVal('inpPadding', p.padding);
              setVal('inpSigma', p.sigma);
              setVal('inpMaxIter', p.maxIter);
              setVal('inpBfsNum', p.bfsNum);
              setVal('inpStrength', p.strength);
              setVal('inpBB', p.bbThreshold);
              setCheck('chkDynamic', p.isDynamic);
            }
            if (json.labels) { state.labels = json.labels; view.updatePaletteUI(); }

            const currentW = state.width;
            const currentH = state.height;

            if (json.width !== currentW || json.height !== currentH) {
              alert(`画像サイズ不一致: 現${currentW}x${currentH} / 保${json.width}x${json.height}`);
            } else if (json.markers) {
              const imgMarkers = new Image();
              imgMarkers.onload = () => {
                const tCanvas = document.createElement('canvas');
                tCanvas.width = currentW; tCanvas.height = currentH;
                const tCtx = tCanvas.getContext('2d');
                tCtx.drawImage(imgMarkers, 0, 0);
                const tData = tCtx.getImageData(0, 0, currentW, currentH).data;

                state.markerBuffer.fill(0);
                state.labelPixelCounts = {};
                for (let i = 0; i < state.markerBuffer.length; i++) {
                  const id = tData[i * 4];
                  if (id > 0) { state.markerBuffer[i] = id; state.updatePixelCount(id, 1); }
                }
                state.isMarkerDirty = true;
                view.redrawMarkers();
                handlers.onRun();
                alert("読み込み完了");
              };
              imgMarkers.src = json.markers;
            }
          };

          const loadImage = (src) => {
            return new Promise((resolve, reject) => {
              const img = new Image();
              img.onload = () => resolve(img);
              img.onerror = () => reject(new Error(`Load failed: ${src}`));
              img.src = src;
            });
          };

          if (json.imagePath) {
            try {
              const img = await loadImage(json.imagePath);
              view.resizeCanvases(img.width, img.height);

              const cvs = document.createElement('canvas');
              cvs.width = img.width; cvs.height = img.height;
              const c = cvs.getContext('2d');
              c.drawImage(img, 0, 0);
              const rgba = c.getImageData(0, 0, img.width, img.height).data;
              convertToGrayscale(rgba);

              state.reset(img.width, img.height, rgba, json.imagePath);

              const vCtx = view.els.ctx.input;
              const vData = vCtx.createImageData(img.width, img.height);
              vData.data.set(rgba);
              vCtx.putImageData(vData, 0, 0);

              view.updateLayerVisibility();
              applyProjectData();

            } catch (err) {
              alert(`画像が見つかりません: ${json.imagePath}\n手動で開いてください。`);
            }
          } else {
            if (!state.isImageLoaded) {
              alert("プロジェクトに画像パスがなく、現在画像もありません。");
              return;
            }
            applyProjectData();
          }

        } catch (err) {
          console.error(err);
          alert("読込エラー: " + err.message);
        }
      };
      reader.readAsText(file);
    },

    onMenuToggleSettings: (isVisible) => {
      if (isVisible) {
        view.els.panelParams.classList.remove('hidden');
      } else {
        view.els.panelParams.classList.add('hidden');
      }
    }
  };

  const view = new AppView(state, handlers);

  if (ipcRenderer) {
    ipcRenderer.on('menu-open-file', (event, filePath) => {
      if (!fs || !path) return;

      try {
        const buffer = fs.readFileSync(filePath);
        const fileName = path.basename(filePath);
        const mimeType = fileName.endsWith('.json') ? 'application/json' : 'image/png';

        const file = new File([buffer], fileName, { type: mimeType });
        Object.defineProperty(file, 'path', { value: filePath });

        if (fileName.toLowerCase().endsWith('.json')) {
          handlers.onLoadProject(file);
        } else {
          handlers.onFileLoad(file);
        }

      } catch (err) {
        console.error("File open error:", err);
        alert("ファイルを開けませんでした: " + err.message);
      }
    });

    ipcRenderer.on('menu-save-project', () => handlers.onSaveProject());
    ipcRenderer.on('menu-export-image', () => handlers.onDownloadImage());
    ipcRenderer.on('menu-export-mask', () => handlers.onDownloadMask());
    ipcRenderer.on('menu-toggle-transparent', (event, isChecked) => {
      handlers.onToggleTransparent(isChecked);
    });
    ipcRenderer.on('menu-clear-markers', () => {
      if (confirm("Clear all markers?")) {
        handlers.onClearMarkers();
        handlers.onRun();
      }
    });
    ipcRenderer.on('menu-toggle-settings', (event, isChecked) => {
      handlers.onMenuToggleSettings(isChecked);
    });
  }
}

document.addEventListener('DOMContentLoaded', main);