// server.js (Electron Main Process)

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

// ウィンドウ状態を保存するファイルのパス
const statePath = path.join(app.getPath('userData'), 'window-state.json');

let mainWindow;

function createWindow() {
  // 1. デフォルトサイズの定義
  let winState = {
    width: 1200,
    height: 800,
    x: undefined,
    y: undefined
  };

  // 2. 保存された状態があれば読み込む
  try {
    if (fs.existsSync(statePath)) {
      const data = fs.readFileSync(statePath, 'utf8');
      const savedState = JSON.parse(data);
      // 読み込んだ値をマージ
      winState = { ...winState, ...savedState };
    }
  } catch (e) {
    console.error("Failed to load window state:", e);
  }

  // 3. ウィンドウ作成 (読み込んだ座標・サイズを適用)
  mainWindow = new BrowserWindow({
    width: winState.width,
    height: winState.height,
    x: winState.x,
    y: winState.y,
    webPreferences: {
      nodeIntegration: true, // 環境に合わせて調整してください
      contextIsolation: false
    }
  });

  // HTMLの読み込み
  mainWindow.loadFile('public/index.html'); // パスは環境に合わせて調整

  // 4. ウィンドウが閉じられる/リサイズ/移動されるタイミングで状態を保存
  const saveState = () => {
    if (!mainWindow) return;
    try {
      const bounds = mainWindow.getBounds(); // { x, y, width, height }
      fs.writeFileSync(statePath, JSON.stringify(bounds));
    } catch (e) {
      console.error("Failed to save window state:", e);
    }
  };

  // イベントリスナー登録 (頻繁な書き込みを防ぐため、実際はdebounce処理推奨ですが、簡易実装としてはこれで動作します)
  mainWindow.on('resize', saveState);
  mainWindow.on('move', saveState);
  mainWindow.on('close', saveState);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', createWindow);

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function () {
  if (mainWindow === null) createWindow();
});