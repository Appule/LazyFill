const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');

const statePath = path.join(app.getPath('userData'), 'window-state.json');

let mainWindow;

function createWindow() {
  let winState = { width: 1200, height: 800 };

  try {
    if (fs.existsSync(statePath)) {
      const data = fs.readFileSync(statePath, 'utf8');
      winState = { ...winState, ...JSON.parse(data) };
    }
  } catch (e) {
    console.error("Failed to load window state:", e);
  }

  mainWindow = new BrowserWindow({
    width: winState.width,
    height: winState.height,
    x: winState.x,
    y: winState.y,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('public/index.html');

  // --- メニューの定義 (日本語版) ---
  const isMac = process.platform === 'darwin';

  const template = [
    // macOS用: アプリ名メニュー (About, Quit等)
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about', label: `${app.name}について` },
        { type: 'separator' },
        { role: 'services', label: 'サービス' },
        { type: 'separator' },
        { role: 'hide', label: `${app.name}を隠す` },
        { role: 'hideOthers', label: 'ほかを隠す' },
        { role: 'unhide', label: 'すべて表示' },
        { type: 'separator' },
        { role: 'quit', label: `${app.name}を終了` }
      ]
    }] : []),

    // File -> ファイル
    {
      label: 'ファイル',
      submenu: [
        {
          label: 'プロジェクト/画像を開く',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile'],
              filters: [
                { name: 'Images & Projects', extensions: ['jpg', 'jpeg', 'png', 'webp', 'json'] }
              ]
            });

            if (!result.canceled && result.filePaths.length > 0) {
              mainWindow.webContents.send('menu-open-file', result.filePaths[0]);
            }
          }
        },
        {
          label: 'プロジェクトを保存',
          accelerator: 'CmdOrCtrl+S',
          click: () => mainWindow.webContents.send('menu-save-project')
        },
        { type: 'separator' },
        {
          label: '画像をエクスポート',
          accelerator: 'CmdOrCtrl+E',
          click: () => mainWindow.webContents.send('menu-export-image')
        },
        {
          label: 'マスクをエクスポート',
          click: () => mainWindow.webContents.send('menu-export-mask')
        },
        { type: 'separator' },
        {
          label: '背景透過',
          type: 'checkbox',
          checked: false,
          click: (menuItem) => {
            mainWindow.webContents.send('menu-toggle-transparent', menuItem.checked);
          }
        },
        // Windows/Linux用: 終了ボタン
        ...(isMac ? [] : [
          { type: 'separator' },
          { role: 'quit', label: '終了' }
        ])
      ]
    },

    // Edit -> 編集
    {
      label: '編集',
      submenu: [
        // Undo/Redo/Copy/Paste等を削除し、カスタム機能のみ配置
        {
          label: 'マーカーをクリア',
          click: () => mainWindow.webContents.send('menu-clear-markers')
        }
      ]
    },

    // Settings -> 設定
    {
      label: '設定',
      submenu: [
        {
          label: '詳細設定を表示',
          type: 'checkbox',
          id: 'menu-show-settings', // 【追加】IDを付与して後で参照できるようにする
          checked: false,
          click: (menuItem) => {
            mainWindow.webContents.send('menu-toggle-settings', menuItem.checked);
          }
        },
        { type: 'separator' },
        { role: 'toggleDevTools', label: '開発者ツール' }
      ]
    },

    // View -> 表示
    {
      label: '表示',
      submenu: [
        { role: 'reload', label: '再読み込み' },
        { role: 'forceReload', label: '強制再読み込み' },
        { role: 'togglefullscreen', label: '全画面表示' },
        { type: 'separator' },
        // 【修正】標準のrole: 'resetZoom'ではなく、カスタムIPCを送る
        {
          label: 'ズームリセット (100%)',
          accelerator: 'CmdOrCtrl+0',
          click: () => mainWindow.webContents.send('menu-reset-zoom')
        }
        // 拡大・縮小メニューは削除しました
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  // 【追加】レンダラーからの同期リクエストを受信してメニューを更新
  ipcMain.on('sync-settings-menu', (event, isVisible) => {
    const item = Menu.getApplicationMenu().getMenuItemById('menu-show-settings');
    if (item) {
      item.checked = isVisible;
    }
  });

  // --- ウィンドウ状態保存 ---
  const saveState = () => {
    if (!mainWindow) return;
    try {
      const bounds = mainWindow.getBounds();
      fs.writeFileSync(statePath, JSON.stringify(bounds));
    } catch (e) {
      console.error("Failed to save window state:", e);
    }
  };

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