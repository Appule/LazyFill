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
      contextIsolation: false // 今回の構成に合わせています
    }
  });

  mainWindow.loadFile('public/index.html');

  // --- メニューの定義 ---
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Project / Image',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile'],
              filters: [
                { name: 'Images & Projects', extensions: ['jpg', 'jpeg', 'png', 'webp', 'json'] }
              ]
            });

            if (!result.canceled && result.filePaths.length > 0) {
              // 選択されたファイルのパスをレンダラーに送る
              mainWindow.webContents.send('menu-open-file', result.filePaths[0]);
            }
          }
        },
        {
          label: 'Save Project',
          accelerator: 'CmdOrCtrl+S',
          click: () => mainWindow.webContents.send('menu-save-project')
        },
        { type: 'separator' },
        {
          label: 'Export Image',
          accelerator: 'CmdOrCtrl+E',
          click: () => mainWindow.webContents.send('menu-export-image')
        },
        {
          label: 'Export Mask',
          click: () => mainWindow.webContents.send('menu-export-mask')
        },
        { type: 'separator' },
        {
          label: 'Transparent Background',
          type: 'checkbox',
          checked: false, // 初期値（アプリ側の初期値と合わせてください）
          click: (menuItem) => {
            mainWindow.webContents.send('menu-toggle-transparent', menuItem.checked);
          }
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        {
          label: 'Clear Markers',
          click: () => mainWindow.webContents.send('menu-clear-markers')
        }
      ]
    },
    {
      label: 'Settings',
      submenu: [
        {
          label: 'Show Advanced Settings',
          type: 'checkbox',
          checked: false,
          click: (menuItem) => {
            mainWindow.webContents.send('menu-toggle-settings', menuItem.checked);
          }
        },
        { type: 'separator' },
        { role: 'toggleDevTools' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'togglefullscreen' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

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