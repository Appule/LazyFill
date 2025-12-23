const { app, BrowserWindow } = require('electron');
const express = require('express');
const path = require('path');

let mainWindow;
let server;

function startServerAndWindow() {
  // 1. Expressサーバーを内部で起動
  const expressApp = express();
  // publicフォルダを静的ファイルとして配信
  expressApp.use(express.static(path.join(__dirname, 'public')));

  // ポート0を指定すると、OSが空いているポートを自動で割り当ててくれます
  server = expressApp.listen(0, () => {
    const port = server.address().port;
    console.log(`Internal Server running on port ${port}`);

    // 2. サーバーが立ち上がったら、Electronのウィンドウを作成して読み込む
    createWindow(port);
  });
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    title: "LazyFill", // アプリのタイトル
    autoHideMenuBar: true,  // 上部のメニューバーを隠す
    webPreferences: {
      nodeIntegration: false, // セキュリティのためfalse推奨
    }
  });

  // 起動したローカルサーバーのURLを読み込む
  mainWindow.loadURL(`http://localhost:${port}`);

  // ウィンドウが閉じられたらメモリを解放
  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

// Electronの準備ができたらスタート
app.on('ready', startServerAndWindow);

// 全てのウィンドウが閉じられたらアプリを終了（ここが重要）
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
    // サーバーも明示的に閉じる（念のため）
    if (server) server.close();
  }
});