const { app, BrowserWindow, Menu, dialog, ipcMain, shell} = require('electron');
const path = require('path');
const SettingsManager = require('./main-process/settingsManager');

// Settings should be a singleton, so we live dangerously and make it global
global.settings = new SettingsManager();

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) { // eslint-disable-line global-require
  app.quit();
}

const createWindow = () => {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  });
  mainWindow.maximize();
  // and load the index.html of the app.
  mainWindow.loadFile(path.join(__dirname, 'index.html'), 
                      {query: {
                        "outputDir": global.settings.get("outputDir"),
                        "cpuMode": global.settings.get("cpuMode"),
                        "skimMode": global.settings.get("skimMode")
                      }});
  Menu.setApplicationMenu(null);
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow);

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

ipcMain.on('open-output-browser', (event) => {
    return dialog.showOpenDialog({
      properties: ['openDirectory']
  }).then(files => {
      if (files !== undefined) {
          if (files.canceled)
              return;
          var outputDir = files.filePaths[0];
          global.settings.set("outputDir", outputDir);
          event.reply("output-dir-updated", {outputPath: outputDir});
      }
  });
});

ipcMain.on('update-cpu-mode', (event, mode) => {
  if (mode !== undefined)
    global.settings.set("cpuMode", mode);
});

ipcMain.on('update-skim-mode', (event, active) => {
  if (active !== undefined)
    global.settings.set("skimMode", active);
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.
require('./main-process/fetchFiles');