const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const SettingsManager = require('./main-process/settingsManager');
const { isTrustedSender } = require('./main-process/security');
const registerFileHandlers = require('./main-process/fetchFiles');

if (require('electron-squirrel-startup'))
    app.quit();

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true
        }
    });
    mainWindow.webContents.on('will-navigate', event => event.preventDefault());
    mainWindow.webContents.on('will-redirect', event => event.preventDefault());
    mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    mainWindow.webContents.on('will-attach-webview', event => event.preventDefault());
    mainWindow.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    mainWindow.webContents.session.setPermissionCheckHandler(() => false);
    mainWindow.maximize();
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
    Menu.setApplicationMenu(null);
}

function handle(channel, handler) {
    ipcMain.handle(channel, async (event, ...args) => {
        if (!isTrustedSender(event, mainWindow))
            throw new Error('Untrusted IPC sender');
        return handler(event, ...args);
    });
}

app.whenReady().then(() => {
    global.settings = new SettingsManager();
    const getSettings = () => ({
        outputDir: global.settings.get('outputDir'),
        cpuMode: global.settings.get('cpuMode'),
        skimMode: global.settings.get('skimMode')
    });
    handle('get-settings', getSettings);
    handle('select-output-dir', async event => {
        const result = await dialog.showOpenDialog(mainWindow, { properties: ['openDirectory'] });
        if (!isTrustedSender(event, mainWindow))
            throw new Error('The requesting page is no longer trusted');
        if (!result.canceled && result.filePaths.length === 1)
            global.settings.set('outputDir', result.filePaths[0]);
        return getSettings();
    });
    handle('set-cpu-mode', (_event, mode) => {
        if (mode !== 'cpu' && mode !== 'gpu')
            throw new TypeError('Invalid CPU mode');
        global.settings.set('cpuMode', mode);
    });
    handle('set-skim-mode', (_event, active) => {
        if (typeof active !== 'boolean')
            throw new TypeError('Invalid skim setting');
        global.settings.set('skimMode', active);
    });
    handle('open-license', () => shell.openExternal('https://opensource.org/licenses/MIT'));
    registerFileHandlers(handle, () => mainWindow);
    createWindow();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0)
        createWindow();
});
