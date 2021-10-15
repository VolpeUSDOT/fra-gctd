const { ipcRenderer, shell } = require('electron');

window.selectFile = () => {
    return ipcRenderer.send('open-file-browser', null);
};

window.selectOutputDir = () => {
    return ipcRenderer.send('open-output-browser', null);
};

window.setCpuMode = (mode) => {
    return ipcRenderer.send('update-cpu-mode', mode);
};

window.setSkimMode = (active) => {
    return ipcRenderer.send('update-skim-mode', active);
};

window.sendFilePaths = (paths) => {
    arg = {};
    arg.paths = paths;
    return ipcRenderer.send('dragdrop-complete', arg);
};

window.openDirectory = (path) => {
    arg = {};
    arg.path = path;
    return ipcRenderer.send('open-dir', arg);
};

window.updateOutput = (path) => {
    arg = {};
    arg.path = path;
    return ipcRenderer.send('output-change', arg);
};

window.ipcRenderer = ipcRenderer;
window.shell = shell;
