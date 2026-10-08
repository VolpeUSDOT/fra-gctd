const { contextBridge, ipcRenderer, webUtils } = require('electron');

function subscribe(channel, callback) {
    if (typeof callback !== 'function')
        throw new TypeError('An event callback is required');
    // Electron event objects expose IPC capabilities; only forward the data.
    const listener = (_event, data) => callback(data);
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld('gctd', {
    selectFiles: () => ipcRenderer.invoke('select-files'),
    selectOutputDir: () => ipcRenderer.invoke('select-output-dir'),
    getSettings: () => ipcRenderer.invoke('get-settings'),
    setCpuMode: mode => ipcRenderer.invoke('set-cpu-mode', mode),
    setSkimMode: active => ipcRenderer.invoke('set-skim-mode', active),
    submitFiles: files => {
        if (!Array.isArray(files) || files.length === 0 || files.length > 1000)
            return Promise.reject(new TypeError('Select between 1 and 1000 files'));
        try {
            const paths = files.map(file => webUtils.getPathForFile(file));
            if (paths.some(filePath => !filePath))
                throw new TypeError('Only files from disk can be submitted');
            return ipcRenderer.invoke('submit-files', paths);
        } catch (error) {
            return Promise.reject(error);
        }
    },
    openResult: id => ipcRenderer.invoke('open-result', id),
    openLicense: () => ipcRenderer.invoke('open-license'),
    onFilesAdded: callback => subscribe('files-added', callback),
    onStatusUpdated: callback => subscribe('status-updated', callback),
    onStatusComplete: callback => subscribe('status-complete', callback)
});
