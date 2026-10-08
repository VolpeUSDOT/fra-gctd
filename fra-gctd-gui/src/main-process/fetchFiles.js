const { dialog, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const parseVideo = require('./parseVideo');
const { isTrustedSender, resolveOutputDirectory } = require('./security');

const validExtensions = ['.mp4', '.avi'];
let videoId = 0;

function analyzeFiles(sender, paths) {
    if (!Array.isArray(paths) || paths.length === 0 || paths.length > 1000 ||
        paths.some(p => typeof p !== 'string' || p.includes('\0') || !path.isAbsolute(p)))
        throw new TypeError('Invalid file selection');
    const videoFiles = [];
    const visited = new Set();
    const pending = paths.slice();
    let entries = 0;
    while (pending.length) {
        if (++entries > 10000)
            throw new Error('Selection is too large; choose a smaller directory');
        const p = pending.pop();
        const stats = fs.lstatSync(p);
        if (stats.isSymbolicLink())
            continue;
        const realPath = fs.realpathSync(p);
        if (visited.has(realPath))
            continue;
        visited.add(realPath);
        if (stats.isDirectory()) {
            const children = fs.readdirSync(p);
            if (pending.length + children.length + entries > 10000)
                throw new Error('Selection is too large; choose a smaller directory');
            pending.push(...children.map(name => path.join(p, name)));
        } else if (stats.isFile() && validExtensions.includes(path.extname(p).toLowerCase())) {
            videoFiles.push(p);
        }
    }
    const videos = videoFiles.map(p => ({ path: p, id: videoId++, name: path.basename(p), status: 'Pending' }));
    // Publish rows before processing can emit status updates.
    sender.send('files-added', videos.map(({ id, name, status }) => ({ id, name, status })));
    for (const video of videos)
        parseVideo.addVideo(video.path, video.id, sender);
}

module.exports = function registerFileHandlers(handle, getWindow) {
    handle('select-files', async event => {
        const result = await dialog.showOpenDialog(getWindow(), {
            properties: ['openFile', 'multiSelections'],
            filters: [{ name: 'Videos', extensions: ['mp4', 'avi'] }]
        });
        if (!isTrustedSender(event, getWindow()))
            throw new Error('The requesting page is no longer trusted');
        if (!result.canceled && result.filePaths.length)
            analyzeFiles(event.sender, result.filePaths);
    });
    handle('submit-files', (event, paths) => analyzeFiles(event.sender, paths));
    handle('open-result', async (event, id) => {
        if (!Number.isSafeInteger(id) || id < 0)
            throw new TypeError('Invalid result ID');
        const result = parseVideo.getResult(id, event.sender);
        if (!result)
            throw new Error('Unknown or incomplete result');
        const target = resolveOutputDirectory(result.root, result.path);
        const error = await shell.openPath(target);
        if (error)
            throw new Error('Unable to open the result directory');
    });
};
