const { dialog, ipcMain, shell} = require('electron');
const fs = require('fs');
const path = require('path');
const parseVideo = require('./parseVideo');
const { parse } = require('path');

// Including .txt for testing, remove later
const validExtensions = ['.mp4', '.avi'];

ipcMain.on('open-file-browser', (event, arg) => {
    return dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections']
    }).then(files => {
        if (files !== undefined) {
            if (files.canceled)
                return;
            analyzeFiles(event, files.filePaths);
        }
    });
});

ipcMain.on('dragdrop-complete', (event, arg) => {
    analyzeFiles(event, arg.paths);
});

ipcMain.on('open-dir', (event, arg) => {
    shell.openPath(arg.path);
});

var videoId = 0;

function analyzeFiles(event, pathList) {
    var validPaths = findValidFiles(pathList);
    var videoData = [];
    for (var p of validPaths) {
        var id = videoId++;
        parseVideo.addVideo(p, id, event.sender);
        // Will we send all at once? Do 2-3 at a time? Measure performance?
        // Need to ensure we send this back to the UI before we try to update it
        var data = {
            "name": path.basename(p),
            "id": id,
            "status": "Pending"
        };
        videoData.push(data);
    }
    event.reply('files-added', videoData);
}

function findValidFiles(paths) {
    var videoFiles = [];
    for (var p of paths) {
        var stats = fs.lstatSync(p);
        if (stats.isDirectory()) {
            var dirContents = fs.readdirSync(p);
            dirContents = dirContents.map((value) => {
                return path.join(p, value);
            });
            videoFiles = videoFiles.concat(findValidFiles(dirContents));
        }
        else if (stats.isFile() && validExtensions.includes(path.extname(p).toLocaleLowerCase())) {
            videoFiles.push(p);
        }
    }
    return videoFiles;
}
