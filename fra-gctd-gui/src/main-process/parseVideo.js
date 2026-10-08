const { app } = require('electron');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const { createOutputDirectory } = require('./security');

const resourceDir = app.isPackaged ? process.resourcesPath : path.join(app.getAppPath(), 'resources');
const processDir = path.join(resourceDir, 'executables', 'process_video');
const videosToProcess = [];
const results = new Map();
let videoInProcess = null;

function send(sender, channel, data) {
    if (!sender.isDestroyed())
        sender.send(channel, data);
}

function startNext() {
    while ((videoInProcess = videosToProcess.shift() || null)) {
        if (startProcess(videoInProcess))
            return;
    }
}

function startProcess(video) {
    let output;
    let writeStream;
    let child;
    let failed = false;
    let closed = false;
    let flushed = false;
    let finalized = false;
    let exitCode;
    let runtime;
    function finalize() {
        if (finalized || !closed || (!flushed && !failed))
            return;
        finalized = true;
        if (exitCode === 0 && !failed) {
            results.set(video.id, { sender: video.sender, root: output.root, path: output.path });
            send(video.sender, 'status-updated', {
                id: video.id, status: runtime ? 'Completed in ' + runtime : 'Complete'
            });
            send(video.sender, 'status-complete', { id: video.id });
        } else if (!failed) {
            send(video.sender, 'status-updated', { id: video.id, status: 'Process stopped unexpectedly' });
        }
        startNext();
    }
    try {
        const name = path.basename(video.path, path.extname(video.path));
        output = createOutputDirectory(global.settings.get('outputDir'), name);
        const logfile = path.join(output.path, 'process-log-' + video.id + '-' + Date.now() + '.txt');
        writeStream = fs.createWriteStream(logfile, { flags: 'wx' });
        writeStream.on('error', () => {
            if (finalized)
                return;
            failed = true;
            if (child)
                child.kill();
            send(video.sender, 'status-updated', { id: video.id, status: 'Unable to write the processing log' });
            finalize();
        });
        writeStream.on('finish', () => { flushed = true; finalize(); });
        const args = ['--inputpath', video.path, '--outputpath', output.path];
        if (global.settings.get('cpuMode') !== 'gpu')
            args.push('--cpu');
        if (global.settings.get('skimMode'))
            args.push('--skim');
        child = execFile(path.join(processDir, 'process_video.exe'), args, { cwd: processDir });
    } catch (error) {
        finalized = true;
        if (writeStream)
            writeStream.end();
        send(video.sender, 'status-updated', { id: video.id, status: 'Unable to start processing: ' + error.message });
        return false;
    }

    send(video.sender, 'status-updated', { id: video.id, status: 'Processing' });
    let buffer = '';
    child.on('error', () => {
        failed = true;
        send(video.sender, 'status-updated', { id: video.id, status: 'Unable to start the video processing executable' });
    });
    child.stdout.on('data', data => {
        writeStream.write(data);
        buffer += data.toString();
        const lines = buffer.split(/[\r\n]/);
        buffer = lines.pop().slice(-4096);
        for (const line of lines) {
            const progress = line.match(/Processing:.*% complete/);
            const completed = line.match(/Total running time: (.*)/);
            if (progress)
                send(video.sender, 'status-updated', { id: video.id, status: progress[0] });
            if (completed)
                runtime = completed[1];
        }
    });
    child.stderr.on('data', data => writeStream.write(data));
    child.on('close', code => {
        closed = true;
        exitCode = code;
        writeStream.end();
        finalize();
    });
    return true;
}

exports.addVideo = (videoPath, id, sender) => {
    if ((videoInProcess && videoInProcess.path === videoPath) ||
        videosToProcess.some(video => video.path === videoPath)) {
        send(sender, 'status-updated', { id, status: 'Already queued' });
        return;
    }
    videosToProcess.push({ path: videoPath, id, sender });
    if (!videoInProcess)
        startNext();
};

exports.getResult = (id, sender) => {
    const result = results.get(id);
    return result && result.sender === sender ? result : undefined;
};
