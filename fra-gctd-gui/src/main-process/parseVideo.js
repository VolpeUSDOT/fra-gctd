const { dialog, ipcMain, app} = require('electron');
const execFile = require("child_process").execFile;
const path = require('path');
const fs = require('fs');

var videosToProcess = [];
var videosProcessed = [];

var videoInProcess = null;

const processDir = app.getAppPath() + "\\resources\\executables\\process_video";

exports.addVideo = addVideo;

function addVideo(videoPath, videoId, sender) {
    for (var video of videosToProcess) {
        if (video.name === videoPath)
            // The video is already in the queue; do nothing.
            return;
    }

    var newVideo = {
        name: path.basename(videoPath),
        path: videoPath,
        id: videoId,
        sender: sender,
        status: "Pending"
    };

    if (videoInProcess == null) {
        startProcess(newVideo);
    } else {
        videosToProcess.push(newVideo);
    }
}

function startProcess(video) {
    video.status = "Processing";
    videoInProcess = video;
    console.log("Processing video " + video.name);
    var fileName = path.basename(video.path, path.extname(video.path));

    var outputPath = path.join(global.settings.get("outputDir"), fileName);
    console.log("OutputPath", outputPath);

    var cpuMode = global.settings.get("cpuMode");

    if (!fs.existsSync(outputPath)){
        fs.mkdirSync(outputPath);
    }
    var logfile = path.join(outputPath, "process-log.txt");
    let writeStream = fs.createWriteStream(logfile);

    var execArgs = ['--inputpath', video.path, '--outputpath', outputPath];
    if (cpuMode !== "gpu")
        execArgs.push("--cpu");
    var child = execFile("process_video.exe", 
                        execArgs, 
                        {cwd : processDir});

    child.stdout.on('data', function(data) {
        // Grab output from the script, and push completion percentage to the renderer
        writeStream.write(data);
        if (data.match("Processing:.*\% complete"))
            updateStatus(video.id, data.match("Processing:.*\% complete")[0], video.sender);
        else if (data.match("Total running time: .*")) {
            var msg = data.match("Total running time: .*")[0];
            var runTime = msg.split("Total running time: ")[1];
            updateStatus(video.id, "Completed in " + runTime, video.sender);
            notifyComplete(video.id, outputPath, video.sender);
        }
    });

    child.stderr.on('data', function(data) {
        writeStream.write(data);
    });

    // When the process is complete, start the next one
    child.on('close', function() {
        writeStream.close();
        console.log("Closed video " + video.name);
        video.status = "Complete";
        videosProcessed.push(video);
        var nextVideo = videosToProcess.shift();
        // If we're out of videos, do nothing
        if (nextVideo == null)
            videoInProcess = null;
        else {
            videoInProcess = nextVideo;
            startProcess(nextVideo);
        }
    });

    child.on('exit', function(code) {
        writeStream.close();
        if (code != 0)
            updateStatus(video.id, "Process stopped unexpectedly", video.sender);
    });
}

function updateStatus(id, status, sender) {
    var args = {
        id: id,
        status: status
    };
    sender.send("status-updated", args);
}

function notifyComplete(id, outputDir, sender) {
    var args = {
        id: id,
        output: outputDir
    };
    sender.send("status-complete", args);
}