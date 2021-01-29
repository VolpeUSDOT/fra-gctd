$(document).ready(function() {
    const urlParams = new URLSearchParams(window.location.search);
    document.querySelector("#outputDir").value = urlParams.get("outputDir");
});

document.querySelector("#outputBrowseBtn").addEventListener('click', function(evt) {
    window.selectOutputDir();
});

window.ipcRenderer.on('output-dir-updated', function(event, arg) {
    document.querySelector("#outputDir").value = arg.outputPath;
});