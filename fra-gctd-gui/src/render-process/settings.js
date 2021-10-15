$(document).ready(function() {
    const urlParams = new URLSearchParams(window.location.search);
    document.querySelector("#outputDir").value = urlParams.get("outputDir");
    var cpuMode = urlParams.get("cpuMode");
    if (!cpuMode)
        cpuMode = "cpu";
    document.getElementById(cpuMode).checked = true;
});

document.querySelector("#outputBrowseBtn").addEventListener('click', function(evt) {
    window.selectOutputDir();
});

$('input[type=radio][name=cpuMode]').change(function() {
    window.setCpuMode(this.value);
});

window.ipcRenderer.on('output-dir-updated', function(event, arg) {
    document.querySelector("#outputDir").value = arg.outputPath;
});