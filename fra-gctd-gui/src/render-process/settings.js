$(document).ready(function() {
    const urlParams = new URLSearchParams(window.location.search);
    document.querySelector("#outputDir").value = urlParams.get("outputDir");
    var cpuMode = urlParams.get("cpuMode");
    if (!cpuMode)
        cpuMode = "cpu";
    document.getElementById(cpuMode).checked = true;
    var skimMode = urlParams.get("skimMode");
    if (skimMode === undefined)
        skimMode = true;
    document.getElementById("skim").checked = skimMode == "true";
});

document.querySelector("#outputBrowseBtn").addEventListener('click', function(evt) {
    window.selectOutputDir();
});

$('input[type=radio][name=cpuMode]').change(() => {
    window.setCpuMode(this.value);
});

$("#skim").change(() => {
    window.setSkimMode($("#skim").prop("checked"));
});

window.ipcRenderer.on('output-dir-updated', function(event, arg) {
    document.querySelector("#outputDir").value = arg.outputPath;
});