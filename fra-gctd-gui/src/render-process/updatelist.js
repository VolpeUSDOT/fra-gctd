
var listVisible = false;

window.ipcRenderer.on('files-added', (event, arg) => {
    console.log("Files added to back-end queue", arg);
    showList();
    addToList(arg);
});

window.ipcRenderer.on('status-updated', (event, arg) => {
    console.log("File status updated", arg);
    var status = $("#" + arg.id + "Status");
    $("#" + arg.id + "Status").html(arg.status);
});

window.ipcRenderer.on('status-complete', (event, arg) => {
    console.log("Video complete", arg);
    var buttonCell = $("#" + arg.id + "Show");
    var button = $("<button id='" + arg.id + " button' href=''>View Result</button>").click(function() {
        window.openDirectory(arg.output);
    });
    buttonCell.append(button);
});

function showList() {
    if (!listVisible) {
        $("#videoList").show();
        $("#welcomeMessage").hide();
    }
}

function addToList(videos) {
    for (var video of videos) {
        var tableRow = "<tr>";
        tableRow += "<td>" + video.name + "</td>";
        tableRow += "<td id='" + video.id + "Status'>" + video.status + "</td>";
        tableRow += "<td id='" + video.id + "Show'></td>";
        tableRow += "</tr>";
        $("#videoTableBody").append(tableRow);
    }
}