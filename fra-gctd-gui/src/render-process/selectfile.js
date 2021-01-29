const dropZone = document.getElementById("dropZone");

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log("Dropped");
    var paths = [];
    for (const f of event.dataTransfer.files) {
        console.log('File Path of dragged files: ', f.path);
        paths.push(f.path);
    }
    window.sendFilePaths(paths);
});

dropZone.addEventListener('dragover', (e) => { 
    e.preventDefault(); 
    e.stopPropagation(); 
}); 

document.querySelector('#fileBrowser').addEventListener('click', function (event) {
    window.selectFile();
});

