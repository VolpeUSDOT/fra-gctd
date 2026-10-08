(function() {
    const dropZone = document.getElementById("dropZone");

    dropZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
            if (!event.dataTransfer || !event.dataTransfer.files) {
                window.reportGctdError("Invalid file drop.");
                return;
            }
            const files = Array.from(event.dataTransfer.files);
            if (files.length === 0 || files.some((file) => !(file instanceof File))) {
                window.reportGctdError("Drop a video file or folder to add it.");
                return;
            }
            await window.gctd.submitFiles(files);
        } catch {
            window.reportGctdError("Unable to add the dropped files.");
        }
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.stopPropagation();
    });

    document.getElementById("fileBrowser").addEventListener("click", async (event) => {
        event.preventDefault();
        try {
            await window.gctd.selectFiles();
        } catch {
            window.reportGctdError("Unable to select video files.");
        }
    });

    document.getElementById("licenseLink").addEventListener("click", async (event) => {
        event.preventDefault();
        try {
            await window.gctd.openLicense();
        } catch {
            window.reportGctdError("Unable to open the MIT License.");
        }
    });
})();

