(function() {
    const errorMessage = document.getElementById("errorMessage");
    const tableBody = document.getElementById("videoTableBody");
    const rows = new Map();
    const unsubscribers = [];

    window.reportGctdError = function(message) {
        errorMessage.textContent = message;
        errorMessage.hidden = false;
    };

    function validRecord(record) {
        return record !== null && typeof record === "object" && !Array.isArray(record) &&
            Number.isSafeInteger(record.id) && record.id >= 0;
    }

    function filesAdded(videos) {
        if (!Array.isArray(videos)) {
            window.reportGctdError("Received invalid video data.");
            return;
        }
        const ids = new Set();
        for (const video of videos) {
            if (!validRecord(video) || typeof video.name !== "string" ||
                typeof video.status !== "string" || ids.has(video.id) || rows.has(video.id)) {
                window.reportGctdError("Received invalid video data.");
                return;
            }
            ids.add(video.id);
        }
        for (const video of videos) {
            const row = document.createElement("tr");
            const nameCell = document.createElement("td");
            const statusCell = document.createElement("td");
            const buttonCell = document.createElement("td");
            nameCell.textContent = video.name;
            statusCell.textContent = video.status;
            row.append(nameCell, statusCell, buttonCell);
            tableBody.appendChild(row);
            rows.set(video.id, { statusCell, buttonCell });
        }
        if (videos.length > 0) {
            document.getElementById("videoList").hidden = false;
            document.getElementById("welcomeMessage").hidden = true;
        }
    }

    function statusUpdated(update) {
        if (!validRecord(update) || typeof update.status !== "string" || !rows.has(update.id)) {
            window.reportGctdError("Received an invalid video status update.");
            return;
        }
        rows.get(update.id).statusCell.textContent = update.status;
    }

    function statusComplete(update) {
        if (!validRecord(update) || !rows.has(update.id)) {
            window.reportGctdError("Received an invalid video completion update.");
            return;
        }
        const buttonCell = rows.get(update.id).buttonCell;
        if (buttonCell.firstChild) {
            return;
        }
        const id = update.id;
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "View Result";
        button.addEventListener("click", async () => {
            try {
                await window.gctd.openResult(id);
            } catch {
                window.reportGctdError("Unable to open the video result.");
            }
        });
        buttonCell.appendChild(button);
    }

    window.addEventListener("unload", () => {
        for (const unsubscribe of unsubscribers) {
            try {
                unsubscribe();
            } catch {
                window.reportGctdError("Unable to remove a queue listener.");
            }
        }
    });

    for (const [method, callback] of [
        ["onFilesAdded", filesAdded],
        ["onStatusUpdated", statusUpdated],
        ["onStatusComplete", statusComplete]
    ]) {
        try {
            const unsubscribe = window.gctd[method](callback);
            if (typeof unsubscribe !== "function") {
                throw new Error("Invalid queue subscription.");
            }
            unsubscribers.push(unsubscribe);
        } catch {
            window.reportGctdError("Unable to listen for video queue updates.");
        }
    }
})();
