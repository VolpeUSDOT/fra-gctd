(function() {
    const outputDir = document.getElementById("outputDir");
    const browseButton = document.getElementById("outputBrowseBtn");
    const cpuModes = document.querySelectorAll('input[name="cpuMode"]');
    const skim = document.getElementById("skim");
    let currentSettings;

    function applySettings(settings) {
        if (!settings || typeof settings !== "object" || Array.isArray(settings) ||
            typeof settings.outputDir !== "string" ||
            (settings.cpuMode !== "cpu" && settings.cpuMode !== "gpu") ||
            typeof settings.skimMode !== "boolean") {
            throw new Error("Invalid settings response.");
        }
        currentSettings = {
            outputDir: settings.outputDir,
            cpuMode: settings.cpuMode,
            skimMode: settings.skimMode
        };
        outputDir.value = currentSettings.outputDir;
        for (const radio of cpuModes) {
            radio.checked = radio.value === currentSettings.cpuMode;
        }
        skim.checked = currentSettings.skimMode;
    }

    function disableControls(disabled) {
        browseButton.disabled = disabled;
        for (const radio of cpuModes) {
            radio.disabled = disabled;
        }
        skim.disabled = disabled;
    }

    browseButton.addEventListener("click", async () => {
        if (!currentSettings) {
            window.reportGctdError("Settings are not available yet.");
            return;
        }
        disableControls(true);
        try {
            applySettings(await window.gctd.selectOutputDir());
        } catch {
            window.reportGctdError("Unable to select the output directory.");
        } finally {
            disableControls(false);
        }
    });

    for (const radio of cpuModes) {
        radio.addEventListener("change", async (event) => {
            const mode = event.currentTarget.value;
            if (!currentSettings) {
                window.reportGctdError("Settings are not available yet.");
                return;
            }
            if (mode !== "cpu" && mode !== "gpu") {
                window.reportGctdError("Invalid processing mode.");
                return;
            }
            disableControls(true);
            try {
                await window.gctd.setCpuMode(mode);
                currentSettings.cpuMode = mode;
            } catch {
                applySettings(currentSettings);
                window.reportGctdError("Unable to update the processing mode.");
            } finally {
                disableControls(false);
            }
        });
    }

    skim.addEventListener("change", async (event) => {
        if (!currentSettings) {
            window.reportGctdError("Settings are not available yet.");
            return;
        }
        const enabled = event.currentTarget.checked;
        disableControls(true);
        try {
            await window.gctd.setSkimMode(enabled);
            currentSettings.skimMode = enabled;
        } catch {
            applySettings(currentSettings);
            window.reportGctdError("Unable to update video skimming.");
        } finally {
            disableControls(false);
        }
    });

    async function loadSettings() {
        disableControls(true);
        try {
            applySettings(await window.gctd.getSettings());
            disableControls(false);
        } catch {
            window.reportGctdError("Unable to load settings.");
        }
    }

    loadSettings();
})();
