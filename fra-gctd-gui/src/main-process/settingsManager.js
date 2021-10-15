const {app} = require("electron");
const fs = require("fs");
const path = require("path");

class SettingsManager {
    constructor(opts) {
        const dataDir = app.getPath('appData');
        var directory = path.join(dataDir, "fragctd");
        if (!fs.existsSync(directory)){
            fs.mkdirSync(directory);
        }
        this.path = path.join(directory, "settings.json");
        try {
            this.data = JSON.parse(fs.readFileSync(this.path));
        } catch(error) {
            // Set to defaults
            this.data = {
                "outputDir": path.join(app.getPath("documents"), "gctdOutput"),
                "cpuMode": "cpu",
                "skimMode": true
            };
        }
    }

    get(settingName) {
        return this.data[settingName];
    }
    
    set(settingName, value) {
        this.data[settingName] = value;
        fs.writeFileSync(this.path, JSON.stringify(this.data));
    }
}


module.exports = SettingsManager;

