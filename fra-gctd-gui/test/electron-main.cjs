const { app } = require('electron');
app.setPath('appData', process.env.FRA_GCTD_TEST_APPDATA);
app.setPath('userData', process.env.FRA_GCTD_TEST_APPDATA);
global.testProcessor = require('../src/main-process/parseVideo');
global.testPath = require('node:path');
require('../src/index.js');
