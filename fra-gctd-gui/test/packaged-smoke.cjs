const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { _electron: electron } = require('playwright-core');

test('packaged Windows GUI loads bundled assets with isolation and sandboxing', { timeout: 60000 }, async () => {
    const executable = path.resolve(__dirname, '../out/fra-gctd-gui-win32-x64/fra-gctd-gui.exe');
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'fra-gctd-package-'));
    let application;
    try {
        const env = { ...process.env, APPDATA: temporary };
        delete env.ELECTRON_RUN_AS_NODE;
        application = await electron.launch({ executablePath: executable,
            args: ['--user-data-dir=' + temporary], env });
        const page = await application.firstWindow();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.waitForFunction(() => window.gctd && !document.getElementById('outputBrowseBtn').disabled);
        assert.equal(await application.evaluate(({ app }) => app.isPackaged), true);
        assert.equal(await application.evaluate(({ app }) => app.getPath('userData')), temporary);
        assert.deepEqual(await page.evaluate(() => [typeof window.ipcRenderer, typeof window.shell]), ['undefined', 'undefined']);
        assert.deepEqual(await application.evaluate(({ BrowserWindow }) => {
            const prefs = BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences();
            return [prefs.contextIsolation, prefs.nodeIntegration, prefs.sandbox];
        }), [true, false, true]);
        assert.deepEqual(await page.evaluate(() => Array.from(document.styleSheets).map(sheet => sheet.cssRules.length > 0)), [true, true]);
        assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('h1')).marginTop), '0px');
        assert.equal(await page.evaluate(() => document.querySelector('.headerImage').naturalWidth > 0), true);
        assert.deepEqual(errors, []);
    } finally {
        if (application)
            await application.close();
        fs.rmSync(temporary, { recursive: true, force: true, maxRetries: 5 });
    }
});
