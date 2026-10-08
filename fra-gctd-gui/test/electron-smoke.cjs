const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { _electron: electron } = require('playwright-core');

test('real Electron isolates the renderer and preserves safe GUI workflows', { timeout: 90000 }, async () => {
    const root = path.resolve(__dirname, '..');
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'fra-gctd-electron-'));
    const filePath = path.join(temporary, "O'Brien & sons.mp4");
    fs.writeFileSync(filePath, 'mock video');
    let application;
    try {
        const env = { ...process.env, FRA_GCTD_TEST_APPDATA: temporary };
        delete env.ELECTRON_RUN_AS_NODE;
        application = await electron.launch({
            executablePath: require('electron'),
            args: [path.join(__dirname, 'electron-main.cjs')],
            env
        });
        const page = await application.firstWindow();
        await page.waitForFunction(() => window.gctd && document.getElementById('cpu').checked);
        const preferences = await application.evaluate(({ BrowserWindow }) => {
            const settings = BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences();
            return { contextIsolation: settings.contextIsolation, nodeIntegration: settings.nodeIntegration,
                sandbox: settings.sandbox, electron: process.versions.electron };
        });
        assert.deepEqual(preferences, {
            contextIsolation: true, nodeIntegration: false, sandbox: true, electron: '44.7.0'
        });
        assert.deepEqual(await page.evaluate(() => [typeof window.ipcRenderer, typeof window.shell,
            typeof window.require, typeof window.gctd.send, typeof window.gctd.on]),
        ['undefined', 'undefined', 'undefined', 'undefined', 'undefined']);

        // Mock OS opening and inference only; run the real preload, IPC handlers and renderer.
        await application.evaluate(({ dialog, shell }, { root, temporary }) => {
            global.testOpenPaths = [];
            global.testExternalURLs = [];
            shell.openPath = async value => { global.testOpenPaths.push(value); return ''; };
            shell.openExternal = async value => { global.testExternalURLs.push(value); };
            dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [temporary] });
            const processor = global.testProcessor;
            processor.addVideo = () => {};
            processor.getResult = (id, sender) => id === 0 ? { sender, root: temporary,
                path: global.testPath.join(temporary, 'result') } : undefined;
        }, { root, temporary });
        fs.mkdirSync(path.join(temporary, 'result'));
        await page.locator('#gpu').check();
        await page.locator('#skim').uncheck();
        await page.waitForFunction(async () => {
            const settings = await window.gctd.getSettings();
            return settings.cpuMode === 'gpu' && settings.skimMode === false;
        });
        await page.locator('#outputBrowseBtn').click();
        await page.waitForFunction(value => document.getElementById('outputDir').value === value, temporary);
        assert.equal(await page.evaluate(async () => {
            try { await window.gctd.setCpuMode('invalid'); return false; } catch { return true; }
        }), true);
        assert.equal(await page.evaluate(async () => {
            try { await window.gctd.setSkimMode('true'); return false; } catch { return true; }
        }), true);

        await page.evaluate(() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.id = 'testFile';
            document.body.append(input);
        });
        await page.locator('#testFile').setInputFiles(filePath);
        await page.evaluate(() => window.gctd.submitFiles(Array.from(document.getElementById('testFile').files)));
        await page.waitForFunction(() => document.querySelectorAll('#videoTableBody tr').length === 1);
        assert.equal(await page.locator('#videoTableBody tr td').first().textContent(), "O'Brien & sons.mp4");
        assert.equal(await page.evaluate(async () => {
            try { await window.gctd.submitFiles([new File(['test'], 'synthetic.mp4')]); return false; }
            catch { return true; }
        }), true);
        await page.evaluate(() => {
            const files = Array.from(document.getElementById('testFile').files);
            const event = new Event('drop', { bubbles: true, cancelable: true });
            Object.defineProperty(event, 'dataTransfer', { value: { files } });
            document.getElementById('dropZone').dispatchEvent(event);
        });
        await page.waitForFunction(() => document.querySelectorAll('#videoTableBody tr').length === 2);

        // Chromium creates native disk-backed File objects for an actual folder drop.
        const dropZone = await page.locator('#dropZone').boundingBox();
        const session = await page.context().newCDPSession(page);
        const drop = { x: dropZone.x + 20, y: dropZone.y + 20,
            data: { items: [], files: [temporary], dragOperationsMask: 1 } };
        for (const type of ['dragEnter', 'dragOver', 'drop'])
            await session.send('Input.dispatchDragEvent', { type, ...drop });
        await page.waitForFunction(() => document.querySelectorAll('#videoTableBody tr').length === 3);
        await session.detach();

        await application.evaluate(({ BrowserWindow }) => {
            const sender = BrowserWindow.getAllWindows()[0].webContents;
            sender.send('files-added', [{ id: "bad' id", name: 'bad.mp4', status: 'Pending' }]);
            sender.send('files-added', [{ id: 20, name: '<span data-vince=marker>.mp4', status: '<b>Pending</b>' }]);
            sender.send('status-updated', { id: 20, status: '<i>50% complete</i>' });
            sender.send('status-complete', { id: 0 });
        });
        await page.waitForFunction(() => document.querySelectorAll('#videoTableBody tr').length === 4);
        const rows = await page.locator('#videoTableBody tr').allTextContents();
        assert.ok(rows.some(row => row.includes('<span data-vince=marker>.mp4') && row.includes('<i>50% complete</i>')));
        assert.equal(await page.locator('#videoTableBody [data-vince], #videoTableBody span, #videoTableBody i').count(), 0);
        await page.getByRole('button', { name: 'View Result' }).click();
        await page.waitForFunction(() => !document.getElementById('errorMessage').textContent.includes('Unknown'));
        assert.equal(await page.evaluate(async () => {
            try { await window.gctd.openResult('../outside'); return false; } catch { return true; }
        }), true);
        assert.equal(await page.evaluate(async () => {
            try { await window.gctd.openResult(9999); return false; } catch { return true; }
        }), true);
        await page.getByRole('link', { name: 'MIT License' }).click();
        assert.deepEqual(await application.evaluate(() => global.testExternalURLs), ['https://opensource.org/licenses/MIT']);
        assert.deepEqual(await application.evaluate(() => global.testOpenPaths), [path.join(temporary, 'result')]);
        const originalURL = page.url();
        await application.evaluate(({ BrowserWindow }) => {
            const contents = BrowserWindow.getAllWindows()[0].webContents;
            global.testNavigation = new Promise(resolve => contents.once('will-navigate', () => resolve(true)));
        });
        await page.evaluate(() => { window.location.href = 'https://example.invalid'; });
        assert.equal(await application.evaluate(() => Promise.race([
            global.testNavigation, new Promise(resolve => setTimeout(() => resolve(false), 5000))
        ])), true);
        assert.equal(page.url(), originalURL);
        assert.equal(await page.evaluate(() => window.open('https://example.invalid') === null), true);
        assert.equal(await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length), 1);
        assert.ok(await page.evaluate(() => {
            const policy = document.querySelector('meta[http-equiv="Content-Security-Policy"]').content;
            return policy.includes("script-src 'self'") && !policy.includes('unsafe-inline') && !policy.includes('unsafe-eval');
        }));
        assert.equal(await page.evaluate(() => Array.from(document.scripts).some(script => script.src.startsWith('https:'))), false);
    } finally {
        if (application)
            await application.close();
        fs.rmSync(temporary, { recursive: true, force: true, maxRetries: 5 });
    }
});
