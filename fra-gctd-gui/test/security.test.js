const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const Module = require('node:module');
const security = require('../src/main-process/security');

const src = path.resolve(__dirname, '../src');
const plain = value => JSON.parse(JSON.stringify(value));

function fileURL(filename) {
    const url = new URL('file:///');
    url.pathname = filename.replace(/\\/g, '/');
    return url.href;
}

// Load actual CommonJS sources in isolation, replacing only explicit dependencies.
function loadSource(filename, mocks = {}, globals = {}) {
    const sandbox = { console, process, ...globals };
    sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    const cache = new Map();
    function load(current) {
        if (cache.has(current))
            return cache.get(current).exports;
        const nativeRequire = Module.createRequire(current);
        const loaded = { exports: {} };
        cache.set(current, loaded);
        const localRequire = request => {
            if (Object.prototype.hasOwnProperty.call(mocks, request))
                return mocks[request];
            const resolved = nativeRequire.resolve(request);
            return path.isAbsolute(resolved) ? load(resolved) : nativeRequire(request);
        };
        const wrapper = vm.runInContext(Module.wrap(fs.readFileSync(current, 'utf8')), context, { filename: current });
        wrapper(loaded.exports, localRequire, loaded, current, path.dirname(current));
        return loaded.exports;
    }
    return load(path.join(src, filename));
}

function temporaryDirectory(t) {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gctd-security-'));
    t.after(() => fs.rmSync(directory, { recursive: true, force: true, maxRetries: 3 }));
    return directory;
}

function directoryLink(t, target, link) {
    try {
        fs.symlinkSync(target, link, process.platform === 'win32' ? 'junction' : 'dir');
        return true;
    } catch (error) {
        if (!['EPERM', 'EACCES', 'ENOTSUP'].includes(error.code))
            throw error;
        t.skip('Directory links are unavailable: ' + error.code);
        return false;
    }
}

function emitter() {
    const listeners = new Map();
    return {
        on(channel, listener) {
            if (!listeners.has(channel))
                listeners.set(channel, []);
            listeners.get(channel).push(listener);
            return this;
        },
        emit(channel, ...args) {
            for (const listener of listeners.get(channel) || [])
                listener(...args);
        }
    };
}

function trustedWindow() {
    const window = {
        destroyed: false,
        isDestroyed() { return this.destroyed; },
        webContents: { mainFrame: { url: fileURL(path.join(src, 'index.html')) } }
    };
    return { window, event: { sender: window.webContents, senderFrame: window.webContents.mainFrame } };
}

function mainHarness(t, fsMock) {
    const root = temporaryDirectory(t);
    const handlers = new Map();
    const timeline = [];
    const parseCalls = [];
    const results = new Map();
    const openPaths = [];
    const externalURLs = [];
    const dialogCalls = [];
    const settings = new Map([['outputDir', root], ['cpuMode', 'cpu'], ['skimMode', false], ['private', 'not exposed']]);
    const writes = [];
    const windows = [];
    const state = { dialogResult: { canceled: true, filePaths: [] }, openError: '' };
    let ready;
    class BrowserWindow {
        constructor(options) {
            this.options = options;
            this.destroyed = false;
            this.webContents = Object.assign(emitter(), {
                mainFrame: { url: '' },
                send: (channel, data) => timeline.push({ channel, data: plain(data) }),
                session: {
                    setPermissionRequestHandler: handler => { this.permissionRequest = handler; },
                    setPermissionCheckHandler: handler => { this.permissionCheck = handler; }
                },
                setWindowOpenHandler: handler => { this.windowOpen = handler; }
            });
            windows.push(this);
        }
        isDestroyed() { return this.destroyed; }
        maximize() {}
        loadFile(filename) {
            this.loadedFile = filename;
            this.webContents.mainFrame.url = fileURL(filename);
        }
        static getAllWindows() { return windows; }
    }
    const electron = {
        app: { whenReady: () => ({ then: callback => { ready = callback; } }), on() {}, quit() {} },
        BrowserWindow,
        Menu: { setApplicationMenu() {} },
        ipcMain: { handle: (channel, handler) => {
            assert.equal(handlers.has(channel), false, 'duplicate handler: ' + channel);
            handlers.set(channel, handler);
        } },
        dialog: { showOpenDialog: async (...args) => {
            dialogCalls.push(args);
            return state.showDialog ? state.showDialog() : state.dialogResult;
        } },
        shell: {
            openPath: async target => { openPaths.push(target); return state.openError; },
            openExternal: async url => { externalURLs.push(url); }
        }
    };
    const parseVideo = {
        addVideo: (filename, id, sender) => {
            timeline.push({ channel: 'parse', id });
            parseCalls.push({ filename, id, sender });
            sender.send('status-updated', { id, status: 'Processing' });
        },
        getResult: (id, sender) => {
            const result = results.get(id);
            return result && result.complete && result.owner === sender ? result : undefined;
        }
    };
    const mocks = {
        electron,
        'electron-squirrel-startup': false,
        './parseVideo': parseVideo,
        './main-process/settingsManager': class {
            get(key) { return settings.get(key); }
            set(key, value) { writes.push([key, value]); settings.set(key, value); }
        }
    };
    if (fsMock)
        mocks.fs = fsMock;
    loadSource('index.js', mocks);
    ready();
    const window = windows[0];
    const event = { sender: window.webContents, senderFrame: window.webContents.mainFrame };
    return {
        root, handlers, timeline, parseCalls, results, openPaths, externalURLs, dialogCalls,
        settings, writes, state, window, event,
        invoke: (channel, ...args) => handlers.get(channel)(event, ...args)
    };
}

function parseHarness(t, packaged = false) {
    const root = temporaryDirectory(t);
    const outputRoot = path.join(root, 'user-selected-output');
    const children = [];
    const streams = [];
    const calls = [];
    const messages = [];
    const state = { spawnError: false, delayedFlush: false };
    const sender = { isDestroyed: () => false, send: (channel, data) => messages.push({ channel, data: plain(data) }) };
    const settings = new Map([['outputDir', outputRoot], ['cpuMode', 'cpu'], ['skimMode', true]]);
    const parser = loadSource('main-process/parseVideo.js', {
        electron: { app: { isPackaged: packaged, getAppPath: () => root } },
        fs: { ...fs, createWriteStream: (filename, options) => {
            const stream = Object.assign(emitter(), {
                filename, options, chunks: [], ended: false,
                write(data) { this.chunks.push(data); },
                end() { this.ended = true; if (!state.delayedFlush) this.emit('finish'); }
            });
            streams.push(stream);
            return stream;
        } },
        child_process: { execFile: (executable, args, options) => {
            calls.push({ executable, args: plain(args), options: plain(options) });
            if (state.spawnError)
                throw new Error('Mocked spawn failure');
            const child = Object.assign(emitter(), {
                stdout: emitter(), stderr: emitter(), killed: false,
                kill() { this.killed = true; }
            });
            children.push(child);
            return child;
        } }
    }, { settings: { get: key => settings.get(key) }, process: { platform: process.platform, resourcesPath: path.join(root, 'packaged-resources') } });
    return { root, outputRoot, children, streams, calls, messages, sender, parser, settings, state };
}

test('sender trust requires the live main window, exact main frame and bundled page URL', () => {
    const { window, event } = trustedWindow();
    assert.equal(security.isTrustedSender(event, window), true);
    for (const sender of [null, {}, { mainFrame: event.senderFrame }])
        assert.equal(security.isTrustedSender({ ...event, sender }, window), false);
    assert.equal(security.isTrustedSender({ ...event, senderFrame: { url: event.senderFrame.url } }, window), false);
    for (const url of ['about:blank', 'https://example.invalid/index.html', fileURL(path.join(src, 'other', 'index.html')), event.senderFrame.url + '?query', event.senderFrame.url + '#fragment']) {
        window.webContents.mainFrame.url = url;
        assert.equal(security.isTrustedSender(event, window), false, url);
    }
    window.webContents.mainFrame.url = fileURL(path.join(src, 'index.html'));
    assert.equal(security.isTrustedSender(event, window), true);
    window.destroyed = true;
    assert.equal(security.isTrustedSender(event, window), false);
    assert.equal(security.isTrustedSender(event, null), false);
});

test('sender trust safely rejects absent events and frames', () => {
    const { window, event } = trustedWindow();
    for (const invalid of [null, undefined, {}, { ...event, senderFrame: null }, { ...event, senderFrame: undefined }])
        assert.equal(security.isTrustedSender(invalid, window), false);
});

test('output directories use the canonical user-selected root and require a strict directory descendant', t => {
    const temporary = temporaryDirectory(t);
    const root = path.join(temporary, 'chosen-output');
    const result = security.createOutputDirectory(root + path.sep + '.', 'video');
    assert.deepEqual(result, { root: fs.realpathSync(root), path: fs.realpathSync(path.join(root, 'video')) });
    assert.deepEqual(security.createOutputDirectory(root, 'video'), result);
    assert.equal(security.resolveOutputDirectory(root, result.path + path.sep + '.'), result.path);
    const file = path.join(root, 'file');
    fs.writeFileSync(file, 'fixture');
    const sibling = root + '-sibling';
    fs.mkdirSync(sibling);
    for (const target of [root, file, sibling, path.join(root, '..'), path.join(root, '..', 'chosen-output-sibling')])
        assert.throws(() => security.resolveOutputDirectory(root, target));
    for (const invalid of [null, '', 'relative']) {
        assert.throws(() => security.resolveOutputDirectory(invalid, result.path));
        assert.throws(() => security.resolveOutputDirectory(root, invalid));
        assert.throws(() => security.createOutputDirectory(invalid, 'video'));
    }
    for (const name of [null, '', '.', '..', '..' + path.sep + 'escape', 'nested' + path.sep + 'video', path.join(temporary, 'escape')])
        assert.throws(() => security.createOutputDirectory(root, name));
    assert.throws(() => security.createOutputDirectory(file, 'video'));
    assert.throws(() => security.createOutputDirectory(root, 'file'));
    assert.throws(() => security.resolveOutputDirectory(file, result.path));
    assert.equal(fs.existsSync(path.join(temporary, 'escape')), false);
});

test('canonical roots allow selected directory links but reject symlink or junction escapes', t => {
    const temporary = temporaryDirectory(t);
    const root = path.join(temporary, 'root');
    const outside = path.join(temporary, 'outside');
    fs.mkdirSync(root);
    fs.mkdirSync(outside);
    const alias = path.join(temporary, 'selected-root-link');
    if (!directoryLink(t, root, alias))
        return;
    const result = security.createOutputDirectory(alias, 'video');
    assert.equal(result.root, fs.realpathSync(root));
    assert.equal(result.path, fs.realpathSync(path.join(root, 'video')));
    const escape = path.join(root, 'escape');
    if (!directoryLink(t, outside, escape))
        return;
    assert.throws(() => security.resolveOutputDirectory(root, escape));
    assert.throws(() => security.createOutputDirectory(root, 'escape'));
    assert.deepEqual(fs.readdirSync(outside), []);
});

test('preload exposes only operation-specific methods with the exact invoke contract', async () => {
    const exposed = [];
    const calls = [];
    const diskFiles = new WeakMap();
    const electron = {
        contextBridge: { exposeInMainWorld: (name, api) => exposed.push({ name, api }) },
        ipcRenderer: { invoke: (...args) => { calls.push(plain(args)); return Promise.resolve(); } },
        webUtils: { getPathForFile: file => {
            if (!diskFiles.has(file))
                throw new TypeError('Not a File');
            return diskFiles.get(file);
        } },
        shell: { openExternal() { assert.fail('shell must not be exposed'); } }
    };
    loadSource('preload.js', { electron });
    assert.equal(exposed.length, 1);
    assert.equal(exposed[0].name, 'gctd');
    const api = exposed[0].api;
    assert.deepEqual(Object.keys(api).sort(), ['selectFiles', 'selectOutputDir', 'getSettings', 'setCpuMode', 'setSkimMode', 'submitFiles', 'openResult', 'openLicense', 'onFilesAdded', 'onStatusUpdated', 'onStatusComplete'].sort());
    for (const forbidden of ['ipcRenderer', 'shell', 'send', 'invoke', 'on', 'require'])
        assert.equal(api[forbidden], undefined);
    const file = Object.defineProperty({}, 'path', { get() { assert.fail('Legacy File.path must not be read'); } });
    const diskPath = path.join(os.tmpdir(), 'disk-backed-video.mp4');
    diskFiles.set(file, diskPath);
    await api.selectFiles();
    await api.selectOutputDir();
    await api.getSettings();
    await api.setCpuMode('gpu');
    await api.setSkimMode(true);
    await api.submitFiles([file]);
    await api.openResult(7);
    await api.openLicense('https://example.invalid/ignored');
    assert.deepEqual(calls, [['select-files'], ['select-output-dir'], ['get-settings'], ['set-cpu-mode', 'gpu'], ['set-skim-mode', true], ['submit-files', [diskPath]], ['open-result', 7], ['open-license']]);
    const memoryFile = {};
    diskFiles.set(memoryFile, '');
    for (const invalid of [null, {}, [], new Array(1001).fill(file), [null], [{}], [{ path: diskPath }], [memoryFile], [file, memoryFile]])
        await assert.rejects(api.submitFiles(invalid));
    assert.equal(calls.length, 8, 'rejected files must not reach IPC');
});

test('preload event wrappers discard privileged events and unsubscribe the exact listener', () => {
    let api;
    const listeners = new Map();
    const removed = [];
    loadSource('preload.js', { electron: {
        contextBridge: { exposeInMainWorld: (_name, value) => { api = value; } },
        ipcRenderer: {
            on: (channel, listener) => listeners.set(channel, listener),
            removeListener: (channel, listener) => {
                assert.equal(listeners.get(channel), listener);
                removed.push(channel);
                listeners.delete(channel);
            }
        },
        webUtils: {}
    } });
    for (const [method, channel] of [['onFilesAdded', 'files-added'], ['onStatusUpdated', 'status-updated'], ['onStatusComplete', 'status-complete']]) {
        assert.throws(() => api[method](null), { name: 'TypeError' });
        const received = [];
        const unsubscribe = api[method]((...args) => received.push(args));
        const data = { id: 7 };
        listeners.get(channel)({ sender: { send() { assert.fail('privileged event leaked'); } } }, data, 'extra');
        assert.deepEqual(received, [[data]]);
        assert.equal(typeof unsubscribe, 'function');
        unsubscribe();
    }
    assert.deepEqual(removed, ['files-added', 'status-updated', 'status-complete']);
    assert.equal(listeners.size, 0);
});

test('main registers only constrained operations and guards every handler against other senders', async t => {
    const h = mainHarness(t);
    assert.deepEqual([...h.handlers.keys()].sort(), ['get-settings', 'select-output-dir', 'set-cpu-mode', 'set-skim-mode', 'open-license', 'select-files', 'submit-files', 'open-result'].sort());
    const badEvent = { sender: {}, senderFrame: h.event.senderFrame };
    for (const handler of h.handlers.values())
        await assert.rejects(handler(badEvent, 'untrusted'), /Untrusted IPC sender/);
    assert.deepEqual(h.writes, []);
    assert.deepEqual(h.openPaths, []);
    assert.deepEqual(h.externalURLs, []);
    assert.deepEqual(h.dialogCalls, []);
    assert.deepEqual(h.parseCalls, []);
    assert.equal(h.window.loadedFile, path.join(src, 'index.html'));
    assert.deepEqual(plain(h.window.options.webPreferences), { preload: path.join(src, 'preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true });
    for (const channel of ['will-navigate', 'will-redirect', 'will-attach-webview']) {
        let prevented = false;
        h.window.webContents.emit(channel, { preventDefault: () => { prevented = true; } });
        assert.equal(prevented, true);
    }
    assert.deepEqual(plain(h.window.windowOpen()), { action: 'deny' });
    h.window.permissionRequest(null, 'permission', allowed => assert.equal(allowed, false));
    assert.equal(h.window.permissionCheck(), false);
});

test('main validates settings and opens only the fixed license URL', async t => {
    const h = mainHarness(t);
    assert.deepEqual(plain(await h.invoke('get-settings')), { outputDir: h.root, cpuMode: 'cpu', skimMode: false });
    for (const invalid of [null, {}, '', 'other', ['cpu']])
        await assert.rejects(h.invoke('set-cpu-mode', invalid), { name: 'TypeError' });
    for (const invalid of [null, {}, 0, 'true'])
        await assert.rejects(h.invoke('set-skim-mode', invalid), { name: 'TypeError' });
    assert.deepEqual(h.writes, []);
    for (const mode of ['cpu', 'gpu'])
        await h.invoke('set-cpu-mode', mode);
    for (const active of [true, false])
        await h.invoke('set-skim-mode', active);
    assert.deepEqual(h.writes, [['cpuMode', 'cpu'], ['cpuMode', 'gpu'], ['skimMode', true], ['skimMode', false]]);
    await h.invoke('open-license', 'https://example.invalid/not-the-license');
    assert.deepEqual(h.externalURLs, ['https://opensource.org/licenses/MIT']);
    assert.deepEqual(h.openPaths, []);
});

test('native selections use dialog paths and recheck trust after each dialog', async t => {
    const h = mainHarness(t);
    const selected = path.join(h.root, 'user-selected-directory');
    fs.mkdirSync(selected);
    h.state.dialogResult = { canceled: false, filePaths: [selected] };
    assert.equal((await h.invoke('select-output-dir', 'ignored-renderer-path')).outputDir, selected);
    assert.deepEqual(plain(h.dialogCalls[0][1]), { properties: ['openDirectory'] });
    h.state.dialogResult = { canceled: true, filePaths: [h.root] };
    await h.invoke('select-output-dir');
    assert.equal(h.settings.get('outputDir'), selected);
    const video = path.join(selected, 'selected.mp4');
    fs.writeFileSync(video, 'fixture');
    h.state.dialogResult = { canceled: false, filePaths: [video] };
    await h.invoke('select-files', ['ignored-renderer-path']);
    assert.deepEqual(plain(h.dialogCalls[2][1]), {
        properties: ['openFile', 'multiSelections'],
        filters: [{ name: 'Videos', extensions: ['mp4', 'avi'] }]
    });
    assert.equal(h.parseCalls.length, 1);
    assert.equal(h.parseCalls[0].filename, video);
    const timeline = plain(h.timeline);
    for (const channel of ['select-output-dir', 'select-files']) {
        h.window.webContents.mainFrame.url = fileURL(path.join(src, 'index.html'));
        h.state.showDialog = () => {
            h.window.webContents.mainFrame.url = 'about:blank';
            return { canceled: false, filePaths: [h.root] };
        };
        await assert.rejects(h.invoke(channel), /no longer trusted/);
    }
    assert.deepEqual(h.writes, [['outputDir', selected]]);
    assert.deepEqual(h.timeline, timeline);
    assert.equal(h.parseCalls.length, 1);
});

test('file submission rejects malformed paths before filesystem traversal or processing', async t => {
    const h = mainHarness(t);
    for (const invalid of [null, {}, h.root, [], new Array(1001).fill(h.root), ['relative.mp4'], [42], [null], [h.root + '\0video.mp4']])
        await assert.rejects(h.invoke('submit-files', invalid), /Invalid file selection/);
    await assert.rejects(h.invoke('submit-files', [path.join(h.root, 'missing.mp4')]), { code: 'ENOENT' });
    assert.deepEqual(h.timeline, []);
    assert.deepEqual(h.parseCalls, []);
    assert.deepEqual(h.openPaths, []);
});

test('file traversal enforces the 10000-entry bound, including before expanding a directory', async t => {
    let children = 10000;
    let visited = 0;
    const directory = path.join(os.tmpdir(), 'virtual-selection');
    const h = mainHarness(t, {
        ...fs,
        lstatSync: filename => {
            visited++;
            return { isSymbolicLink: () => false, isDirectory: () => filename === directory, isFile: () => filename !== directory };
        },
        realpathSync: filename => filename,
        readdirSync: () => Array.from({ length: children }, (_value, index) => 'ignored-' + index + '.txt')
    });
    await assert.rejects(h.invoke('submit-files', [directory]), /Selection is too large/);
    assert.equal(visited, 1, 'reject oversized directories before visiting children');
    assert.deepEqual(h.timeline, []);
    children = 9999;
    visited = 0;
    await h.invoke('submit-files', [directory]);
    assert.equal(visited, 10000);
    assert.deepEqual(h.timeline, [{ channel: 'files-added', data: [] }]);
    assert.deepEqual(h.parseCalls, []);
});

test('file traversal filters extensions, deduplicates canonical paths and publishes rows before parsing', async t => {
    const h = mainHarness(t);
    const nested = path.join(h.root, 'nested');
    fs.mkdirSync(nested);
    for (const filename of ['one.MP4', 'two.avi', 'ignored.mp4.txt', 'ignored.txt'])
        fs.writeFileSync(path.join(h.root, filename), 'fixture');
    fs.writeFileSync(path.join(nested, 'three.AVI'), 'fixture');
    await h.invoke('submit-files', [h.root, path.join(h.root, 'one.MP4'), h.root + path.sep + '.' + path.sep + 'one.MP4']);
    assert.equal(h.timeline[0].channel, 'files-added');
    const rows = h.timeline[0].data;
    assert.deepEqual(rows.map(row => row.name).sort(), ['one.MP4', 'three.AVI', 'two.avi']);
    assert.equal(new Set(rows.map(row => row.id)).size, 3);
    for (const row of rows) {
        assert.deepEqual(Object.keys(row).sort(), ['id', 'name', 'status']);
        assert.equal(Number.isSafeInteger(row.id) && row.id >= 0, true);
        assert.equal(row.status, 'Pending');
    }
    assert.equal(h.parseCalls.length, 3);
    for (const call of h.parseCalls) {
        assert.equal(call.sender, h.event.sender);
        assert.equal(rows.find(row => row.id === call.id).name, path.basename(call.filename));
    }
    assert.deepEqual(h.timeline.slice(1).map(item => item.channel), ['parse', 'status-updated', 'parse', 'status-updated', 'parse', 'status-updated']);
});

test('file traversal does not follow symlink or junction directories', async t => {
    const h = mainHarness(t);
    const selected = path.join(h.root, 'selected');
    const outside = path.join(h.root, 'outside');
    fs.mkdirSync(selected);
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, 'not-selected.mp4'), 'fixture');
    const link = path.join(selected, 'linked-directory');
    if (!directoryLink(t, outside, link))
        return;
    await h.invoke('submit-files', [selected, link]);
    assert.deepEqual(h.timeline, [{ channel: 'files-added', data: [] }]);
    assert.deepEqual(h.parseCalls, []);
});

test('open-result rejects invalid, unknown, incomplete, wrong-owner and escaping results without shell calls', async t => {
    const h = mainHarness(t);
    const output = security.createOutputDirectory(path.join(h.root, 'output'), 'video');
    const outside = path.join(h.root, 'output-sibling');
    fs.mkdirSync(outside);
    const file = path.join(output.root, 'file');
    fs.writeFileSync(file, 'fixture');
    for (const invalid of [-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, '1', null, {}])
        await assert.rejects(h.invoke('open-result', invalid), /Invalid result ID/);
    await assert.rejects(h.invoke('open-result', 1), /Unknown or incomplete/);
    const result = { ...output, owner: h.event.sender, complete: false };
    h.results.set(1, result);
    await assert.rejects(h.invoke('open-result', 1), /Unknown or incomplete/);
    result.complete = true;
    result.owner = {};
    await assert.rejects(h.invoke('open-result', 1), /Unknown or incomplete/);
    result.owner = h.event.sender;
    await assert.rejects(h.handlers.get('open-result')({ sender: {}, senderFrame: h.event.senderFrame }, 1), /Untrusted IPC sender/);
    for (const target of [outside, output.root, file, path.join(output.root, '..')]) {
        result.path = target;
        await assert.rejects(h.invoke('open-result', 1));
    }
    assert.deepEqual(h.openPaths, []);
    result.path = output.path;
    await h.invoke('open-result', 1);
    assert.deepEqual(h.openPaths, [fs.realpathSync(output.path)]);
    h.state.openError = 'mocked OS error';
    await assert.rejects(h.invoke('open-result', 1), /Unable to open the result directory/);
});

test('open-result revalidates canonical paths instead of trusting a previously recorded directory', async t => {
    const h = mainHarness(t);
    const output = security.createOutputDirectory(path.join(h.root, 'output'), 'video');
    const outside = path.join(h.root, 'outside');
    fs.mkdirSync(outside);
    fs.rmdirSync(output.path);
    if (!directoryLink(t, outside, output.path))
        return;
    h.results.set(1, { ...output, owner: h.event.sender, complete: true });
    await assert.rejects(h.invoke('open-result', 1));
    assert.deepEqual(h.openPaths, []);
});

test('parseVideo uses an absolute executable without a shell and exposes results only after successful close to the owner', t => {
    for (const packaged of [false, true]) {
        const h = parseHarness(t, packaged);
        const input = path.join(h.root, 'video with spaces.mp4');
        h.parser.addVideo(input, 7, h.sender);
        assert.equal(h.calls.length, 1);
        const resources = path.join(h.root, packaged ? 'packaged-resources' : 'resources');
        const processDirectory = path.join(resources, 'executables', 'process_video');
        const call = h.calls[0];
        assert.equal(path.isAbsolute(call.executable), true);
        assert.equal(call.executable, path.join(processDirectory, 'process_video.exe'));
        assert.deepEqual(call.options, { cwd: processDirectory });
        assert.notEqual(call.options.shell, true);
        const output = path.join(h.outputRoot, 'video with spaces');
        assert.deepEqual(call.args, ['--inputpath', input, '--outputpath', output, '--cpu', '--skim']);
        assert.equal(h.streams[0].options.flags, 'wx');
        assert.equal(path.dirname(h.streams[0].filename), output);
        assert.equal(h.parser.getResult(7, h.sender), undefined);
        h.children[0].stdout.emit('data', 'Processing: 50% complete\nTotal running ');
        h.children[0].stdout.emit('data', 'time: 3 seconds\n');
        h.children[0].stderr.emit('data', 'diagnostic');
        assert.equal(h.parser.getResult(7, h.sender), undefined, 'completion text alone is not success');
        h.children[0].emit('exit', 0);
        assert.equal(h.parser.getResult(7, h.sender), undefined, 'exit alone is not a closed process');
        assert.equal(h.messages.some(message => message.channel === 'status-complete'), false);
        h.children[0].emit('close', 0);
        const result = h.parser.getResult(7, h.sender);
        assert.equal(result.sender, h.sender);
        assert.equal(result.root, fs.realpathSync(h.outputRoot));
        assert.equal(result.path, fs.realpathSync(output));
        assert.equal(h.parser.getResult(7, {}), undefined);
        assert.equal(h.parser.getResult(8, h.sender), undefined);
        assert.equal(h.streams[0].ended, true);
        assert.equal(h.messages.some(message => message.data.status === 'Completed in 3 seconds'), true);
        assert.deepEqual(h.messages.filter(message => message.channel === 'status-complete'), [{ channel: 'status-complete', data: { id: 7 } }]);
    }
});

test('parseVideo never exposes a result after nonzero/signal exit, child/log errors or spawn failure', t => {
    for (const failure of ['nonzero', 'signal', 'child-error', 'log-error', 'spawn-error']) {
        const h = parseHarness(t);
        h.state.spawnError = failure === 'spawn-error';
        h.parser.addVideo(path.join(h.root, 'video.mp4'), 1, h.sender);
        if (failure !== 'spawn-error') {
            const child = h.children[0];
            child.stdout.emit('data', 'Total running time: 1 second\n');
            if (failure === 'child-error')
                child.emit('error', new Error('Mocked child error'));
            if (failure === 'log-error') {
                h.streams[0].emit('error', new Error('Mocked log error'));
                assert.equal(child.killed, true);
            }
            child.emit('close', failure === 'nonzero' ? 1 : failure === 'signal' ? null : 0);
        }
        assert.equal(h.parser.getResult(1, h.sender), undefined, failure);
        assert.equal(h.messages.some(message => message.channel === 'status-complete'), false, failure);
        assert.equal(h.streams[0].ended, true, failure);
    }
});

test('parseVideo waits for log flush and rejects late log failures after successful child close', t => {
    for (const failed of [false, true]) {
        const h = parseHarness(t);
        h.state.delayedFlush = true;
        h.parser.addVideo(path.join(h.root, 'video.mp4'), 1, h.sender);
        h.children[0].emit('close', 0);
        assert.equal(h.parser.getResult(1, h.sender), undefined);
        assert.equal(h.messages.some(message => message.channel === 'status-complete'), false);
        h.streams[0].emit(failed ? 'error' : 'finish', new Error('Mocked final write error'));
        assert.equal(Boolean(h.parser.getResult(1, h.sender)), !failed);
        assert.equal(h.messages.filter(message => message.channel === 'status-complete').length, failed ? 0 : 1);
    }
});

test('parseVideo drains thousands of consecutive start failures without recursion', t => {
    const h = parseHarness(t);
    h.parser.addVideo(path.join(h.root, 'first.mp4'), 0, h.sender);
    for (let id = 1; id < 10000; id++)
        h.parser.addVideo(path.join(h.root, id + '.mp4'), id, h.sender);
    const blocked = path.join(h.root, 'not-a-directory');
    fs.writeFileSync(blocked, 'fixture');
    h.settings.set('outputDir', blocked);
    assert.doesNotThrow(() => h.children[0].emit('close', 0));
    assert.equal(h.messages.filter(message => message.data.status && message.data.status.startsWith('Unable to start processing:')).length, 9999);
    assert.equal(h.calls.length, 1);
});
