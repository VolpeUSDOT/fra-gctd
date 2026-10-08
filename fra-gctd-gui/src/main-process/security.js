const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const pageURL = pathToFileURL(path.join(__dirname, '..', 'index.html')).href;

function isTrustedSender(event, window) {
    return Boolean(event && window && !window.isDestroyed() &&
        event.sender === window.webContents &&
        event.senderFrame === window.webContents.mainFrame &&
        event.senderFrame.url === pageURL);
}

function isInside(root, target) {
    const relative = path.relative(root, target);
    return relative !== '' && relative !== '..' &&
        !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
}

function resolveOutputDirectory(root, target) {
    if (typeof root !== 'string' || typeof target !== 'string' ||
        !path.isAbsolute(root) || !path.isAbsolute(target))
        throw new Error('Invalid output directory');
    const realRoot = fs.realpathSync(root);
    const realTarget = fs.realpathSync(target);
    if (!fs.statSync(realRoot).isDirectory() ||
        !fs.statSync(realTarget).isDirectory() || !isInside(realRoot, realTarget))
        throw new Error('Result must be a directory inside its selected output root');
    return realTarget;
}

function createOutputDirectory(root, name) {
    if (typeof root !== 'string' || !path.isAbsolute(root) ||
        !name || name === '.' || name === '..' || path.basename(name) !== name)
        throw new Error('Invalid output directory');
    fs.mkdirSync(root, { recursive: true });
    const realRoot = fs.realpathSync(root);
    const target = path.join(realRoot, name);
    if (!isInside(realRoot, target))
        throw new Error('Output path escapes the selected directory');
    if (!fs.existsSync(target))
        fs.mkdirSync(target);
    return { root: realRoot, path: resolveOutputDirectory(realRoot, target) };
}

module.exports = { isTrustedSender, resolveOutputDirectory, createOutputDirectory };
