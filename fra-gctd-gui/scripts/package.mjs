import { packager } from '@electron/packager';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const executables = path.join(root, 'resources', 'executables');
const outputs = await packager({
    dir: root,
    name: 'fra-gctd-gui',
    out: path.join(root, 'out'),
    overwrite: true,
    asar: true,
    ignore: [/^\/test($|\/)/, /^\/scripts($|\/)/, /^\/resources($|\/)/],
    extraResource: existsSync(executables) ? [executables] : []
});
console.log('Packaged GUI:', outputs.join(', '));
if (!existsSync(path.join(executables, 'process_video', 'process_video.exe')))
    console.warn('GUI-only package: add the PhaseTwo PyInstaller output under resources/executables/process_video for video inference.');
