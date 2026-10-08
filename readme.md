# FRA Grade Crossing Trespass Detection 
This repository is home to the python scripts used to perform the trespass detection on various FRA data sets.

#### Folder Structure
**/PhaseOne**   *Phase one inference tools built using Tensorflow 1.x*

**/PhaseTwo**   *Phase two inference tools built using PyTorch 1.x*

**/fra-gctd-gui** *Desktop GUI to use alongside phase two inference tools. Built using Electron.

## Requirements

- Python 3.x (Tested on 3.7.x)
- Tensorflow 1.15.x (Phase One)
- PyTorch 1.5.x (Phase Two)

## Installation

Installation instructions for each component may be found in the relevant folder within this repository.

### Desktop GUI

The GUI uses Electron 44.7.0 and requires Node.js 24 or later for development.
Run these commands from `fra-gctd-gui`:

```text
npm ci
npm test
npm run test:electron
npm start
npm run package
npm run test:package
```

`npm run package` creates a platform-specific GUI bundle in `fra-gctd-gui/out`.
`npm run test:package` smoke-tests the Windows x64 bundle after packaging.
For Windows video processing, copy the complete PhaseTwo PyInstaller output
directory (including models and runtime files) to
`fra-gctd-gui/resources/executables/process_video` before packaging. Without this
external payload the GUI and security tests work, but inference is unavailable.
The Python models and processing executable are not distributed in this repository.

The GUI loads only bundled assets, isolates and sandboxes the renderer, and
opens result directories through a main-process result ID rather than accepting
arbitrary renderer-supplied paths. Keep Electron current when releasing updates.

## License

MIT
