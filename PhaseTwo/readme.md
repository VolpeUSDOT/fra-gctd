# FRA Grade Crossing Trespass Detection 
This repository is home to the python scripts used to perform the trespass detection on various FRA data sets.

#### Folder Structure
**/PhaseOne**   *Phase one inference tools built using Tensorflow 1.x*

**/PhaseTwo**   *Phase two inference tools built using PyTorch 1.x*

## Requirements

- Python 3.x (Tested on 3.7.x)
- Tensorflow 1.15.x (Phase One)
- PyTorch 1.5.x (Phase Two)

## Installation

Installation instructions are coming soon...

## Usage Examples

Useful FFMPEG frame extraction commands
```
ffmpeg -i I:\projects\temp\fravideos\KALK0090_20130307133638.asf -qscale:v 2 I:\projects\temp\fraframes\KALK0090_20130307133638_%05d.jpg
```
Generate video
```
ffmpeg -i fraoutput\frame_%05d.jpg -c:v libx264 -vf "fps=15,format=yuv420p,scale=w=900:h=-1:force_original_aspect_ratio=decrease,pad='iw+mod(iw\,2)':'ih+mod(ih\,2)'" FRA_GCTD_demo1.mp4
```

## License

MIT
