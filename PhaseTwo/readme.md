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

Download or clone this directory from Git. Ensure the required dependancies are installed in your python enviornment. A full list of dependancies may be found in 'requirements.txt'.

Next, download the required models from (URL to be determined) and save thime to '/PhaseTwo/models'.

## Usage

The script may be run with the following command:

```
python process_video.py -i /path/to/your/video/file.mp4 -o /path/to/your/output/directory
```

A full list of options may be found below.

Flag | Short Flag | Properties | Description
:------:|:---------------:|:---------------------:|:-----------:
--inputpath|-i|type=string|Path to the video file you wish to process
--outputpath|-o|type=string|Path to the directory where extracted data is stored.
--cpu|-c|action='store_true'|Process using only CPU. Use this flag if your system does not include an NVIDIA GPU, or have CUDA toolkit installed
--skim|-s|taction='store_true'|Skim videos for grade crossing activations, and begin object detection only when an activation is found. Reduces time required for inference, though may not be desireable in all use cases.

## Compilation

For use with the GCTD GUI, this script may be compiled to an executable using Pyinstaller, which may be installed using pip.

If you intend to compile using pyinstaller it is recommended that you create a python virtual environment for the script. This will help ensure that pyinstaller is able to find all necessary dependancies.

If compiling on Windows, you must first uncomment the line 'multiprocessing.freeze_support()' within process_video.py. This may be found immediately after "if __name__ == '__main__':", and is necessary to ensure the compiled executable will run on Windows.

To compile, use the following command:

```
pyinstaller --add-data models;models --add-data deep_sort;deep_sort --exclude-module=torch.distributions --onedir process_video.py
```




## License

MIT
