##################################################
## Script to generate eye/glance tracking videos using 
## the pre-trained Retinaface computer vision model
##################################################
## MIT License
##################################################
## Author: Robert Rittmuller
## Copyright: Copyright 2020, Volpe National Transportation Systems Center
## Credits: <add reference to SORT & Deep SORT here>
## License: MIT
## Version: 0.0.5
## Mmaintainer: Robert Rittmuller
## Email: robert.rittmuller@dot.gov
## Status: Active Development
##################################################

# import random
import warnings
import datetime
import time
import timeit
import platform
import json
import csv
import os
import sys

## 'Unused' imports, required to make pyinstaller included them during compilation
import scipy
import av
from scipy.stats import statlib
from scipy.integrate import _odepack
from scipy.interpolate import _fitpack
from scipy.optimize import linesearch
import scipy.optimize
import numpy.core.multiarray
import torch.onnx.symbolic_opset7
from scipy.optimize import minpack2
import scipy.linalg._fblas as fblas
import scipy.sparse.linalg.isolve.iterative

import cv2
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.video_utils import VideoClips
import argparse
import numpy as np
from pathlib import Path
from hurry.filesize import size, si
from sort import *
from deep_sort import build_tracker
from utils import get_event_detections, image_resize, pil_to_cv, parse_seg_prediction, instance_segmentation_visualize_sort, get_model_instance_segmentation, instance_grade_segmentation_visualize, detect_object_overlap

# -------------------------------------------------------------
# Configuration settings
computer_device = "cuda:0"
warnings.filterwarnings("ignore")
segmentation_threshold = 0.75
INPUT_SIZE = 225
new_size = INPUT_SIZE, INPUT_SIZE
sort_max_age = 2
sort_min_hits = 5
sort_ios_threshold = .1
boxes_thickness = 1
boxes_text_size = 1
boxes_text_thickness = 1
batch_size = 24
num_of_workers = 4
force_video_fps = 0
force_video_width = None
force_video_height = 480

# grade / right-of-way segmentation model settings
grade_num_classes = 2
GRADE_CATEGORY_NAMES = [
    '__background__', 'GradeCrossing', 'RightOfWay'
]
GRADE_LABEL_COLORS = [[0, 0, 255],[0, 255, 0],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180]]

# Roadway features model settings
COCO_INSTANCE_VISIBLE_CATEGORY_NAMES = [
    'person', 'train', 'car', 'bus', 'truck', 'motorcycle', 'bicycle'
]

# The label that is used by Deep Sort
DEEPSORT_LABEL = 'person'

# LABEL_COLORS = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
# REMINDER: Label colors are Blue, Green, Red (BGR)
LABEL_COLORS = [[0, 255, 0],[0, 255, 0],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[0, 0, 255]]

# Deep Sort Config
class DEEPSORT_CONFIG_CLASS():
        REID_CKPT =  "./deep_sort/deep/checkpoint/ckpt.t7"
        MAX_DIST =  0.9
        MIN_CONFIDENCE = 0.4
        NMS_MAX_OVERLAP = 0.8
        MAX_IOU_DISTANCE = 0.9
        MAX_AGE = 100
        N_INIT = 3
        NN_BUDGET = 100

DEEPSORT_CONFIG = DEEPSORT_CONFIG_CLASS()
# -------------------------------------------------------------

if __name__ == '__main__':

    # Necessary to run compiled on Windows
    #multiprocessing.freeze_support()

    # Deal with command line arguments.
    parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
    parser.add_argument('--outputpath', '-o',   action='store',         required=False,     default='../../temp/fraoutput',         help='Path to the directory where extracted data is stored.')
    parser.add_argument('--inputpath',  '-i',   action='store',         required=False,     default='/mnt/ml_data/FRA/sourcevideos/ramsey/20180418/Ch02_20180418000000_20180418235959_1e.avi',    help='Path to the extracted video frames in JPG format.')
    parser.add_argument('--cpu', '-c',          action='store_true',    required=False,     default=False,                           help='Toggles CPU-only mode.')

    args = parser.parse_args()

    # Detect if we have a GPU available
    if(args.cpu == False):
        device = torch.device(computer_device if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
    try:
        device
    except NameError:
        # device was not set to something, we assume CPU to be more compatible. 
        device = "cpu"

    device_memory = 0
    device_name = 'CPU'
    if(device != 'cpu'):
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        device_memory = torch.cuda.get_device_properties(device).total_memory
        # device_memory_reserved = torch.cuda.memory_reserved(device) 
        # device_memory_allocated = torch.cuda.memory_allocated(device)
        # device_memory_free = device_memory_reserved - device_memory_allocated  # free inside reserved
        
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.enabled = True

    # Debug data
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print("Device in use: ",device, device_name)
    print("Device memory: ", str(size(device_memory, system=si)))

    # detect platform for FFMPEG
    # if platform.system() == 'Windows':
    #     # path to ffmpeg bin
    #     FFMPEG_PATH = 'ffmpeg.exe'
    #     FFPROBE_PATH = 'ffprobe.exe'
    # else:
    #     # path to ffmpeg bin
    #     default_ffmpeg_path = '/usr/local/bin/ffmpeg'
    #     default_ffprobe_path = '/usr/local/bin/ffprobe'
    #     FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) else '/usr/bin/ffmpeg'
    #     FFPROBE_PATH = default_ffprobe_path if path.exists(default_ffprobe_path) else '/usr/bin/ffprobe'

    # -------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------------------------

    def run_model(model, data):
        with torch.no_grad():
            return model(data)

    def create_csv_from_json():
        with open(Path(args.outputpath) / Path(str(input_filename + '-data.json'))) as json_output:
            data = json.load(json_output)
        csv_file = open(Path(args.outputpath) / Path(str(input_filename + '-data.csv')), 'w')
        writer = csv.writer(csv_file)
        header = ["frame_number", "frame_timestamp", "label", "bbox"]
        writer.writerow(header)
        for frame in data:
            videoData = frame["video"]
            trackingData = frame["tracking"]
            for label in trackingData:
                if len(trackingData[label][0]) > 0:
                    for bbox in trackingData[label]:
                        row = []
                        row.append(videoData["frame_number"])
                        row.append(videoData["frame_timestamp"])
                        row.append(label)
                        row.append(str(bbox))
                        writer.writerow(row)
        csv_file.close()

    # def ffmpeg_getinfo(vid_file_path):

    #     if type(vid_file_path) != str:
    #         raise Exception('Give ffprobe a full file path of the video')
    #         return

    #     command = ["ffprobe",
    #             "-loglevel",  "quiet",
    #             "-print_format", "json",
    #              "-show_format",
    #              "-show_streams",
    #              vid_file_path
    #              ]

    # def ffmpeg_process_video(video_file_name, deinterlace=False):
    #     # FFMPEG commands for video frame extraction
        
    #     video_filename, video_file_extension = path.splitext(path.basename(video_file_name))
    #     video_metadata = ffmpeg_getinfo(video_file_name)
    #     num_seconds = int(float(video_metadata['streams'][0]['duration']))
    #     num_of_frames = int(float(video_metadata['streams'][0]['duration_ts']))
    #     video_width = int(video_metadata['streams'][0]['width'])
    #     video_height = int(video_metadata['streams'][0]['height'])
        
    #     if deinterlace == True:
    #         deinterlace = 'yadif'
    #     else:
    #         deinterlace = ''

    #     ffmpeg_command = [
    #         FFMPEG_PATH, '-i', video_file_name,
    #         '-vf', 'fps=' + args.fps, '-r', args.fps, '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
    #         '-hide_banner', '-loglevel', '0', '-vf', ffmpeg_deinterlace, '-f', 'image2pipe', '-vf', 'scale=' + frame_size, '-'
    #     ]

    #     image_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4*1024*1024)

    # def instance_segmentation_visualize_sort(img, predictions, threshold=0.5, rect_th=1, text_size=1, text_th=1):
    #     masks, boxes, pred_cls = parse_seg_prediction(predictions, threshold)
    #     fixed_boxes = fix_box_format(boxes)
    #     if(fixed_boxes != []):
    #         sort_boxes = mot_tracker.update(fixed_boxes)
    #     else:
    #         sort_boxes = mot_tracker.update(np.empty((0, 5)))
        
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     for i in range(len(masks)):
    #         rgb_mask = colour_masks(masks[i])
    #         img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

    #     for i in range(len(sort_boxes)):
    #         x = (int(sort_boxes[i][0]), int(sort_boxes[i][1]))
    #         y = (int(sort_boxes[i][2]), int(sort_boxes[i][3]))
    #         cv2.rectangle(img,x, y,color=(0, 255, 0), thickness=rect_th)
    #         cv2.putText(img,'Object #' + str(int(sort_boxes[i][4])), (int(sort_boxes[i][0]), int(sort_boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)
    #     return img, sort_boxes

    def get_sort_id(box_x, sort_boxes):
        for box in sort_boxes:
            if(int(box_x) == int(box[0])):
                return int(box[4])

    # -------------------------------------------------------------

    # Load the object detection model
    model_road = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model_road.eval()

    # load the grade / right-of-way model
    grade_model = get_model_instance_segmentation(grade_num_classes)
    if(device != 'cpu'):
        grade_model.load_state_dict(torch.load('models/gctd_grade-row.pt'))
    else: 
        grade_model.load_state_dict(torch.load('models/gctd_grade-row.pt', map_location=torch.device('cpu')))
        
    grade_model.eval()
    grade_model.to(device)

    # load the activity classification model
    # model_class = torch.load(args.modelfile)
    # model_class = model_class.to(device)
    # model_class.eval()

    # load the labels for the classification model
    # labels = load_labels(args.labelfile)

    # setup regular SORT tracking
    sort_trackers = {}
    for label in COCO_INSTANCE_VISIBLE_CATEGORY_NAMES:
        sort_trackers[label] = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_ios_threshold)

    # setup deep sort tracking for people label
    deep_sort_tracker = build_tracker(DEEPSORT_CONFIG, use_cuda=device)

    # set start time
    start = timeit.default_timer()

    # load up the data
    videopath = Path(args.inputpath)
    print('Indexing input video...')
    video_clips = VideoClips([str(videopath)], clip_length_in_frames=batch_size, frames_between_clips=batch_size, num_workers=num_of_workers)

    # let's grab some video metadata
    if(force_video_fps == 0):
        video_output_fps = int(video_clips.video_fps[0])

    # tabular and video data output
    input_filename, ext = os.path.splitext(os.path.basename(args.inputpath))
    dataoutput = open(Path(args.outputpath) / Path(str(input_filename + '-data.json')), 'w')
    dataoutput.write('[\n')
    dataoutput.flush()

    event_output = open(Path(args.outputpath) / Path(str(input_filename + '-events.csv')), 'w')
    event_writer = csv.writer(event_output)
    header = ["frame_number", "frame_timestamp", "label", "bbox"]
    event_writer.writerow(header)

    output_filename = str(Path(args.outputpath) / Path(str(input_filename) + '-processed.mp4'))
    video, audio, info, video_idx = video_clips.get_clip(0)
    
    org_frame_height = video[0].size()[0]
    org_frame_width = video[0].size()[1]
    frame_width, frame_height = image_resize(org_frame_width, org_frame_height,force_video_width, force_video_height)
    
    videoSize = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    

    print('Saving video as: ', output_filename)
    video_out = cv2.VideoWriter(output_filename, fourcc, video_output_fps, (videoSize))

    grade_segmented = False
    framecount = 0
    extracted_data = []
    
    for chunk in range(video_clips.num_clips()):
        progress = str(round((chunk / video_clips.num_clips() * 100), 1)) + "% complete"
        sys.stdout.write("Processing: %s   \r" % (progress) )
        sys.stdout.flush()
        video, audio, info, video_idx = video_clips.get_clip(chunk)
        video_output_fps = int(info['video_fps'])

        image_stack = []
        org_image_stack = []
        for frame in video:
            org_image = frame.permute(2,0,1)
            org_image = transforms.ToPILImage()(org_image)
            org_image = transforms.Resize((frame_height,frame_width))(org_image)
            new_image = transforms.ToTensor()(org_image)
            new_image = new_image.numpy()
            image_stack.append(new_image)
            org_image_stack.append(org_image)

        image_stack = np.asarray(image_stack)
        image_stack = torch.from_numpy(image_stack)

        # load batch onto GPU / or register in CPU
        image_stack = image_stack.to(device)

        # get the grade segmentation masks *only on the first batch*
        if(grade_segmented == False):
            grade_annotations = run_model(grade_model, image_stack)
            grade_segmented = True

            # upload the model since we are done with it
            # del grade_model
        

        # get the model predictions / annotations
        road_annotations = run_model(model_road, image_stack)

        # drop batch out of memory
        del image_stack
        if(device != 'cpu'):
            torch.cuda.empty_cache()

        n = 0
        for road_annotation in road_annotations:

            # get timestamps for each frame
            if framecount == 0:
                video_timestamp = 0
            else:
                video_timestamp = framecount / video_output_fps

            video_timestamp = datetime.timedelta(seconds=video_timestamp)
            video_data = {}
            video_data["frame_number"] = str(framecount)
            video_data["frame_timestamp"] = str(video_timestamp)

            # setup our images
            org_image1 = pil_to_cv(org_image_stack[n])

            # create JSON output
            is_annotation = False
            if ([i for i in road_annotation['scores'] if i >= segmentation_threshold]):
                road_masks, road_boxes, road_labels, road_scores = parse_seg_prediction(road_annotation, segmentation_threshold, COCO_INSTANCE_VISIBLE_CATEGORY_NAMES)
                is_annotation = True

            road_img = org_image1

            # fix the colors..
            # road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB)
            
            boxes_by_label = {}
            if is_annotation == True:
                for label in COCO_INSTANCE_VISIBLE_CATEGORY_NAMES:
                    collected_boxes = []
                    collected_masks = []
                    collected_labels = []
                    collected_scores = []
                    idx = 0
                    for road_label in road_labels:
                        if(label == road_label):
                            collected_boxes.append(road_boxes[idx])
                            collected_masks.append(road_masks[idx])
                            collected_labels.append(road_labels[idx])
                            collected_scores.append(road_scores[idx])
                        idx += 1
                    road_img, grade_masks = instance_grade_segmentation_visualize(road_img, grade_annotations[0], GRADE_CATEGORY_NAMES, GRADE_LABEL_COLORS)
                    event_detections = get_event_detections(collected_masks, grade_masks,  GRADE_CATEGORY_NAMES, label)
                    road_img = instance_segmentation_visualize_sort(road_img, collected_masks, collected_boxes, collected_labels, collected_scores, COCO_INSTANCE_VISIBLE_CATEGORY_NAMES, event_detections, LABEL_COLORS, DEEPSORT_LABEL, sort_trackers, deep_sort_tracker, grade_masks, GRADE_CATEGORY_NAMES ,segmentation_threshold, classname=label)
                    new_line = [collected_boxes]
                    boxes_by_label[label] = new_line
            
            # save the video to disk
            video_out.write(road_img)

            # write out the JSON
            combined_output = {}
            combined_output.update({"video":video_data})
            combined_output.update({"tracking":boxes_by_label})
            # combined_output.update({'roadway':road_output})
            # End our JSON line with a comma, unless this is the last one
            lineEnd = ',\n'
            if (n == len(road_annotations)-1 and chunk == video_clips.num_clips() - 1):
                lineEnd = '\n'
            # Replace characters in str output to make this valid JSON
            outputline = (str(combined_output) + lineEnd).replace("'", '"')
            outputline = outputline.replace("(", "\"(")
            outputline = outputline.replace(")", ")\"")
            dataoutput.write(outputline)
            dataoutput.flush()

            framecount += 1
            n += 1
        
        # clean up memory
        del road_img
        del road_annotations

    dataoutput.write('\n]')
    dataoutput.flush()
    dataoutput.close()
    video_out.release() 
    create_csv_from_json()
    # Wrap up (might want to remove this once integrated into Electron)
    print(' ')
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))