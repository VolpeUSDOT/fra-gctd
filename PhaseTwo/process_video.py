from ctypes import resize
import warnings
import datetime
import timeit
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
from utils import get_event_detections, image_resize, pil_to_cv, parse_seg_prediction, instance_segmentation_visualize_sort, get_model_instance_segmentation, instance_grade_segmentation_visualize, detect_object_overlap, update_sort

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
batch_size = 32
num_of_workers = 4
force_video_fps = 0
force_video_width = None
force_video_height = 480

# scene model detection thresholds
scene_detection_base_th = 3                 # 10 frames before an event is detected
scene_non_detection_label = 'noactivation'  # label indicates no event detected

# grade / right-of-way segmentation model settings
grade_num_classes = 2
GRADE_CATEGORY_NAMES = [
    '__background__', 'GradeCrossing', 'RightOfWay'
]
SCENE_LABEL_NAMES = ["activation","noactivation"]
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
    parser.add_argument('--inputpath',  '-i',   action='store',         required=False,     default='/mnt/ml_data/FRA/sourcevideos/Other/16465283-427E-4A87-8329-AE77087BA33A.MP4',    help='Path to the extracted video frames in JPG format.')
    parser.add_argument('--cpu',        '-c',   action='store_true',    required=False,     default=False,                           help='Toggles CPU-only mode.')
    parser.add_argument('--force',      '-f',   action='store_true',    required=False,     default=False,                           help='Force object-based trespass detection on.')

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
    print("Device in use: ",device)
    print("Device memory: ", str(size(device_memory, system=si)))

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

    def get_sort_id(box_x, sort_boxes):
        for box in sort_boxes:
            if(int(box_x) == int(box[0])):
                return int(box[4])


    def detect_skim_events(model, image_stack):
        event_detected_th = scene_detection_base_th
        event_detected = False

        # get the scene predictions
        scene_preds = run_model(model,image_stack)
        
        for pred in scene_preds:
            scene_max_value, scene_max_index = torch.max(pred,0)
            max_scene_classification_label = SCENE_LABEL_NAMES[scene_max_index]
            if(max_scene_classification_label != scene_non_detection_label):
                event_detected_th -= 1
                if(event_detected_th < 1):
                    event_detected = max_scene_classification_label
                    # print(scene_max_value, scene_max_index, max_scene_classification_label)
            else:
                if(event_detected_th < scene_detection_base_th):
                    event_detected_th += .5
        
        return event_detected

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

    # load the scene classification model
    if(device != 'cpu'):
        model_scene = torch.load('models/saved_model_squeezenet.pt')
    else: 
        model_scene = torch.load('models/saved_model_squeezenet.pt', map_location=torch.device('cpu'))
    model_scene = model_scene.to(device)
    model_scene.eval()

    # setup regular SORT tracking
    sort_trackers = {}
    event_trackers = {}
    activationGroups = []
    activated = False
    currentAct = {}
    for label in COCO_INSTANCE_VISIBLE_CATEGORY_NAMES:
        sort_trackers[label] = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_ios_threshold)
        event_trackers[label] = []
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
    header = ["Object ID", "label", "event_type", "Start Time", "End Time", "Gate Descent Start", "Gate Ascent End", "Train Present?", "TAT"]
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
    videoskim = True
    scn_event = False
    
    for chunk in range(video_clips.num_clips()):
        progress = str(round((chunk / video_clips.num_clips() * 100), 1)) + "% complete - " + str(scn_event) + "     "
        sys.stdout.write("Processing: %s   \r" % (progress) )
        sys.stdout.flush()
        video, audio, info, video_idx = video_clips.get_clip(chunk)
        video_output_fps = int(info['video_fps'])

        obj_image_stack = []
        scn_image_stack = []
        org_image_stack = []

        for frame in video:
            org_image = frame.permute(2,0,1)
            org_image = transforms.ToPILImage()(org_image)
            new_image = transforms.Resize(new_size)(org_image)
            new_image = transforms.ToTensor()(new_image)
            new_image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(new_image)
            new_image = new_image.numpy()
            scn_image_stack.append(new_image)

        scn_image_stack = np.asarray(scn_image_stack)
        scn_image_stack = torch.from_numpy(scn_image_stack)

        # load batch onto GPU / or register in CPU
        scn_image_stack = scn_image_stack.to(device)

        # perform scene detection
        scn_event = detect_skim_events(model_scene, scn_image_stack)
        if(scn_event):
            videoskim = False
            scn_image_stack = []

        if(videoskim == False):
            # create a batch for the object detection model to process
            for frame in video:
                org_image = frame.permute(2,0,1)
                org_image = transforms.ToPILImage()(org_image)
                org_image = transforms.Resize((frame_height,frame_width))(org_image)
                new_image = transforms.ToTensor()(org_image)
                new_image = new_image.numpy()
                obj_image_stack.append(new_image)
                org_image_stack.append(org_image)
            
            obj_image_stack = np.asarray(obj_image_stack)
            obj_image_stack = torch.from_numpy(obj_image_stack)

            # load batch onto GPU / or register in CPU
            obj_image_stack = obj_image_stack.to(device)
            
            # get the grade segmentation masks *only on the first batch*
            if(grade_segmented == False):
                grade_annotations = run_model(grade_model, obj_image_stack)
                grade_segmented = True

                # upload the model since we are done with it
                del grade_model

            # get the model predictions / annotations
            road_annotations = run_model(model_road, obj_image_stack)

            # drop batch out of memory
            del obj_image_stack
            del scn_image_stack
            if(device != 'cpu'):
                torch.cuda.empty_cache()

            n = 0

            # get timestamps for each frame
            if framecount == 0:
                video_timestamp = 0
            else:
                video_timestamp = framecount / video_output_fps
            video_timestamp = datetime.timedelta(seconds=video_timestamp)
            if not activated and scn_event == 'activation':
                currentAct['start'] = video_timestamp
                activated = True
            if scn_event =="noactivation" and activated:
                currentAct['end'] = video_timestamp
                activationGroups.append(currentAct)
                currentAct = {}
                activated = False

            for road_annotation in road_annotations:

                video_data = {}
                video_data["frame_number"] = str(framecount)
                video_data["frame_timestamp"] = str(video_timestamp)
                video_data["frame_timestamp_raw"] = video_timestamp

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
                        road_img, grade_masks, grade_cls = instance_grade_segmentation_visualize(road_img, grade_annotations[0], GRADE_CATEGORY_NAMES, GRADE_LABEL_COLORS)
                        sort_boxes = update_sort(road_img, collected_boxes, collected_scores, sort_trackers, deep_sort_tracker, DEEPSORT_LABEL, classname=label)
                        event_detections = get_event_detections(collected_masks, collected_boxes, grade_masks,  grade_cls, sort_boxes, event_trackers, video_data["frame_timestamp_raw"], label)
                        # output all events in current frame
                        # for evt in event_detections:
                        #     if evt != False and label != "train":
                        #         row = []
                        #         row.append(video_data["frame_number"])
                        #         row.append(video_data["frame_timestamp"])
                        #         row.append(label)
                        #         row.append(evt)
                        #         event_writer.writerow(row)
                        road_img = instance_segmentation_visualize_sort(road_img, collected_masks, sort_boxes, collected_boxes, collected_labels, collected_scores, COCO_INSTANCE_VISIBLE_CATEGORY_NAMES, event_detections, LABEL_COLORS, DEEPSORT_LABEL, sort_trackers, deep_sort_tracker, grade_masks, GRADE_CATEGORY_NAMES ,segmentation_threshold, classname=label)
                        new_line = [collected_boxes]
                        boxes_by_label[label] = new_line
                
                # save the video to disk
                video_out.write(road_img)

                # write out the JSON
                # We no longer need the raw timestamp here
                del video_data["frame_timestamp_raw"]
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

    # If the gate is still 'activated' at the end of the video, close out the activation
    if activated:
        currentAct['end'] = video_timestamp
        activationGroups.append(currentAct)
    dataoutput.write('\n]')
    dataoutput.flush()
    dataoutput.close()

    

    # Output Events
    # Find any trains
    # these are not events, but we will report if a train is present for a given event
    train_events = event_trackers['train']

    def is_train_present(event):
        for tevt in train_events:
            # Event starts during train event
            if event["start_time"] <= tevt["stop_time"] and event["start_time"] >= tevt["start_time"]:
                return tevt
            # Event ends during train event
            if event["stop_time"] <= tevt["stop_time"] and event["stop_time"] >= tevt["start_time"]:
                return tevt
            # Train event is contained within event
            if tevt["start_time"] >= event["start_time"] and tevt["stop_time"] <= event["stop_time"]:
                return tevt
        return None

    def is_during_activation(event):
        for act in activationGroups:
            # Event starts during train event
            if event["start_time"] <= act["end"] and event["start_time"] >= act["start"]:
                return act
            # Event ends during train event
            if event["stop_time"] <= act["end"] and event["stop_time"] >= act["start"]:
                return act
            # Train event is contained within event
            if act["start"] >= event["start_time"] and act["end"] <= event["stop_time"]:
                return act
        return None

    for label in COCO_INSTANCE_VISIBLE_CATEGORY_NAMES:
        if label != 'train':
            for event in event_trackers[label]:
                activation_start, activation_end = '', ''
                if event["evt_type"] == 'GradeCrossing':
                    activ = is_during_activation(event)
                    if not activ:
                        continue
                    else:
                        activation_start = str(activ["start"])
                        activation_end = str(activ["end"])
                row = []
                row.append(event["id"])
                row.append(event["label"])
                row.append(event["evt_type"])
                row.append(str(event["start_time"]))
                row.append(str(event["stop_time"]))
                row.append(activation_start)
                row.append(activation_end)
                train_event = is_train_present(event)
                if (train_event is not None):
                    row.append("Yes")
                    row.append(train_event["start_time"])
                else:
                    row.append("No")
                    row.append("")
                event_writer.writerow(row)
    event_output.close()
    video_out.release() 
    create_csv_from_json()
    # Wrap up (might want to remove this once integrated into Electron)
    print(' ')
    print(activationGroups)
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))