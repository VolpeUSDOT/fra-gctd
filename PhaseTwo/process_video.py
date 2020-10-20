##################################################
## Script to generate eye/glance tracking videos using 
## the pre-trained Retinaface computer vision model
##################################################
## MIT License
##################################################
## Author: Robert Rittmuller
## Copyright: Copyright 2020, Volpe National Transportation Systems Center
## Credits: <add reference to SORT here>
## License: MIT
## Version: 0.0.1
## Mmaintainer: Robert Rittmuller
## Email: robert.rittmuller@dot.gov
## Status: Active Development
##################################################

import random
import warnings
import time
import timeit
import platform
import json
import os
import sys
from os import listdir
import csv
import cv2
import torch
import torchvision
from torchvision import datasets, models, transforms
import argparse
import numpy as np
from pathlib import Path
from retinaface.pre_trained_models import get_model
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.style as style
from retinaface.utils import vis_annotations
from sort import *

# -------------------------------------------------------------
# Configuration settings
# matplotlib.use('Agg')
warnings.filterwarnings("ignore")
segmentation_threshold = 0.70
INPUT_SIZE = 225
new_size = INPUT_SIZE, INPUT_SIZE
sort_max_age = 2
sort_min_hits = 5
sort_ios_threshold = .1
boxes_thickness = 2
boxes_text_size = 1
boxes_text_thickness = 2
# -------------------------------------------------------------

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Debug data
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("Device in use: ",device)

# FFMPEG command configuration
if platform.system() == 'Windows':
    # path to ffmpeg bin
    FFMPEG_PATH = 'ffmpeg.exe'
    FFPROBE_PATH = 'ffprobe.exe'
else:
    # path to ffmpeg bin
    default_ffmpeg_path = '/usr/local/bin/ffmpeg'
    default_ffprobe_path = '/usr/local/bin/ffprobe'
    FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) else '/usr/bin/ffmpeg'
    FFPROBE_PATH = default_ffprobe_path if path.exists(default_ffprobe_path) else '/usr/bin/ffprobe'

# ffmpeg_command = [
#     FFMPEG_PATH, '-i', video_file_name,
#     '-filter_complex', '"[0:0]vflip,crop=2400:800:1000:0;[0:5]crop=1200:1344:1500,transpose=1"', '-map', '0:0', '-map', '0:5' '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
#     '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', '-'
# ]

# -------------------------------------------------------------
# Annotation Labels / Classes

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COCO_INSTANCE_VISIBLE_CATEGORY_NAMES = [
    '__background__','person', 'train'
]

# Helper functions

def load_labels(filePath):
    label_file = open(filePath, 'r')
    file_data = label_file.readlines()
    labels = []
    for line in file_data:
        labelset = []
        idx,label = line.split(':')
        labelset.append(idx)
        labelset.append(label)
        labels.append(labelset)
    return labels

def save_csv(outputpath, data):
    with open(outputpath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for data_row in data:
            csv_writer.writerow(data_row)

def get_file_list(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def isodd(num):
    mod = num % 2
    if mod > 0:
        return True
    else:
        return False

def parse_seg_prediction(pred, threshold):

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    labels = pred_class.copy()
    new_masks = []
    new_class = []
    new_boxes = []
    n = 0
    for label in pred_class:
        if label in COCO_INSTANCE_VISIBLE_CATEGORY_NAMES:
            new_masks.append(masks[n])
            new_boxes.append(pred_boxes[n])
            new_class.append(pred_class[n])
        n += 1

    return new_masks, new_boxes, new_class

def colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[1]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def fix_box_format(boxes):
    new_boxes = []
    for box in boxes:
        line = []
        for value in box:
            for subvalue in value:
                line.append(subvalue)
        new_boxes.append(line)
    return np.array(new_boxes)

def instance_segmentation_visualize(img, predictions, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls = parse_seg_prediction(predictions, threshold)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    return img

def instance_segmentation_visualize_sort(img, predictions, threshold=0.5, rect_th=1, text_size=1, text_th=1):
    masks, boxes, pred_cls = parse_seg_prediction(predictions, threshold)
    fixed_boxes = fix_box_format(boxes)
    if(fixed_boxes != []):
        sort_boxes = mot_tracker.update(fixed_boxes)
    else:
        sort_boxes = mot_tracker.update(np.empty((0, 5)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

    for i in range(len(sort_boxes)):
        x = (int(sort_boxes[i][0]), int(sort_boxes[i][1]))
        y = (int(sort_boxes[i][2]), int(sort_boxes[i][3]))
        cv2.rectangle(img,x, y,color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,'Object #' + str(int(sort_boxes[i][4])), (int(sort_boxes[i][0]), int(sort_boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)
    return img, sort_boxes

def get_sort_id(box_x, sort_boxes):
    for box in sort_boxes:
        if(int(box_x) == int(box[0])):
            return int(box[4])

# -------------------------------------------------------------
# Deal with command line arguments.
parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
parser.add_argument('--outputpath', '-o',   action='store',     required=False,     default='../temp/fraoutput',        help='Path to the directory where extracted data is stored.')
parser.add_argument('--imagespath',  '-i',  action='store',     required=False,     default='../temp/fraframes',    help='Path to the extracted video frames in JPG format.')
# parser.add_argument('--labelfile',  '-l',   action='store',     required=False,     default='./models/labels.txt',       help='Path to label file.')
# parser.add_argument('--modelfile',  '-m',   action='store',     required=False,     help='Path to saved model file.',   default='./models/saved_model_squeezenet.pt')

args = parser.parse_args()

# Load the segmentation model
model_road = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
# model_road = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
# model_road = torchvision.models.segmentation.fcn_resnet50(pretrained=True).to(device)
model_road.eval()

# load the activity classification model
# model_class = torch.load(args.modelfile)
# model_class = model_class.to(device)
# model_class.eval()

# load the labels for the classification model
# labels = load_labels(args.labelfile)

# setup regular SORT tracking
mot_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_ios_threshold)

# get the list of images we will process
images_list = []
images_list = get_file_list(args.imagespath, 'jpg')

dataoutput = open(Path(args.outputpath) / Path('data.json'), 'w')

# set start time
start = timeit.default_timer()

# it = iter(images_list)
frame_num = 1
for image in images_list:

    # setup our two images
    source_image1 = image
    # source_image2 = next(it)

    image_path = Path(args.imagespath) / Path(source_image1)
    target_image1 = cv2.imread(str(image_path))
    org_image1 = target_image1

    # image_path = Path(args.imagespath) / Path(source_image2)
    # target_image2 = cv2.imread(str(image_path))
    # org_image2 = target_image2

    # process image for activiity classification
    # target_image2_class = transforms.ToPILImage()(org_image2)
    # target_image2_class = transforms.Resize(new_size)(target_image2_class)
    # target_image2_class = torchvision.transforms.functional.to_tensor(target_image2_class)
    # target_image2_class = target_image2_class.to(device)
    # target_image2_class = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(target_image2_class)
    # target_image2_class = target_image2_class.unsqueeze(0)

    # run models on the processed data
    # activitiy_classification = model_class(target_image2_class)
    
    target_image1 = torchvision.transforms.functional.to_tensor(target_image1)
    target_image1 = target_image1.to(device)
    target_image1 = target_image1.unsqueeze(0)
    
    # get model predictions
    road_annotation = model_road(target_image1)

    # create JSON output
    road_output = []

    if ([i for i in road_annotation[0]['scores'] if i >= segmentation_threshold]):        
        road_masks, road_boxes, road_scores = parse_seg_prediction(road_annotation, segmentation_threshold)

        # Process predictions for SORT tracking
        road_img, road_sort_boxes = instance_segmentation_visualize_sort(org_image1, road_annotation, segmentation_threshold, boxes_thickness, boxes_text_size, boxes_text_thickness)
        road_output.append(road_scores)
        road_output.append(road_boxes)
    else:
        mot_tracker.update(np.empty((0, 5)))
        road_img = org_image1

    # save the frame to disk
    new_img_filename = Path(args.outputpath) / Path('frame_' + str(frame_num).zfill(5) + '.jpg')
    cv2.imwrite(str(new_img_filename), road_img,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # write out the JSON
    combined_output = {}
    combined_output.update({'tracking':road_sort_boxes})
    combined_output.update({'roadway':road_output})
    dataoutput.write(str(combined_output) + '\n')
    dataoutput.flush()

    # setup our output visuals
    # style.use('default')
    # plt.axis('off')
    # plt.tight_layout(pad=0.1)
    # plt.rcParams['text.color'] = 'white'

    # fig, final_visual = plt.subplots(2,1)
    # fig.patch.set_facecolor('xkcd:dark grey')
    # final_visual[0].set_xticks([])
    # final_visual[0].set_yticks([])
    # final_visual[1].set_xticks([])
    # final_visual[1].set_yticks([])
    # plt.title('Glance Classification: ' + str(max_classification_label), fontsize=7)
    # if(face_annotation[0]['score'] == -1):
    #    final_visual[0].imshow(org_image2)
    # else:
    #    final_visual[0].imshow(vis_annotations(target_image2, face_annotation))
    # plt.title('Roadway Objects Detected: ' + str(len(road_scores)), fontsize=10)
    # final_visual[1].imshow(road_img)
    # plt.savefig(Path(args.outputpath) / Path('frame_' + str(frame_num).zfill(5) + '.jpg'),bbox_inches='tight', dpi=250)
    # plt.close()
    frame_num += 1

    # clean up the GPU cache
    # torch.cuda.empty_cache()

dataoutput.close()
# Wrap up
print(' ')
stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))