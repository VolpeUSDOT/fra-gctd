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

# import random
import warnings
import datetime
import time
import timeit
import platform
import json
import os
import sys
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

# -------------------------------------------------------------
# Configuration settings
# matplotlib.use('Agg')
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
batch_size = 6
num_of_workers = 2
force_video_fps = 0
force_video_width = None
force_video_height = 720

# Deep Sort Config
class DEEPSORT_CONFIG_CLASS():
        REID_CKPT =  "./PhaseTwo/deep_sort/deep/checkpoint/ckpt.t7"
        MAX_DIST =  0.9
        MIN_CONFIDENCE = 0.4
        NMS_MAX_OVERLAP = 0.8
        MAX_IOU_DISTANCE = 0.9
        MAX_AGE = 100
        N_INIT = 3
        NN_BUDGET = 100

DEEPSORT_CONFIG = DEEPSORT_CONFIG_CLASS()
# -------------------------------------------------------------

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    device
except NameError:
    # device was not set to something, we assume CPU to be more compatible. 
    device = "cpu"

device_memory = 0
if(device != 'cpu'):
    device_memory = torch.cuda.get_device_properties(device).total_memory
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True

if __name__ == '__main__':

    # Debug data
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print("Device in use: ",device)
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
        'person', 'train', 'car', 'bus', 'truck', 'motorcycle', 'bicycle'
    ]

    # The label that is used by Deep Sort
    DEEPSORT_LABEL = 'person'

    # LABEL_COLORS = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    # REMINDER: Label colors are Blue, Green, Red (BGR)
    LABEL_COLORS = [[0, 0, 255],[0, 255, 0],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180]]
    
    # Helper functions
    # -----------------------------------------------------------------------------------
    
    def image_resize(org_width, org_height, width = None, height = None):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return org_width, org_height

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(org_height)
            dim = (int(org_width * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(org_width)
            dim = (width, int(org_height * r))

        # return the resized dimensions
        return dim

    def run_model(model, data):
        with torch.no_grad():
            return model(data)

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

    def pil_to_cv(image):
        new_image = np.array(image)
        return new_image[:, :, ::-1].copy()

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

        pred_score = list(pred['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        masks = (pred['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]

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

        return new_masks, new_boxes, new_class, pred_score

    def colour_masks(image, color):
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = color
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
            rgb_mask = colour_masks(masks[i], LABEL_COLORS[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        return img

    def instance_segmentation_visualize_sort(img, masks, boxes, pred_cls, scores, threshold=0.5, rect_th=2, text_size=.75, text_th=2, classname='Object'):
        
        # if there are no objects detected, we still need to notify the SORT object
        if(len(boxes) == 0):
            if classname == DEEPSORT_LABEL:
                sort_boxes = []
            else:
                sort_boxes = sort_trackers[classname].update()
        else:
            if classname == DEEPSORT_LABEL:
                sort_boxes = deep_sort_tracker.update(fix_box_format(boxes),scores,img)
            else:
                sort_boxes = sort_trackers[classname].update(fix_box_format(boxes))
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_color_idx = COCO_INSTANCE_VISIBLE_CATEGORY_NAMES.index(classname)

        for i in range(len(masks)):
            rgb_mask = colour_masks(masks[i], LABEL_COLORS[label_color_idx])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        for i in range(len(sort_boxes)):
            x = (int(sort_boxes[i][0]), int(sort_boxes[i][1]))
            y = (int(sort_boxes[i][2]), int(sort_boxes[i][3]))
            cv2.rectangle(img,x, y,color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, str(classname) + ' #' + str(int(sort_boxes[i][4])), (int(sort_boxes[i][0]), int(sort_boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)
        
        return img

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
    # Deal with command line arguments.
    parser = argparse.ArgumentParser(description='Process some video files using Machine Learning!')
    parser.add_argument('--outputpath', '-o',   action='store',     required=False,     default='../temp/fraoutput',        help='Path to the directory where extracted data is stored.')
    parser.add_argument('--inputpath',  '-i',   action='store',     required=False,     default='E:/FRA/ramsey/20180418/Ch02_20180418000000_20180418235959_1a.avi',    help='Path to the extracted video frames in JPG format.')
    args = parser.parse_args()

    # Load the object detection model
    model_road = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model_road.eval()

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

    output_filename = str(Path(args.outputpath) / Path(str(input_filename) + '-processed.mp4'))
    video, audio, info, video_idx = video_clips.get_clip(0)
    
    org_frame_height = video[0].size()[0]
    org_frame_width = video[0].size()[1]
    frame_width, frame_height = image_resize(org_frame_width, org_frame_height,force_video_width, force_video_height)
    
    videoSize = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    

    print('Saving video as: ', output_filename)
    video_out = cv2.VideoWriter(output_filename, fourcc, video_output_fps, (videoSize))

    framecount = 0
    extracted_data = []
    
    for chunk in range(video_clips.num_clips()):
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
        
        # print(str(size(sys.getsizeof(image_stack.storage()), system=si)))

        # load batch onto GPU / or register in CPU
        image_stack = image_stack.to(device)

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
            video_data['frame_number'] = str(framecount)
            video_data['frame_timestamp'] = str(video_timestamp)

            # setup our images
            org_image1 = pil_to_cv(org_image_stack[n])

            # create JSON output
            # road_output = []

            if ([i for i in road_annotation['scores'] if i >= segmentation_threshold]):
                road_masks, road_boxes, road_labels, road_scores = parse_seg_prediction(road_annotation, segmentation_threshold)
                # print(road_scores, road_labels, road_boxes)

                # Process predictions for SORT tracking
                # road_img, road_sort_boxes = instance_segmentation_visualize_sort(org_image1, road_annotation, segmentation_threshold, boxes_thickness, boxes_text_size, boxes_text_thickness)
                # road_output.append(road_scores)
                # road_output.append(road_boxes)
            # else:
                # mot_tracker.update(np.empty((0, 5)))
            road_img = org_image1

            # fix the colors..
            # road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB)
            
            boxes_by_label = {}
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
                road_img = instance_segmentation_visualize_sort(road_img, collected_masks, collected_boxes, collected_labels, collected_scores, segmentation_threshold, classname=label)
                new_line = [collected_boxes]
                boxes_by_label[label] = new_line
            
            # save the video to disk
            video_out.write(road_img)

            # write out the JSON
            combined_output = {}
            combined_output.update({'video':video_data})
            combined_output.update({'tracking':boxes_by_label})
            # combined_output.update({'roadway':road_output})
            dataoutput.write(str(combined_output) + '\n')
            dataoutput.flush()

            framecount += 1
            n += 1
        
        # clean up memory
        del road_img
        del road_annotations

    dataoutput.close()
    video_out.release() 

    # Wrap up (might want to remove this once integrated into Electron)
    print(' ')
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    # For memory debugging
    #print(torch.cuda.memory_summary())