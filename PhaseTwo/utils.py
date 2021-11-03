##################################################
## Utility script to support image segmentation tasks
##################################################
## MIT License
##################################################
## Author: Robert Rittmuller
## Copyright: Copyright 2021, Volpe National Transportation Systems Center
## Credits: 
## License: MIT
## Version: 0.0.1
## Mmaintainer: Robert Rittmuller
## Email: robert.rittmuller@dot.gov
## Status: Active Development
##################################################

# import modules
import cv2
import csv
import numpy as np
from numpy.core.arrayprint import format_float_scientific
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

def is_in_area(detections, threshold=0.001):
    for detection in detections:
        if(detection[2] >= threshold):
            return detection[0]
    
    return False

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def detect_object_overlap(detection_masks, detection_masks_labels, object_masks, object_masks_labels):
    # compare masks from passed lists to see if we get high overlap scores for any objects
    object_index = 0
    for object_mask in object_masks:
        object_label = object_masks_labels[object_index]
        object_index += 1

        detection_index = 0
        for detection_mask in detection_masks:
            overlap_score = dice_metric(detection_mask,object_mask)
            # print(object_label, overlap_score, detection_masks_labels[detection_index])
            detection_index += 1

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

def parse_seg_prediction(pred, threshold, COCO_INSTANCE_VISIBLE_CATEGORY_NAMES):

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
    i = 0
    for box in boxes:
        line = []
        for value in box:
            for subvalue in value:
                line.append(subvalue)
        line.append(i)
        i += 1
        new_boxes.append(line)
    return np.array(new_boxes)

def instance_segmentation_visualize(img, predictions, LABEL_COLORS, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls = parse_seg_prediction(predictions, threshold)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i], LABEL_COLORS[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    return img

def update_sort(img, boxes, scores, sort_trackers, deep_sort_tracker, DEEPSORT_LABEL, classname='Object'):
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
    return sort_boxes
        
def instance_segmentation_visualize_sort(img, masks, sort_boxes, boxes, pred_cls, scores, labels, event_detections,
                                        LABEL_COLORS, DEEPSORT_LABEL, sort_trackers, 
                                        deep_sort_tracker, detection_masks, detection_masks_labels, event_in_progress=False,
                                        threshold=0.5, rect_th=2, text_size=.50, text_th=2, classname='Object'):

        label_color_idx = labels.index(classname)
        for i in range(len(masks)):
            if(masks[i].ndim > 1):
                violation_detection = event_detections[i]
                if(violation_detection != False):
                    if(event_in_progress == True):
                        rgb_mask = colour_masks(masks[i], LABEL_COLORS[7])
                    else:
                        if(violation_detection == 'RightOfWay'):
                            rgb_mask = colour_masks(masks[i], LABEL_COLORS[7])
                        else:
                            rgb_mask = colour_masks(masks[i], LABEL_COLORS[3])
                else:
                    rgb_mask = colour_masks(masks[i], LABEL_COLORS[label_color_idx])
                
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        for i in range(len(sort_boxes)):
            x = (int(sort_boxes[i][0]), int(sort_boxes[i][1]))
            y = (int(sort_boxes[i][2]), int(sort_boxes[i][3]))
            # if(is_in_area(detection_zones_scores)):
            cv2.rectangle(img,x, y,color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, str(classname) + ' #' + str(int(sort_boxes[i][4])), (int(sort_boxes[i][0]), int(sort_boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)
        
        return img

def get_event_detections(masks, boxes, detection_masks, detection_masks_labels, sort_boxes, event_trackers, frame_timestamp, classname='Object'):
    all_events = [False] * len(masks)
    evts = event_trackers[classname]
    for i in range(len(masks)):
        if(masks[i].ndim > 1):
            detection_index = 0
            detection_zones_scores = []
            for detection_zone in detection_masks:
                detection = detection_masks_labels[detection_index], classname, dice_metric(masks[i],detection_zone)
                detection_zones_scores.append(detection)
                detection_index += 1
            all_events[i] = is_in_area(detection_zones_scores, 0.01)
            #print("ID is " + str(sort_boxes[i][4]))
            if all_events[i]:
                object_id = -1
                for j in range(len(sort_boxes)):
                    sbox = sort_boxes[j]
                    if i == sbox[5]:
                        object_id = sbox[4]
                        break
                if object_id == -1:
                    # We aren't able to track this event. 
                    event = {}
                    event["id"] = -1
                    event["start_time"] = frame_timestamp
                    event["stop_time"] = frame_timestamp
                    event["label"] = classname
                    event["evt_type"] = all_events[i]
                else:
                    found = False
                    for e in evts:
                        if e["id"] == object_id:
                            found = True
                            e["stop_time"] = frame_timestamp
                            break
                    if not found:
                        event = {}
                        event["id"] = object_id
                        event["start_time"] = frame_timestamp
                        event["stop_time"] = frame_timestamp
                        event["label"] = classname
                        event["evt_type"] = all_events[i]
                        evts.append(event)

    return all_events          

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def parse_grade_seg_prediction(pred, threshold, CATEGORY_NAMES):
    pred_score = list(pred['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]

    masks = []

    if len(pred['masks']) > 1:
        masks = (pred['masks']>0.5).squeeze().detach().cpu().numpy()
    else:
        masks.append((pred['masks'][0][0]>0.5).detach().cpu().numpy())
    
    pred_class = [CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().numpy())]
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
        if label in CATEGORY_NAMES:
            new_masks.append(masks[n])
            new_boxes.append(pred_boxes[n])
            new_class.append(pred_class[n])
        n += 1
    return new_masks, new_boxes, new_class

def instance_grade_segmentation_visualize(img, predictions, CATEGORY_NAMES, LABEL_COLORS, threshold=0.00001, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls = parse_grade_seg_prediction(predictions, threshold, CATEGORY_NAMES)
    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i], LABEL_COLORS[pred_cls[i]])
        # rgb_mask = rgb_mask.transpose(2,0,1)
        img = cv2.addWeighted(img, 1, rgb_mask, .02, 0)
        # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    return img, masks, pred_cls