import pickle
import glob as glob
import os
from xml.etree import ElementTree as et

file_name = 'test'

iou_threshold = 0.5

from config import (
    DIR_TEST, CLASSES
)

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Read all the test files:

image_paths = glob.glob(f"{DIR_TEST}/*.jpeg")
all_images = [image_path.split(os.path.sep)[-1] for image_path in image_paths]
all_images = sorted(all_images)

data_actual = {}
# capture the image name and the full image path
for idx in range(len(all_images)): # len(all_images)
    image_name = all_images[idx]
    image_path = os.path.join(DIR_TEST, image_name)

    annot_filename = image_name[:-5] + '.xml'
    annot_file_path = os.path.join(DIR_TEST, annot_filename)

    boxes = []
    labels = []
    tree = et.parse(annot_file_path)
    root = tree.getroot()
    # labels_box_list = {}
    # labels = []
    # box coordinates for xml files are extracted and corrected for image size given
    j = 0
    for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        
        # labels.append(CLASSES.index(member.find('name').text))
        labels.append(member.find('name').text)
        # labels_box = {}

        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)

        # # labels['label'] = CLASSES.index(member.find('name').text)
        # labels_box['label'] = member.find('name').text
        # labels_box['box'] = [xmin, ymin, xmax, ymax]

        # labels_box_list[j] = labels_box
        # j += 1

        boxes.append([xmin, ymin, xmax, ymax])
    
    # data_actual[image_name.split('.')[0]] = labels_box_list
    data_actual[image_name.split('.')[0]] = {"labels": labels, "boxes": boxes}

result_addr = f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/prediction_labels/prediction_{file_name}.data"
file = open(result_addr, 'rb') 
data_pred = pickle.load(file) 

total_labels = 0
correct_pred = 0

labels_wise_prediction = {}
for i in range(10):
    labels_wise_prediction[i] = {'actual': 0, 'pred': 0}

keys_pred = list(data_pred.keys())
for key in data_actual:
    # Not required for full data
    actual = data_actual[key]
    act_labels = actual['labels']
    total_labels += len(act_labels)
    if key not in keys_pred:
        continue
    prediction = data_pred[key]
    pred_labels = prediction['labels']
    for j in range(len(act_labels)):
        label = act_labels[j]
        labels_wise_prediction[int(label)]['actual'] += 1
        indices = [i for i in range(len(pred_labels)) if pred_labels[i] == label]
        if len(indices) > 0:
            for ind in indices:
                iou = bb_iou(actual['boxes'][j], prediction['boxes'][ind])
                if iou > iou_threshold:
                    correct_pred += 1
                    labels_wise_prediction[int(label)]['pred'] += 1
                    break


print("The total number of labels in the dataset: ", total_labels, 
      "\n",
      "The total number of correct predictions made: ", correct_pred,
      "\n")

print("Prediction for each label in the format \nDigit Actual Prediction")
for key in labels_wise_prediction:
    print(key, labels_wise_prediction[key]['actual'], labels_wise_prediction[key]['pred'])

