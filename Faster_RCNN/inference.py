import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import pickle
from tqdm import tqdm

from model import create_model

from config import (
    NUM_CLASSES, DEVICE, CLASSES, DIR_TEST
)

file_name = 'test'
dir_name = 'ultramnist_2800'

DIR_TEST = f'/UltraMNIST_FasterRCNN/data/{dir_name}/{file_name}'

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(f'/UltraMNIST_FasterRCNN/Faster_RCNN/outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# directory where all the images are present
print(f"{DIR_TEST}/*.jpeg")
test_images = glob.glob(f"{DIR_TEST}/*.jpeg")
print(f"Test instances: {len(test_images)}")

# directory to save the inferenced images
if not os.path.exists(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/images/{file_name}/"):
    os.makedirs(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/images/{file_name}/")
if not os.path.exists(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/prediction_labels/"):
    os.makedirs(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/prediction_labels/")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.4 

# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0
dict_labels = {}
# print(len(test_images))
for i in tqdm(range(len(test_images))): # len(test_images)
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    # print("test_images: ", test_images[i])
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        labels = []
        boxes = []
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            labels.append(class_name)
            boxes.append(box.tolist())

            # If you want to create the predicted images with bounding boxes 
            color = COLORS[CLASSES.index(class_name)]
            color = [255, 0, 0] # Red
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 4)
            cv2.putText(orig_image, class_name, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 
                        4, lineType=cv2.LINE_AA) #0.7
        dict_labels[image_name] = {"labels": labels, "boxes": boxes}
    cv2.imwrite(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/images/{file_name}/{image_name}.jpeg", orig_image)


with open(f"/UltraMNIST_FasterRCNN/Faster_RCNN/inference_outputs/prediction_labels/prediction_{file_name}.data", 'wb') as file: 
    pickle.dump(dict_labels, file) 
print('TEST PREDICTIONS COMPLETE')
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")