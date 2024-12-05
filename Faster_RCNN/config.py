import torch

dir_name = 'ultramnist_2800'
BATCH_SIZE = 16 # increase / decrease according to GPU memeory
NUM_EPOCHS = 50 # number of epochs to train for

RESIZE_TO = 4000 # 416 # resize the image for training and transforms
NUM_WORKERS = 0

print("BATCH_SIZE: ", BATCH_SIZE, " RESIZE_TO: ", RESIZE_TO, " NUM_EPOCHS:", NUM_EPOCHS)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
# TRAIN_DIR = 'data/Uno Cards.v2-raw.voc/train'
TRAIN_DIR = f'/UltraMNIST_FasterRCNN/data/{dir_name}/train'
# validation images and XML files directory
# VALID_DIR = 'data/Uno Cards.v2-raw.voc/valid'
VALID_DIR = f'/UltraMNIST_FasterRCNN/data/{dir_name}/valid'

# DIR_TEST
DIR_TEST = f'/UltraMNIST_FasterRCNN/data/{dir_name}/test'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
]

NUM_CLASSES = len(CLASSES)
print("Number of classes: ", NUM_CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'outputs'