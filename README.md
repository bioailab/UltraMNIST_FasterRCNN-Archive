### See [the author's](https://rohit102497.github.io/) git repository here: https://github.com/Rohit102497/UltraMNIST_FasterRCNN

# An UltraMNIST classification benchmark to train CNNs for very large images

This repository contains the experiment corresponding to section **Implications of Noise** in the above paper.

In this repository, we create a 2800 test and train Ultramnist dataset to determine the performance of FasterRCNN as the amount of complexity (noise) increase in the images. Following steps are done for this. 
1. A noise free dataset of 2800 images for each test and train set is created.
2. Varying level of noise is added to the above created test set.
    - Constant checkerboard size of 4 is fixed and the number of shape pairs is varied from 1 to 5.
    - Fixed the number of shape pairs to 1 and varied the checkerboard sizes to 4, 8, 12, 16, 20, and 24.

## Configuration
Please install necessary packages from requirements.txt as follows: 
```
pip install -r requirements.txt
```

## Data Creation
### Folders
1. `data_generation` - Contains code to create the dataset.
2. `data` - Stores the data. 

### Code

#### Creating noise free data
We create the noise free data by: 
```
python data_generation/create_ultramnist.py --root_path /data/ --n_samples 2800
```

**Note: You might get an error in the cv2.dnn package as AttributeError: module 'cv2.dnn' has no attribute 'DictValue'. If you get that error, please go to the "/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py" file and comment that line.**

Here: 
- `--root_path`: path to the root directory where the generated data will be stored. This is relative to the current directory.
- `--n_samples`: number of train and test samples.

The created dataset is saved in `data/ultramnist_2800/` folder. It contains four subfolders:
- `labels`: contains the label information of each image. This folder is not required for the experiment done here.
- `test`: contains the test images and its corresponding .xml file with information about each digits in the images.
- `train`: train images with format same as test.
- `valid`: validation images with format same as test.

#### Add noise
Add noise by the below command:
```
python data_generation/add_noise.py --root_path /data/ --dir_name ultramnist_2800 --n_shapes 1 --checker_size 4
```

Here:
- `--root_path`: path to the root directory where data is stored.
- `--dir_name`: name of the data directory.
- `--n_shapes`: number of each kind of (traingle and circle) shapes. Values used: 1, 2, 3, 4, 5.
- `checker_size`: size of checkerboard patterns. Values used: 4, 8, 12, 16, 20, 24. 

The created dataset is saved in the `data/ultramnist_2800/test_1_4` folder, where 1 represent the `n_shapes` and  4 represent the `checker_size`. It only contains the .jpeg images.

## Faster-RCNN Code
Closely followed the tutorial provided in (https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/), to implement Faster RCCN.

The code for Faster_RCCN can be found at `Faster_RCNN` folder. 

### Training the model
To train the model, run the following code:
```
python Faster_RCNN/train.py
```

The default parameters are already set in the `Faster_RCNN/config.py` file, like:
- `dir_name`: 'ultramnist_2800'
- `BATCH_SIZE`: 16
- `NUM_EPOCHS`: 50

Other parameters can also be set in the `Faster_RCNN/train.py` file, like:
- `lr`: Set learning rate to 0.0001.

Training the model, saves" two 4 files in `Faster_RCNN/outputs`:
- `best_model.pth`: Contains the trained model based on the best performance on validation set.
- `last_model.pth`: Contains the trained model till last epoch.
- `train_loss.png`: Graph of training loss of the model.
- `val_loss.png`: Graph of validation loss of the model.

### Inference
To perform inference, run the following code:
```
python Faster_RCNN/inference.py
```

The parameters to set in the `Faster_RCNN/inference.py` file are:
- `file_name`: The name of the test file to inference on. Example: 'test', 'test_1_4', ...
- `dir_name`: 'ultramnist_2800'

This saves the results in the `Faster_RCNN/inference_outputs` folder. This folder contains two sub folders:
- `images`: Contains all the inferenced image with bounding box and the predicted digits mentioned on top of the boc
- `prediciton_labels`: Contains all the prediction labels for each image. This will be used to select the top labels of digits with higher iou.

### Performance
To find the total number of correct digit predictions and also the total number of correct individual digit predictions, run the following code:
```
python Faster_RCNN/performance.py
```

The parameters to set in the `Faster_RCNN/performance.py` file are:
- `file_name`: The name of the test file to check performance on. Example: 'test', 'test_1_4', ...

