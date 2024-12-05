import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image, ImageDraw
from torchvision.transforms import RandomAffine, CenterCrop, Compose, Resize

from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--root_path", type=str, default="/data/", help="path to the root directory where data is stored")
ap.add_argument("--dir_name", type=str, default="ultramnist_2800", help="directory name")
ap.add_argument("--n_shapes", type=int, default=3, help = "number of each kind of (traingle and circle) shapes")
ap.add_argument("--checker_size", type=int, default=16, help = "size of checkers")

# ap.add_argument("--checker_low", type=int, default=12, help = "minimum size of the checker")
# ap.add_argument("--checker_high", type=int, default=24, help = "maximum size of the checker")
# ap.add_argument("--shapes_high", type=int, default=5, help = "maximum number of each kind (traingle and circle) shapes")
args = ap.parse_args()

# train_img_paths = glob(os.path.join(args.root_path, "ultramnist/train/*/*.jpeg"))
# valid_img_paths = glob(os.path.join(args.root_path, "ultramnist/val/*/*.jpeg"))
dir_name = args.dir_name
test_img_paths = glob(os.path.join(f"{os.getcwd()}{args.root_path}", f"{dir_name}/test/*.jpeg"))

# npath = os.path.join(args.root_path, "ultramnist_noised")
npath = os.path.join(f"{os.getcwd()}{args.root_path}", dir_name)
os.makedirs(npath, exist_ok=True)
# os.makedirs(os.path.join(npath, "train"), exist_ok=True)
# os.makedirs(os.path.join(npath, "val"), exist_ok=True)


trfm = Compose([RandomAffine((-45, 45), scale=(0.8, 1.2), shear=50), CenterCrop(2000), Resize(4000)])

n_shapes = args.n_shapes
checker_size = args.checker_size
checker_low = 12
checker_high = 24 
shapes_high = 5


os.makedirs(os.path.join(npath, f'test_{n_shapes}_{checker_size}'), exist_ok=True)

def img_resize(path):
    LocalProcRandGen = np.random.RandomState()
    # checker_size = LocalProcRandGen.randint(checker_low, high=checker_high)

    def randTriangle():
        x1, y1 = LocalProcRandGen.randint(0, high=4000), LocalProcRandGen.randint(0, high=4000)
        x2, y2 = LocalProcRandGen.randint(0, high=4000), LocalProcRandGen.randint(0, high=4000)
        x3, y3 = LocalProcRandGen.randint(0, high=4000), LocalProcRandGen.randint(0, high=4000)
        return [(x1, y1), (x2, y2), (x3, y3)]

    def randCircle():
        x = LocalProcRandGen.randint(0, high=4000)
        y = LocalProcRandGen.randint(0, high=4000)
        r = LocalProcRandGen.randint(50, high=1000)
        return [(x - r, y - r), (x + r, y + r)]

    bgr = (LocalProcRandGen.randint(0, high=2, size=(checker_size, checker_size)) * 255).astype("uint8")
    bgr = cv2.resize(bgr, (4000, 4000), interpolation=cv2.INTER_NEAREST)
    bgr = Image.fromarray(bgr)
    bgr = trfm(bgr)

    im = Image.fromarray(np.zeros((4000, 4000)))
    draw = ImageDraw.Draw(im)

    for i in range(n_shapes): # LocalProcRandGen.randint(0, high=shapes_high)
        draw.ellipse(randCircle(), fill=255, outline=255)
    for i in range(n_shapes): # LocalProcRandGen.randint(0, high=shapes_high)
        draw.polygon(randTriangle(), fill=255, outline=255)

    bgr = np.array(bgr)
    im = np.array(im).astype("uint8")

    _, bgr = cv2.threshold(bgr, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    bgr = (np.sign(bgr - im) * 255).astype("uint8")

    path_split = path.split("/")
    image = cv2.imread(path)
    _, image = cv2.threshold(image[:, :, 2], 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, bgr = cv2.threshold(bgr, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image = (np.sign(bgr - image) * 255).astype("uint8")
    # save_path = os.path.join(npath, path_split[-3], path_split[-2])
    save_path = os.path.join(npath, f'test_{n_shapes}_{checker_size}')
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, path_split[-1]), image)


# Parallel(n_jobs=mp.cpu_count(), backend="multiprocessing")(delayed(img_resize)(path) for path in tqdm(train_img_paths))
# Parallel(n_jobs=mp.cpu_count(), backend="multiprocessing")(delayed(img_resize)(path) for path in tqdm(valid_img_paths))
Parallel(n_jobs=mp.cpu_count(), backend="multiprocessing")(delayed(img_resize)(path) for path in tqdm(test_img_paths))