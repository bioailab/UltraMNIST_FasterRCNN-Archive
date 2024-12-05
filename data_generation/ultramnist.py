import os
import cv2
import random
import string
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torchvision.datasets as datasets
import xml.etree.ElementTree as gfg  
import pickle

from image_metrics import get_iou

class CreateUltraMNIST:
    def __init__(self, root: str, base_data_path: str, n_samples: list = [28000, 28000], 
                    img_size: int = 4000, img_scale_fact: list = [1, 76])->None:
        self.root_path = root
        self.base_data_path = base_data_path
        self.n_samples = n_samples
        self.img_size = img_size
        self.img_scale_fact = img_scale_fact
        self.n_classes = 28
        self.data = None
        self.targets = None

        # check if data exists
        # self.data_exists_flag = self.check_if_data_exists()
        self.sum_list = defaultdict(list)

        for i in range(10):
            for j in range(10):
                for k in range(10):
                    result = [i, j, k]
                    result.sort()
                    if result not in self.sum_list[sum(result)]:
                        self.sum_list[sum(result)].append(result)
                    for u in range(10):
                        result = [i, j, k, u]
                        result.sort()
                        if sum(result)<=self.n_classes and result not in self.sum_list[sum(result)]: self.sum_list[sum(result)].append(result)
                        for v in range(10):
                            result = [i, j, k, u, v]
                            result.sort()
                            if sum(result)<=self.n_classes and result not in self.sum_list[sum(result)]: self.sum_list[sum(result)].append(result)

    def generate_dataset(self):
        # if self.data_exists_flag:
        #     raise Exception('Data already exists, delete the content to download again')

        print('Checking for base dataset, if needed')
        self.get_base_dataset()

        print('Preparing storage locations')
        print("Root Path: ", self.root_path)
        os.mkdir(self.root_path)

        # creating train test and validation folders
        os.mkdir(os.path.join(self.root_path, 'train'))
        os.mkdir(os.path.join(self.root_path, 'valid'))
        os.mkdir(os.path.join(self.root_path, 'test'))
        os.mkdir(os.path.join(self.root_path, 'labels'))

        # limiting the sample per class (spc)
        # train_spc = int(self.n_samples[0] / self.n_classes)
        # valid_spc = int(self.n_samples[1] / self.n_classes)
        train_samples = int(self.n_samples[0]*0.8)
        valid_samples = self.n_samples[0] - train_samples
        train_spc = int(train_samples / self.n_classes)
        valid_spc = int(valid_samples / self.n_classes)
        test_spc = int(self.n_samples[1] / self.n_classes)

        

        print("samples: ", train_spc, valid_spc, test_spc)

        # generating samples
        self._generate_samples(os.path.join(self.root_path, 'train'), train_spc, 'train')
        self._generate_samples(os.path.join(self.root_path, 'valid'), valid_spc, 'valid')
        self._generate_samples(os.path.join(self.root_path, 'test'), test_spc, 'test')

    def _generate_samples(self, data_path, spc, data_type):
        # spc denotes samples per class
        dict_labels = {}
        dict_sum_label = {}
        for num_class in range(self.n_classes):
            combinations = self.sum_list[num_class]
            for i in tqdm(range(spc)):
                labels = combinations[np.random.choice(len(combinations))]
                images = [self.data[self.targets==label][np.random.choice(len(self.data[self.targets==label]))] for label in labels]
                # generate sample
                img, label, b_box = self._generate_one_sample(images, labels)

                # img_dir = os.path.join(data_path, str(label))

                # if not os.path.isdir(img_dir):
                #     os.mkdir(img_dir)
                letters = string.ascii_lowercase
                fname = ''.join(random.choice(letters) for j in range(10))
                im = Image.fromarray(img*255)
                # im.convert('L').save(os.path.join(img_dir, fname+'.jpeg'))
                im.convert('L').save(os.path.join(data_path, fname+'.jpeg'))
                dict_labels[fname] = labels
                dict_sum_label[fname] = label
                # Creating an xml file from the bounding box
                self.xml_file(b_box, data_path, fname)
        with open(os.path.join(self.root_path, f'labels/{data_type}_labels.data'), 'wb') as file: 
            pickle.dump(dict_labels, file)
        with open(os.path.join(self.root_path, f'labels/{data_type}_sum_labels.data'), 'wb') as file: 
            pickle.dump(dict_sum_label, file) 

    def _generate_one_sample(self, images, labels):
        # Saving the bounding box
        b_box = {}

        # creating the background
        img = np.zeros((self.img_size, self.img_size))

        label = 0
        prev_boxes = []
        # print("Inside one_sample: ", labels, images)
        # Add scaled versions of base image into the main image at random locations
        i = 0
        j = 0 # For saving the bounding box
        while i < len(images):
            sub_img = images[i]

            # random sample a resoltion from V-shape distribution
            k = int(np.ceil((self.img_scale_fact[1]-self.img_scale_fact[0])/2))
            prob = np.array([i for i in range(k, 0, -1)] + [i for i in range(1, k)])
            res_fact = np.random.choice(range(self.img_scale_fact[0], self.img_scale_fact[1]), p=prob/prob.sum())

            if res_fact == 1 and np.random.rand()<0.5:
                scaled_simg = sub_img.numpy()
                scaled_simg = cv2.resize(scaled_simg, (14, 14), interpolation=cv2.INTER_NEAREST)
            else: scaled_simg = np.kron(sub_img, np.ones((res_fact,res_fact)))

            # add to img
            sub_len = scaled_simg.shape[0]
            # print("Shape: ", img.shape[0], )
            randx = random.randint(0, img.shape[0]-sub_len)
            randy = random.randint(0, img.shape[0]-sub_len)

            # add to prev_boxes, if overlap with all boxes in prev_boxes is less
            new_box = {}
            new_box = {'x1': randx, 'x2': randx+sub_len-1, 'y1': randy, 'y2': randy+sub_len-1}
            add_flag = self._check_for_low_overlap(new_box, prev_boxes)

            if add_flag:
                img[randx:randx+sub_len, randy:randy+sub_len] += scaled_simg
                
                b_box[j] = {'label': labels[i], 'box': new_box} # saving bounding boxes
                j += 1
                # # Added for creating a box outside the image:
                # img[randx:randx+sub_len, randy] = 1
                # img[randx:randx+sub_len, randy+sub_len] = 1
                # img[randx, randy:randy+sub_len] = 1
                # img[randx+sub_len, randy:randy+sub_len] = 1

                prev_boxes.append(new_box)
                # updating the label
                label += labels[i]
                i += 1

        img[img > 1] = 1
        return img, label, b_box

    def get_base_dataset(self):
        # check if base dataset exists, else download it
        # print("Base data path: ", self.base_data_path)
        self.download_base_flag = False
        if not os.path.exists(self.base_data_path):
            print('Base dataset does not exist at specified path, downloading now...')
            self.download_base_flag = True

        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])

        if self.download_base_flag:            
            mnist_trainset = datasets.MNIST(root=self.base_data_path, train=True, download=True, transform=transform)
        else:
            mnist_trainset = datasets.MNIST(root=self.base_data_path, train=True, download=False, transform=transform)

        self.data = mnist_trainset.data/255.
        self.targets = mnist_trainset.targets

    def _check_for_low_overlap(self, new_box, prev_boxes):
        # if prev_boxes is empty, add this box, so return True
        if not prev_boxes:
            return True

        # if there is atleast one element in prev_boxes
        add_flag = True
        for box in prev_boxes:
            iou = get_iou(new_box, box)
            if iou > 0:
                add_flag = False

        return add_flag
    
    def xml_file(self, b_box, img_dir, fname):
        root = gfg.Element("annotation") 

        folder_elem = gfg.SubElement(root, "folder")
        folder_elem.text = " "
        # root.append (folder_elem) 

        filename_elem = gfg.SubElement(root, "filename")
        filename_elem.text = fname

        path_elem = gfg.SubElement(root, "path")
        path_elem.text = img_dir

        source = gfg.Element("source") 
        root.append (source) 
        
        database = gfg.SubElement(source, "database") 
        database.text = "UltraMNIST"
        
        size = gfg.Element("size") 
        root.append (size) 
        
        width = gfg.SubElement(size, "width") 
        width.text = "4000"
        height = gfg.SubElement(size, "height") 
        height.text = "4000"
        depth = gfg.SubElement(size, "depth") 
        depth.text = "3"

        segmented_elem = gfg.SubElement(root, "segmented")
        segmented_elem.text = "0"
        
        for i in b_box:
            object = gfg.Element("object") 
            root.append (object) 
            
            name = gfg.SubElement(object, "name") 
            name.text = str(b_box[i]['label'])

            pose = gfg.SubElement(object, "pose") 
            pose.text = "Unspecified"
            truncated = gfg.SubElement(object, "truncated") 
            truncated.text = "0"
            difficult = gfg.SubElement(object, "difficult") 
            difficult.text = "0"
            occluded = gfg.SubElement(object, "occluded") 
            occluded.text = "0"

            bndbox = gfg.SubElement(object, "bndbox")
            xmin = gfg.SubElement(bndbox, "xmin")
            xmin.text = str(b_box[i]['box']['x1'])
            xmax = gfg.SubElement(bndbox, "xmax")
            xmax.text = str(b_box[i]['box']['x2'])
            ymin = gfg.SubElement(bndbox, "ymin")
            ymin.text = str(b_box[i]['box']['y1'])
            ymax = gfg.SubElement(bndbox, "ymax")
            ymax.text = str(b_box[i]['box']['y2'])
        
        tree = gfg.ElementTree(root) 

        gfg.indent(tree, '  ')
        filename = os.path.join(img_dir, fname+".xml")
        with open (filename, "wb") as files :
            tree.write(files) 
