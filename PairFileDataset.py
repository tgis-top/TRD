import numpy as np
from PIL import Image
from skimage import io, transform
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from os import listdir
from os.path import join

__all__ = (
    "PairFileDataset",\
    )

class PairFileDataset(Dataset):
    def __init__(self, root_dir, img_ext = '.png', transform=None):
        self.root_dir = root_dir
        self.img_ext = img_ext
        self.files = []
        self.transform = transform
        for i in os.listdir(root_dir):
            image_id,image_ext = os.path.splitext(i)
            if image_ext.lower() == '.txt':
                image_path = join(root_dir,image_id+img_ext)
                if os.path.exists(image_path):
                    self.files.append(image_id)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        image_path = join(self.root_dir,image_id+self.img_ext)
        label_path = join(self.root_dir,image_id+'.txt')
        image = Image.open(image_path)
        # image = io.imread(image_path)          
        label_file = open(label_path,'r')
        bboxes = []
        labels = []
        for line in label_file:
            parts = line.split()
            if(len(parts) > 6):
                bboxes.append([float(x) for x in parts[1:7]])
                labels.append(int(parts[0]))
        target = {}
        target["bboxes"] = np.array(bboxes)
        target["labels"] = np.array(labels)
        if self.transform:
            image = self.transform(image) 
        return image, target

