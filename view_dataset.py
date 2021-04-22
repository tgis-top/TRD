
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.optim as optim


from TRD import TRD, TRDLoss
from bbox_tr import plot_bbox_tr

from PairFileDataset import PairFileDataset

transform = transforms.Compose([
    # transforms.Resize([net.image_size,net.image_size]),
    transforms.ToTensor()])
    
trainset = PairFileDataset(r'D:\ImageSamples\DOTA v1.5\train\images','.png',transform= transform)

image, target = trainset[0]
bboxes = target['bboxes']
cids = target['labels']
image = image*255    # unnormalize
image = image.numpy()
image = np.transpose(image, (1, 2, 0))
plot_bbox_tr(image, bboxes, labels=cids,absolute_coordinates=True)
plt.show()