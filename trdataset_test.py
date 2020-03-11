from bbox_tr import plot_bbox
from PairFileDataset import PairFileDataset
from matplotlib import pyplot as plt
import numpy as np

ds = PairFileDataset(r'D:\cvImageSamples\lan4\SampleImages','.bmp')

image, target = ds[100]

bboxes = target['bboxes']
cids = target['labels']
plot_bbox(np.asarray(image), bboxes, labels=cids,absolute_coordinates=False)
plt.show()
