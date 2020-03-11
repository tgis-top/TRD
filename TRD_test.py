import torch
import torchvision.transforms as transforms
import numpy as np
import resnet_backbone
from bbox_tr import plot_bbox
from matplotlib import pyplot as plt
from PIL import Image
from TRD import TRD
from polynms import nms_poly

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    num_classes = 1
    image_size = 416
    bboxw_range = [(48,144),(24,72),(12,36)]

    transform = transforms.Compose([
        transforms.ToTensor(),        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = TRD(bboxw_range,image_size,num_classes)    
    net.load_state_dict(torch.load('./param/TRD_final.pth'))
    net.to(device)

    image = Image.open(r'D:\cvImageSamples\lan4\test\image_170227_027.JPG')

    with torch.no_grad():
        pred = net.bigdetect(image,transform,score_thresh=0.6,iou_thresh=0.5,cd_thresh=0.1,device=device)
        # pred,_ = nms_poly(pred,1,0.4)
        if pred is not None:
            plot_bbox(np.asarray(image), pred)
            plt.show()