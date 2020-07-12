import os
import sys

import multiprocessing  

import argparse


import torch
import torchvision.transforms as transforms
import numpy as np
import resnet_backbone
from bbox_tr import plot_bbox
from matplotlib import pyplot as plt
from PIL import Image
from TRD import TRD
from polynms import nms_poly


def predit_image(net, 
                 device,
                 transform,
                 img_path, 
                 overlap,
                 score_thresh,
                 iou_thresh,
                 cd_thresh,
                 show=False,
                 save=True):
    image = Image.open(img_path)
    net.eval()
    with torch.no_grad():
        pred = net.bigdetect(image,
                             transform,
                             overlap,
                             score_thresh=score_thresh,
                             iou_thresh=iou_thresh,
                             cd_thresh=cd_thresh,
                             device=device)
        if pred is not None:
            if(show):
                plot_bbox(np.asarray(image), pred, scores=pred[:,7], labels=pred[:,6])
                plt.show()
            if(save):
                image_id,_ = os.path.splitext(img_path)
                lbl_file = open(image_id+'.txt', 'w')
                for _, bbox in enumerate(pred):
                    lbl_file.write(
                        str(int(bbox[6]))  + " " 
                        + " ".join(['%.7f'%a for a in bbox[:4]]) + " " 
                        + str(int(bbox[4])) + " " 
                        + '%.7f'%bbox[5] + " " 
                        + '%.7f'%bbox[7] + '\n')
                lbl_file.close()



def get_args():
    parser = argparse.ArgumentParser(description='Predict Objects by TRD on input image or dir',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='M', type=str, default=r"E:\SourceCode\Python\pytorch_test\resnet50-19c8e357.pth",
                        help='TRD model path', dest='model')       
    parser.add_argument('-i', '--input', metavar='IN', type=str, default=r'D:\cvImageSamples\lan4\test\四尾栅藻 (4).JPG',
                        help='Filename or dir of input images',dest='image_path')

    parser.add_argument('-iz', '--image-size', metavar='IZ', type=int, default=416,
                        help='Network input image size', dest='image_size')
    parser.add_argument('-ie', '--image-ext', metavar='IE', type=str, default='.bmp',
                        help='Image extension name, must provided when input path is dir', dest='image_ext')
    parser.add_argument('-o', '--overlap', metavar='O', type=int, default=172,
                        help='Overlap of spliting image to fit network input', dest='overlap')
    parser.add_argument('-c', '--num-classes', metavar='C', type=int, default=1,
                        help='Number of classes', dest='num_classes')

    parser.add_argument('-st', '--score-thresh', metavar='ST', type=float, default=0.51,
                        help='Score threshold', dest='score_thresh')
    parser.add_argument('-it', '--iou-thresh', metavar='IT', type=float, default=0.3,
                        help='IOU threshold', dest='iou_thresh')
    parser.add_argument('-ct', '--cen-dis-thresh', metavar='CT', type=float, default=0.1,
                        help='Box center distance threshold', dest='cd_thresh')

    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    args = get_args()
    
    bboxw_range = [(48,144),(24,72),(12,36)]
    net = TRD(bboxw_range,args.image_size,args.num_classes)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载预训练的resnet参数
    pretrained_dict = torch.load(args.model)
    model_dict = net.state_dict()  
    #将pretrained_dict里不属于model_dict的键剔除掉 
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}         
    # 更新现有的model_dict 
    model_dict.update(pretrained_dict)         
    # 加载我们真正需要的state_dict 
    net.load_state_dict(model_dict)

    # net.load_state_dict(torch.load(args.model))
    net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor()])

    if(os.path.isfile(args.image_path)):
        predit_image(net,
                     device,
                     transform,
                     args.image_path,
                     args.overlap,
                     score_thresh=args.score_thresh,
                     iou_thresh=args.iou_thresh,
                     cd_thresh=args.cd_thresh,
                     show=True,
                     save=True)
    elif (os.path.isdir(args.image_path)):
        for i in os.listdir(args.image_path):
            image_id,image_ext = os.path.splitext(i)
            if image_ext.lower() == args.image_ext:
                image_path = os.path.join(args.image_path,i)
                predit_image(net,
                             device,
                             transform,
                             image_path,
                             args.overlap,
                             score_thresh=args.score_thresh,
                             iou_thresh=args.iou_thresh,
                             cd_thresh=args.cd_thresh,
                             show=False,
                             save=True)