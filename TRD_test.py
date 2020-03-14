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
    # pretrained = './param/TRD_4900.pth'
    # pretrained_dict = torch.load(pretrained)
    # model_dict = net.state_dict()  
    # #将pretrained_dict里不属于model_dict的键剔除掉 
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}         
    # # 更新现有的model_dict 
    # model_dict.update(pretrained_dict)         
    # # 加载我们真正需要的state_dict 
    # net.load_state_dict(model_dict)
    # torch.save(model_dict, './param/TRD_4900.pth')
    net.load_state_dict(torch.load('./lan4/TRD_1000.pth'))
    net.to(device)

    image = Image.open(r'D:\cvImageSamples\lan4\test\Untitled 1.jpg')
    net.eval()
    with torch.no_grad():
        pred = net.bigdetect(image,transform,217,score_thresh=0.5,iou_thresh=0.5,cd_thresh=0.1,device=device)
        # pred,_ = nms_poly(pred,1,0.4)
        if pred is not None:
            plot_bbox(np.asarray(image), pred, scores=pred[:,7])
            plt.show()