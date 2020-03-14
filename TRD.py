import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import resnet_backbone
from bbox_tr import plot_bbox,bbox_tr_get_wh
from PairFileDataset import PairFileDataset
from PIL import Image
from polyiou import iou_poly
from polynms import nms_poly


class TRD(nn.Module):
    def __init__(self, bboxw_range, image_size=416, num_classes=20, pretrained=None):  
        super(TRD, self).__init__()   
        self.num_classes = num_classes  
        self.bboxw_range = bboxw_range
        self.image_size = image_size
        self.backbone = resnet_backbone.resnet50(pretrained)

        self.fpn_layer2_1 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[2], self.backbone.feature_planes[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[1]),
            nn.ReLU(inplace=True)
        )

        self.fpn_layer2_2 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[1], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone.feature_planes[0], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU()
        )

        self.fpn_layer1_1 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[2], self.backbone.feature_planes[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone.feature_planes[1], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU(inplace=True)
        )

        self.fpn_layer1_2 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[0], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU()
        )

        self.fpn_layer0_1 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[1], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU(inplace=True)
        )

        self.fpn_layer0_2 = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[0], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.feature_planes[0]),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.output = nn.Sequential(
            nn.Conv2d(self.backbone.feature_planes[0], self.backbone.feature_planes[0], kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone.feature_planes[0], 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + 2 + 1 + (0 if num_classes==1 else num_classes), kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        assert x.shape[-1] == self.image_size and x.shape[-2] == self.image_size

        ft0,ft1,ft2 = self.backbone(x)

        ft2_1 = self.fpn_layer2_1(ft2)        
        ft2_2 = self.fpn_layer2_2(ft2_1)

        ft2_1 = self.upsample(ft2_1)
        ft1_1 = torch.cat((ft1,ft2_1),dim=1)
        ft1_1 = self.fpn_layer1_1(ft1_1)
        ft1_2 = self.fpn_layer1_2(ft1_1)

        ft1_1 = self.upsample(ft1_1)
        ft0_1 = torch.cat((ft0,ft1_1),dim=1)
        ft0_1 = self.fpn_layer0_1(ft0_1)
        ft0_2 = self.fpn_layer0_2(ft0_1)

        x0 = self.output(ft0_2)
        x1 = self.output(ft1_2)
        x2 = self.output(ft2_2)

        return (x2,x1,x0)

    def __parse_output(self, output, bboxw, score_thresh):
        shape = output.shape
        obj_list = []
        ft_size = shape[-1]
        grid_size = self.image_size / ft_size
        for i in range(ft_size):
            for j in range(ft_size):
                if output[0,7,i,j] < score_thresh:
                    continue
                bbox = [0.]*8
                bbox[0] = (j + output[0,0,i,j].item())*grid_size
                bbox[1] = (i + output[0,1,i,j].item())*grid_size
                bbox[2] = output[0,2,i,j].item()*self.image_size
                bbox[3] = output[0,3,i,j].item()*self.image_size
                bbox[4] = 0 if output[0,5,i,j] > output[0,6,i,j] else 1
                bbox[5] = output[0,4,i,j].item()
                if self.num_classes > 1:
                    bbox[6] = output[0,8:,i,j].argmax().item()
                else:
                    bbox[6] = 0
                bbox[7] = output[0,7,i,j].item()
                w,_ = bbox_tr_get_wh(bbox)
                if w < bboxw[0] or w > bboxw[1]:
                    continue
                else:
                    obj_list.append(bbox)
                
        return obj_list

    def detect(self, x, score_thresh = 0.7):
        x2,x1,x0 = self.forward(x)
        obj_list = []
        obj_list.extend(self.__parse_output(x2,self.bboxw_range[0],score_thresh))
        obj_list.extend(self.__parse_output(x1,self.bboxw_range[1],score_thresh))
        obj_list.extend(self.__parse_output(x0,self.bboxw_range[2],score_thresh))

        return np.array(obj_list)
    
    def bigdetect(self, image, transform, overlap, score_thresh = 0.7, iou_thresh = 0.5, cd_thresh = 0.1, device = None):
        iw, ih = image.size
        assert iw >= self.image_size and ih > self.image_size
        step = self.image_size - overlap
        # - 5 表示4个像素及以内的目标就不检测了
        byc = (ih - overlap + step - 5)/step
        bxc = (iw - overlap + step - 5)/step
        byc = int(byc)
        bxc = int(bxc)
        uls = []
        for by in range(byc):
            upper = by*step
            lower = upper + self.image_size
            if lower > ih:
                lower = ih
                upper = ih - self.image_size
            uls.append([upper,lower])
        lrs = []
        for bx in range(bxc):
            left = bx*step
            right = left + self.image_size
            if right > iw:
                right = iw
                left = iw - self.image_size
            lrs.append([left,right])
        final_pred = None
        for by in range(byc):
            for bx in range(bxc):
                box = (lrs[bx][0], uls[by][0], lrs[bx][1], uls[by][1])
                subimage = image.crop(box)
                tsubimage = transform(subimage)
                tsubimage = tsubimage.unsqueeze(0)
                if device is not None:
                    tsubimage = tsubimage.to(device)
                pred = self.detect(tsubimage,score_thresh)
                if pred.size > 0:
                    pred[:,0] = pred[:,0] + box[0]
                    pred[:,1] = pred[:,1] + box[1]
                    final_pred = pred if final_pred is None else np.vstack((final_pred,pred))
        
        if final_pred is not None:
            final_pred,_ = nms_poly(final_pred,self.num_classes,iou_thresh,cd_thresh)
        
        return final_pred



        

class TRDLoss(nn.Module):

    def __init__(self,bboxw_range,image_size = 416):
        super(TRDLoss,self).__init__()
        self.bboxw_range = bboxw_range
        self.image_size = image_size
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
        # 正例置信度
        self.score_loss_p_log = 0.0            
        # 负例置信度
        self.score_loss_n_log = 0.0         
        # 范围框数值部分
        self.bboxv_loss_log = 0.0 
        # 范围框符号部分
        self.bboxs_loss_log = 0.0 
        # 类别标签
        self.label_loss_log = 0.0 

    def forward(self, outputs, targets):
        # 置信度 MSE 加平衡系数    
        # 以下loss只对有目标的部分计算
        # 范围框框数值部分 Smooth L1
        # 范围框符号部分 SoftMax + CrossEntropy
        # 类别部分 SoftMax + CoressEntropy 如果只有一个类别就没有本部分loss
        bboxw_range = self.bboxw_range
        batch_size = outputs[0].shape[0]
        num_classes = outputs[0].shape[1] - 8
        # 特征图像素对应输入图像的尺寸
        grid_size = []
        # 创建目标张量
        targets_ = []
        # 各特征图是否对应目标像素地数量
        pix_p_count = torch.tensor([0,0,0],device=outputs[0].device,dtype=outputs[0].dtype)
        pix_n_count = torch.tensor([0,0,0],device=outputs[0].device,dtype=outputs[0].dtype)
        for j in range(3):
            target_ = {}
            shape = list(outputs[j].shape)
            grid_size.append(self.image_size/shape[-1])
            pix_n_count[j] = shape[-1]*shape[-2]*batch_size
            # 范围框数值部分
            shape[1] = 5
            target_['bboxv'] = torch.zeros(*shape,device=outputs[j].device,dtype=outputs[j].dtype)
            # 范围框符号部分 同号为1 异号为0
            shape[1] = shape[0]
            target_['bboxs'] = torch.zeros(*shape[1:],device=outputs[j].device,dtype=torch.long)
            # 置信度部分 有目标为1 无为0
            target_['score'] = torch.zeros(*shape[1:],device=outputs[j].device,dtype=outputs[j].dtype)
            # 类别标签部分 
            if num_classes > 1:
                target_['label'] = torch.zeros(*shape[1:],device=outputs[j].device,dtype=torch.long)
            targets_.append(target_)
        # 为目标张量赋值
        for j in range(batch_size):
            bboxes = targets[j]['bboxes']
            labels = targets[j]['labels']
            for k in range(len(labels)):
                bbox = bboxes[k]
                label = labels[k]
                abs_bbox = [bbox[b]*self.image_size if b < 4 else bbox[b] for b in range(6)]
                w,_ = bbox_tr_get_wh(abs_bbox)
                x_c = abs_bbox[0]
                y_c = abs_bbox[1]
                u = bbox[2]
                v = bbox[3]
                s = int(abs_bbox[4])
                p = abs_bbox[5]
                for l in range(3):
                    if w >= bboxw_range[l][0] and w <= bboxw_range[l][1]:
                        pix_p_count[l] = pix_p_count[l] + 1
                        pix_n_count[l] = pix_n_count[l] - 1
                        x_cf = x_c/grid_size[l]
                        x_ci = int(x_cf)
                        x_cf = x_cf - x_ci
                        y_cf = y_c/grid_size[l]
                        y_ci = int(y_cf)
                        y_cf = y_cf - y_ci
                        # x_ci,y_ci就是用于预测目标的特征向量在特征图上坐标
                        # 用于赋值的目标tensor 值分别是 中心点坐标 向量分量的模 投影系数 同号 异号 置信度 目标
                        tb = torch.tensor([x_cf,y_cf,u,v,p])
                        targets_[l]['bboxv'][j,:,y_ci,x_ci] = tb
                        targets_[l]['score'][j,y_ci,x_ci] = 1
                        targets_[l]['bboxs'][j,y_ci,x_ci] = s
                        if num_classes > 1:
                            targets_[l]['label'][j,y_ci,x_ci] = label
        
        # 正例置信度
        self.score_loss_p_log = 0.0            
        # 负例置信度
        self.score_loss_n_log = 0.0         
        # 范围框数值部分
        self.bboxv_loss_log = 0.0 
        # 范围框符号部分
        self.bboxs_loss_log = 0.0 
        # 类别标签
        self.label_loss_log = 0.0 

        sum_loss = torch.tensor([0.],dtype=outputs[0].dtype,device=outputs[0].device,requires_grad=True)

        # 逐特征图计算损失
        for j in range(3):
            output = outputs[j]
            target = targets_[j]

            # 置信度损失
            score_output = output[:,7,:,:]
            score_target = target['score'] 
            score_loss = self.mse(score_output,score_target)
            #    正例置信度
            score_loss_p = torch.sum(score_loss*score_target)
            #    负例置信度
            score_weight_n = torch.tensor(1) - score_target
            score_loss_n = torch.sum(score_loss*score_weight_n)

            # 范围框数值部分损失
            bboxv_output = output[:,:5,:,:]
            bboxv_target = target['bboxv']
            bboxv_loss = self.smooth_l1(bboxv_output,bboxv_target)
            bboxv_loss = torch.sum(bboxv_loss,dim = 1)
            bboxv_loss = torch.sum(bboxv_loss*score_target)

            # 范围框符号部分损失
            bboxs_output = output[:,5:7,:,:]
            bboxs_target = target['bboxs']
            bboxs_loss = self.cross_entropy(bboxs_output,bboxs_target)
            bboxs_loss = torch.sum(bboxs_loss*score_target)

            # 对loss求均值
            if pix_n_count[j] > 0:
                score_loss_n = score_loss_n / pix_n_count[j] 
            if pix_p_count[j] > 0:
                score_loss_p = score_loss_p / pix_p_count[j]
                bboxs_loss = bboxs_loss / pix_p_count[j]
                bboxv_loss = bboxv_loss / pix_p_count[j]
            
            # 由于 正负例不均衡  符号部分对范围框相同性的影响更大
            sum_loss = sum_loss + 0.25*score_loss_n + 1.75*score_loss_p + 0.8*bboxs_loss + 1.2*bboxv_loss

            # 类别标签部分损失
            if num_classes > 1:
                label_output = output[:,8:,:,:]
                label_target = target['label']
                label_loss = self.cross_entropy(label_output,label_target)
                label_loss = torch.sum(label_loss*score_target)
                if pix_p_count[j] > 0:
                    label_loss = label_loss / pix_p_count[j]
                sum_loss = sum_loss + 2*label_loss
                self.label_loss_log = self.label_loss_log + label_loss

            self.score_loss_p_log = self.score_loss_p_log + score_loss_p
            self.score_loss_n_log = self.score_loss_n_log + score_loss_n   
            self.bboxv_loss_log = self.bboxv_loss_log + bboxv_loss 
            self.bboxs_loss_log = self.bboxs_loss_log + bboxs_loss 
        
        return sum_loss
    

def my_collate_fn(batch):
    images = [item[0] for item in batch]
    images = torch.stack(images,0)
    targets_np = [item[1] for item in batch]
    targets = []
    for target_np in targets_np:
        target = {key: torch.tensor(target_np[key]) for key in target_np}
        targets.append(target)
    return images, targets_np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    num_classes = 1
    batch_size = 8
    image_size = 416
    bboxw_range = [(48,144),(24,72),(12,36)]
    log_batchs = 20
    start_epoch = 1
    end_epoch = 1001

    transform = transforms.Compose([
        # transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    lan4set = PairFileDataset(r'D:\cvImageSamples\lan4\SampleImages','.bmp',transform= transform)
    
    image, target = lan4set[3]

    bboxes = target['bboxes']
    cids = target['labels']
    image = (image / 2 + 0.5)*255    # unnormalize
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plot_bbox(image, bboxes, labels=cids,absolute_coordinates=False)
    plt.show()
    
    trainloader = torch.utils.data.DataLoader(lan4set, batch_size=batch_size,shuffle=True, num_workers=1,collate_fn = my_collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = TRD(bboxw_range,image_size,num_classes)    
    net.load_state_dict(torch.load('./param/TRD_final.pth'))
    net.to(device)
    criterion = TRDLoss(bboxw_range)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # dataiter = iter(trainloader)
    # images, targets = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # images = images.to(device)
    # pred = net.detect(images)
    # image = images[0,:,:,:]
    # image = (image / 2 + 0.5)*255    # unnormalize
    # image = image.to('cpu')
    # image = image.numpy()
    # image = np.transpose(image, (1, 2, 0))
    # plot_bbox(image, pred)
    # plt.show()    

    
    for epoch in range(start_epoch,end_epoch):  # loop over the dataset multiple times
        net.train()
        for i, (images, targets) in enumerate(trainloader, 0):
            # get the inputs
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print log
            if i % log_batchs == (log_batchs-1):    # print every 2000 mini-batches
                print('[%d, %7d] sum_loss: %.3f score_loss_p: %.3f score_loss_n: %.3f bboxv_loss: %.3f bboxs_loss: %.3f label_loss: %.3f' %
                      (epoch, i + 1, 
                      loss, 
                      criterion.score_loss_p_log, criterion.score_loss_n_log,
                      criterion.bboxv_loss_log, criterion.bboxs_loss_log, 
                      criterion.label_loss_log))
        
        if epoch % 50 == 0:
            save_path = './param/TRD_%d.pth'%epoch
            torch.save(net.state_dict(), save_path)
    
    torch.save(net.state_dict(), './param/TRD_final.pth')
    print('Finished Training')