import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet_backbone

from bbox_tr import bbox_tr_get_wh
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
            # nn.Conv2d(self.backbone.feature_planes[0], 256, kernel_size=1, stride=1),
            # nn.ReLU(inplace=True),
            # 5: 中心点坐标 CD向量 投影长度 1：CD向量同号或者异号 3：是否有目标
            nn.Conv2d(self.backbone.feature_planes[0], 5 + 1 + 1 + (0 if num_classes==1 else num_classes), kernel_size=1, stride=1),
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
                if output[0,6,i,j] < score_thresh:
                    continue
                bbox = [0.]*8
                bbox[0] = (j + output[0,0,i,j].item())*grid_size
                bbox[1] = (i + output[0,1,i,j].item())*grid_size
                bbox[2] = output[0,2,i,j].item()*self.image_size
                bbox[3] = output[0,3,i,j].item()*self.image_size
                bbox[4] = 0 if output[0,5,i,j] > 0.5 else 1
                bbox[5] = output[0,4,i,j].item()
                if self.num_classes > 1:
                    bbox[6] = output[0,7:,i,j].argmax().item()
                else:
                    bbox[6] = 0
                bbox[7] = output[0,6,i,j].item()
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

    def __init__(self,bboxw_range,image_size = 416, num_classes=20, is_abs_bbox=False):
        super(TRDLoss,self).__init__()
        self.bboxw_range = bboxw_range
        self.label_div = -1.0
        self.image_size = image_size
        self.is_abs_bbox = is_abs_bbox
        self.score_loss = nn.MSELoss(reduction='none')
        self.bboxs_loss = nn.MSELoss(reduction='none')
        self.bboxv_loss = nn.MSELoss(reduction='none')
        self.label_loss = nn.CrossEntropyLoss(reduction='none')

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
        if num_classes > 1 and self.label_div < 0.0:
            self.label_div = math.log2(num_classes)
        # 特征图像素对应输入图像的尺寸
        grid_size = []
        # 创建目标张量
        targets_ = []
        # 各层特征图对应目标的像素数量（三层特征图金字塔）
        pix_p_count = torch.tensor([0,0,0],device=outputs[0].device,dtype=outputs[0].dtype)
        # 各层特征图不对应目标的像素数量（三层特征图金字塔）
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
            target_['bboxs'] = torch.zeros(*shape[1:],device=outputs[j].device,dtype=outputs[j].dtype)
            # 范围框符号的权重 顶点向量靠近坐标轴轴时符号无论取何值范围框误差都不会太大 此时权重应该很小
            target_['bboxs_weight'] = torch.zeros(*shape[1:],device=outputs[j].device,dtype=outputs[j].dtype)
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
                abs_bbox = bbox
                if not self.is_abs_bbox:
                    abs_bbox = [bbox[b]*self.image_size if b < 4 else bbox[b] for b in range(6)]
                w,_ = bbox_tr_get_wh(abs_bbox)
                x_c = abs_bbox[0]
                y_c = abs_bbox[1]
                u = bbox[2]
                v = bbox[3]
                if self.is_abs_bbox:
                    u = u/self.image_size
                    v = v/self.image_size
                s = int(bbox[4])
                p = bbox[5]
                # 顶点向量两个分量模值差异越大下述算式的结果越接近1
                bsw = (abs_bbox[2]-abs_bbox[3])/(abs_bbox[2]+abs_bbox[3])
                bsw = bsw*bsw
                # 反s曲线函数 bsw变大到一定程度权重应该突然减小
                bsw = 1/(1+math.exp(100*(bsw-0.70)))
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
                        targets_[l]['bboxs_weight'][j,y_ci,x_ci] = bsw
                        if num_classes > 1:
                            targets_[l]['label'][j,y_ci,x_ci] = torch.LongTensor([label])
        
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
            score_output = output[:,6,:,:]
            score_target = target['score'] 
            score_loss = self.score_loss(score_output,score_target)
            #    正例置信度
            score_loss_p = torch.sum(score_loss*score_target)
            #    负例置信度
            score_weight_n = torch.tensor(1) - score_target
            score_loss_n = torch.sum(score_loss*score_weight_n)

            # 范围框符号部分损失
            bboxs_output = output[:,5,:,:]
            bboxs_target = target['bboxs']
            bboxs_weight = target['bboxs_weight']
            bboxs_loss = self.bboxs_loss(bboxs_output,bboxs_target)
            # bboxs_output = F.softmax(bboxs_output, dim=1)
            # bboxs_loss = F.nll_loss(bboxs_output,bboxs_target, reduction='none')
            bboxs_loss = torch.sum(bboxs_loss*score_target)

            # 范围框数值部分损失
            bboxv_output = output[:,:5,:,:]
            bboxv_target = target['bboxv']
            # 为了让 bboxv_weight + bboxs_weight == 2
            # bboxv_weight = score_target + score_target - bboxs_weight
            bboxv_loss = self.bboxv_loss(bboxv_output,bboxv_target)
            bboxv_loss = torch.sum(bboxv_loss,dim = 1)
            bboxv_loss = torch.sum(bboxv_loss*score_target)

            # 对loss求均值
            if pix_n_count[j] > 0:
                score_loss_n = score_loss_n / pix_n_count[j] 
            if pix_p_count[j] > 0:
                score_loss_p = score_loss_p / pix_p_count[j]
                bboxs_loss = bboxs_loss / pix_p_count[j]
                bboxv_loss = bboxv_loss / pix_p_count[j]
            
            # 由于 正负例不均衡  等原因所以额外加了下列权重，可以自定义权重
            # 为了提升训练效果，可以在训练过程中根据损失数值编号规律进行调整
            sum_loss = sum_loss + 0.7*score_loss_n + 1.3*score_loss_p + 0.7*bboxs_loss + 0.3*bboxv_loss

            # 类别标签部分损失
            if num_classes > 1:
                label_output = output[:,7:,:,:]
                label_target = target['label']
                label_loss = self.label_loss(label_output,label_target)
                label_loss = torch.sum(label_loss*score_target)
                if pix_p_count[j] > 0:
                    label_loss = label_loss / (pix_p_count[j]*self.label_div)
                sum_loss = sum_loss + 2*label_loss
                self.label_loss_log = self.label_loss_log + label_loss

            self.score_loss_p_log = self.score_loss_p_log + score_loss_p
            self.score_loss_n_log = self.score_loss_n_log + score_loss_n   
            self.bboxv_loss_log = self.bboxv_loss_log + bboxv_loss 
            self.bboxs_loss_log = self.bboxs_loss_log + bboxs_loss 
        
        return sum_loss

