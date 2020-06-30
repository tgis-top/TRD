import os
import numpy as np
from polyiou import iou_poly
from bbox_tr import bbox_tr_2_4pt

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# num_classes 类别数量，文件中的类别编号最大值应该为num_classes-1
def eval_net(gtspath, detspath, iou_thresh = 0.5, num_classes=1, use_07_metric=False):
    # 各个类别所有bbox
    dets_bbox = [[] for i in range(num_classes)]
    # 各个类别所有bbox对应的图像文件编号
    dets_imgid = [[]]*num_classes
    # 各个类别所有bbox对应的置信度
    dets_score = [[] for i in range(num_classes)]

    # 各个类别各图像文件中的所有真值bbox
    gts_files = [[]]*num_classes
    # 各个类别真值bbox的数量
    gts_num = [0.0]*num_classes

    imgid = 0
    for i in os.listdir(detspath):
        _,file_ext = os.path.splitext(i)
        if file_ext.lower() == '.txt':
            gts_lbl_path = os.path.join(gtspath,i)
            if not os.path.exists(gts_lbl_path):
                continue

            dets_lbl = open(os.path.join(detspath,i),'r')
            for line in dets_lbl:
                parts = line.split()
                lbl = int(parts[0])
                bbox = [float(x) for x in parts[1:7]]
                bbox_4pt = bbox_tr_2_4pt(bbox)
                # score = float(parts[7])
                score = np.random.uniform()
                dets_score[lbl].append(score)
                dets_bbox[lbl].append(bbox_4pt)
                dets_imgid[lbl].append(imgid)
            
            for n in range(num_classes):
                gts_files[n].append([])
            
            gts_lbl = open(gts_lbl_path,'r')
            for line in gts_lbl:
                parts = line.split()
                lbl = int(parts[0])
                bbox = [float(x) for x in parts[1:7]]
                bbox_4pt = bbox_tr_2_4pt(bbox)
                gts_files[lbl][imgid].append(bbox_4pt)
                gts_num[lbl] = gts_num[lbl] + 1
            
            imgid = imgid+1
    
    aps = np.zeros(num_classes)  
    # 逐类别计算precision recall
    for i in range(num_classes):
        sorted_idx = np.argsort(dets_score[i])
        bbox_num = len(sorted_idx)
        tp = np.zeros(bbox_num)
        fp = np.zeros(bbox_num)
        for n in range(bbox_num-1,-1,-1):
            si = sorted_idx[n]
            score = dets_score[i][si]
            bb = dets_bbox[i][si]
            imgid = dets_imgid[i][si]
            # 找到imgid上的所有本类别的真值范围框
            bbgts = gts_files[i][imgid]
            # 判断该预测框是否是true positive
            is_tp = False
            for bbgt in bbgts:
                iou = iou_poly(bb,bbgt)
                if(iou >= iou_thresh):
                    is_tp = True
                    break
            if is_tp:
                tp[n] = 1
            else:
                fp[n] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / gts_num[i]
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        aps[i] = ap
    return aps

                
if __name__ == '__main__':
    aps = eval_net(r'D:\cvImageSamples\lan4\lanl',r'D:\cvImageSamples\lan4\SampleImages')
    mAP = np.mean(aps)
    print(aps)
    print("mAP=%.7f"%mAP)


