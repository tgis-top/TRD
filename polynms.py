from polyiou import iou_poly
from bbox_tr import bbox_tr_2_4pt
import numpy as np
import math


# 范围框顶点之间的最大距离
def __max_bpt_square_dis(bbox_4pt1, bbox_4pt2):
    bbox_4pt1_ = np.array(bbox_4pt1)
    bbox_4pt2_ = np.array(bbox_4pt2)
    bbox_4pt1_ = bbox_4pt1_.reshape(-1,2)
    bbox_4pt1_ = bbox_4pt1_.repeat(4,0)
    bbox_4pt2_ = np.expand_dims(bbox_4pt2_,0).repeat(4,0)
    bbox_4pt2_ = bbox_4pt2_.reshape(-1,2)
    vetex_dis = np.square(bbox_4pt2_-bbox_4pt1_)
    vetex_dis = vetex_dis.sum(1)
    return vetex_dis.max()



def nms_poly(dets, num_classes, iou_thresh = 0.5, cd_thresh = 0.1):
    # 后面没开方 这里先平方一下
    cd_thresh = cd_thresh**2
    cls_ids = dets[:,6]
    scores = dets[:,7]
    bboxs_4pt = []
    keep_bboxs_4pt = []
    for i in range(dets.shape[0]):
        bboxs_4pt.append(bbox_tr_2_4pt(dets[i]))
    keep = []
    for cls_id in range(num_classes):
        cls_idx = np.where(cls_ids == cls_id)[0]
        cls_scores = scores[cls_idx]
        sorted_index = cls_scores.argsort()[::-1]
        while sorted_index.size > 0:
            idx_in_cls_idx = sorted_index[0]
            dets_idx = cls_idx[idx_in_cls_idx]
            keep.append(dets_idx)
            bbox_4pt = bboxs_4pt[dets_idx]
            bbox_tr = dets[dets_idx]
            keep_bboxs_4pt.append(bbox_4pt)
            index_mask = np.zeros_like(sorted_index,dtype=bool)
            for j in range(1,sorted_index.size):
                idx_in_cls_idx_ = sorted_index[j]
                dets_idx_ = cls_idx[idx_in_cls_idx_]
                bbox_4pt_ = bboxs_4pt[dets_idx_]
                bbox_tr_ = dets[dets_idx_]
                iou = iou_poly(bbox_4pt,bbox_4pt_)
                csd = (bbox_tr_[0] - bbox_tr[0])**2 + (bbox_tr_[1] - bbox_tr[1])**2
                mbsd = __max_bpt_square_dis(bbox_4pt,bbox_4pt_)
                if(iou < iou_thresh and csd/mbsd > cd_thresh):
                    index_mask[j] = True
            sorted_index = sorted_index[index_mask]
    
    return dets[keep],np.array(keep_bboxs_4pt)
            


