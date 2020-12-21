import os
from os import listdir
from os.path import join
import numpy as np 
from PIL import Image

from bbox_tr import *


in_dir = r'E:\RS_PRJ\植株识别\玉米'
out_dir = r'E:\RS_PRJ\植株识别\玉米\cwh'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.txt':  
        out_file = open(join(out_dir,i), 'w')
        in_file = open(join(in_dir,i), 'r')
        for line in in_file:
            parts = line.split()
            if(len(parts) > 6):
                bbox_tr_ = [float(x) for x in parts[1:7]]
                w,h = bbox_tr_get_wh(bbox_tr_)
                bbox_cwh_ = [bbox_tr_[0],bbox_tr_[1],w,h]
                if w > 1:
                    img = Image.open(join(in_dir,image_id)+".jpg")
                    img_w = img.size[0]
                    img_h = img.size[1]
                    bbox_cwh_ = [bbox_tr_[0]/img_w,bbox_tr_[1]/img_h,w/img_w,h/img_h]
                out_file.write(
                    parts[0] + " " 
                    + " ".join(['%.7f'%a for a in bbox_cwh_]) + '\n')
        out_file.close()   
        in_file.close() 

            

