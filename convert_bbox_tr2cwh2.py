import os
from os import listdir
from os.path import join
import numpy as np 
from PIL import Image

from bbox_tr import *


in_dir = r'D:\ImageSamples\DOTA v1.5\train\images\images'
out_dir = r'D:\ImageSamples\DOTA v1.5\train\images\cwh'

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
                bbox_4pt_ = bbox_tr_2_4pt(bbox_tr_)
                bbox_cwh_ =  bbox_4pt_2_cwh(bbox_4pt_)
                img_path = join(in_dir,image_id)+".png"
                if bbox_cwh_[2] > 1 and os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_w = img.size[0]
                    img_h = img.size[1]
                    bbox_cwh_ = [bbox_tr_[0]/img_w,bbox_tr_[1]/img_h,bbox_cwh_[2]/img_w,bbox_cwh_[3]/img_h]
                out_file.write(
                    parts[0] + " " 
                    + " ".join(['%.7f'%a for a in bbox_cwh_]) + '\n')
        out_file.close()   
        in_file.close() 

            

