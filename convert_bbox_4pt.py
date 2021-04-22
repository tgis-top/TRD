import os
from os import listdir
from os.path import join
import numpy as np 

from bbox_tr import *


in_dir = r'D:\ImageSamples\DOTA v1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
out_dir = r'E:\SourceCode\Python\TRD\DOTA'
classes = {}

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.txt':  
        out_file = open(join(out_dir,i), 'w')
        in_file = open(join(in_dir,i), 'r')
        for line in in_file:
            parts = line.split()
            if len(parts) > 8:
                cls_id = len(classes)
                if parts[8] in classes:
                    cls_id = classes[parts[8]]
                else:
                    classes[parts[8]] = cls_id
                bbox_4pt_ = [float(x) for x in parts[:8]]
                bbox_tr_ = bbox_4pt_2_tr(bbox_4pt_)
                out_file.write(
                    str(cls_id) + " " 
                    + " ".join(['%.3f'%a for a in bbox_tr_[:4]]) + " " 
                    + str(int(bbox_tr_[4])) + " " 
                    + '%.3f'%bbox_tr_[5] + " " 
                    + parts[9] + '\n')
        out_file.close()   
        in_file.close() 

cls_file = open(join(out_dir,"cls_id.txt"), 'w')
for cls_name in classes:
    cls_id = classes[cls_name]
    cls_file.write( '%3d, %s\n'%(cls_id,cls_name))
cls_file.close()
            

