import os
from os import listdir
from os.path import join
import numpy as np 

from bbox_tr import *

# bbox_4pt = np.array([2753,2408,2861,2385,2888,2468,2805,2502])
# bbox_4pt = np.array([2890,2468,2796,2378,2763,2412,2856,2503])
# bbox_4pt = np.array([2890.13290218,2468.92017875,2798.28550745,2377.49872394,2763.36709782,2412.57982125,2855.21449255,2504.00127606])
bbox_4pt = np.array([0,90,50,90,50,0,0,0])

bbox_tr = bbox_4pt_2_tr(bbox_4pt)
print(bbox_tr)
bbox_wh = bbox_tr_get_wh(bbox_tr)
print(bbox_wh)
bbox_4pt2 = bbox_tr_2_4pt(bbox_tr)
bbox_4pt2 = np.reshape(bbox_4pt2,(-1,2))
print(bbox_4pt)
print(bbox_4pt2)

in_dir = r''
out_dir = r''
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
                w,h = bbox_tr_get_wh(bbox_tr_)
                if w > 11 and h/w < 20.:
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
            

