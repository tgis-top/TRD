import os
from os import listdir
from os.path import join
import numpy as np 

from bbox_tr import *


bbox_cwh = np.array([25,45,50,90])

bbox_tr = bbox_cwh_2_tr(bbox_cwh)
print(bbox_tr)
bbox_wh = bbox_tr_get_wh(bbox_tr)
print(bbox_wh)
bbox_4pt2 = bbox_tr_2_4pt(bbox_tr)
bbox_4pt2 = np.reshape(bbox_4pt2,(-1,2))
print(bbox_4pt2)

in_dir = r'D:\cvImageSamples\lan4\labelcwh'
out_dir = r'D:\cvImageSamples\lan4\labeltra'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.txt':  
        out_file = open(join(out_dir,i), 'w')
        in_file = open(join(in_dir,i), 'r')
        for line in in_file:
            parts = line.split()
            if(len(parts) > 4):
                bbox_cwh_ = [float(x) for x in parts[1:5]]
                bbox_tr_ = bbox_cwh_2_tr(bbox_cwh_)
                out_file.write(
                    parts[0] + " " 
                    + " ".join(['%.7f'%a for a in bbox_tr_[:4]]) + " " 
                    + str(int(bbox_tr_[4])) + " " 
                    + '%.7f'%bbox_tr_[5] + '\n')
        out_file.close()   
        in_file.close() 

            

