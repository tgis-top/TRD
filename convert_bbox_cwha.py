import os
from os import listdir
from os.path import join
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import pickle
import shutil

from bbox_tr import *


# # bbox_cwha = np.array([708.3678,557.4165,230,230,0.700186])
# bbox_cwha = np.array([708.3678,557.4165,218.2801,240.193,0.700186])
# # bbox_cwha = np.array([323.5205,337.9315,148.0,172.0,0.151939])

# bbox_tr = bbox_cwha_2_tr(bbox_cwha)

# image = Image.open(r'D:\cvImageSamples\lan4\test\image_170210_024.JPG')
# image = np.array(image)
# plot_bbox(image, np.array(bbox_tr).reshape(-1,len(bbox_tr)))
# plt.show()  


def convert_annotation(in_dir,image_id,out_dir,classes):
    annotation_file = '%s/%s.xml'%(in_dir, image_id)
    if(os.path.exists(annotation_file)): 
        obj_cnt = 0       
        in_file = open(annotation_file, encoding='utf8')
        try:
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            label_file = '%s/%s.txt'%(out_dir, image_id)
            out_file = open(label_file, 'w')
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('robndbox')
                b = (float(xmlbox.find('cx').text), 
                     float(xmlbox.find('cy').text), 
                     float(xmlbox.find('w').text), 
                     float(xmlbox.find('h').text),
                     float(xmlbox.find('angle').text))
                bb = bbox_cwha_2_tr(b)
                out_file.write(
                    str(cls_id) + " " 
                    + " ".join(['%.7f'%a for a in bb[:4]]) + " " 
                    + str(int(bb[4])) + " " 
                    + '%.7f'%bb[5] + '\n')
                obj_cnt = obj_cnt + 1
            out_file.close()
            # if obj_cnt == 0:
            #     os.remove(label_file)
        except Exception as e:
            print(annotation_file)
            print('Error:',e)
        finally:
            in_file.close()
        if obj_cnt > 0:
            return True
    return False

in_dir = r'E:\RS_PRJ\zonglv'
out_dir = r'E:\RS_PRJ\zonglv'
classes = ["1"]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.xml': 
        convert_annotation(in_dir,image_id,out_dir,classes)

            

