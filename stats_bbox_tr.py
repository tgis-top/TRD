import os
from os import listdir
from os.path import join
from bbox_tr import bbox_tr_get_wh

in_dir = r'E:\SourceCode\Python\TRD\DOTA'
out_dir = r'E:\SourceCode\Python\TRD'
cls_stats = {}

valid_objs = 0
total_objs = 0

for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.txt': 
        in_file = open(join(in_dir,i), 'r')
        for line in in_file:
            parts = line.split()
            if len(parts) > 6:
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:7]]
                w,h = bbox_tr_get_wh(bbox)
                if w <= 0.0:
                    continue
                total_objs = total_objs + 1
                if w >= 12:
                    valid_objs = valid_objs+1
                ar = h/w
                if cls_id in cls_stats:
                    cls_stat = cls_stats[cls_id]
                    if cls_stat[0] > ar:
                        cls_stat[0] = ar
                    if cls_stat[1] < ar:
                        cls_stat[1] = ar                    
                    cls_stat[2] = cls_stat[2] + ar
                    if cls_stat[3] > w:
                        cls_stat[3] = w
                    if cls_stat[4] < w:
                        cls_stat[4] = w
                    if cls_stat[5] > h:
                        cls_stat[5] = h
                    if cls_stat[6] < h:
                        cls_stat[6] = h
                    cls_stat[7] = cls_stat[7] + 1
                else:
                    cls_stats[cls_id] = [ar,ar,ar,w,w,h,h,1]          
        in_file.close() 

for cls_id in cls_stats:
    cls_stat = cls_stats[cls_id]
    cls_stat[2] = cls_stat[2]/cls_stat[7]

cls_stats_file = open(join(out_dir,"cls_stats.txt"), 'w')
for cls_id in cls_stats:
    cls_stat = cls_stats[cls_id]
    cls_stats_file.write( '%3d, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %5d\n'%(
        cls_id,cls_stat[0],cls_stat[1],cls_stat[2],
        cls_stat[3],cls_stat[4],cls_stat[5],cls_stat[6],
        cls_stat[7])
        )
cls_stats_file.write('vaid_objs = %d\n'%valid_objs)
cls_stats_file.write('total_objs = %d'%total_objs)
cls_stats_file.close()
