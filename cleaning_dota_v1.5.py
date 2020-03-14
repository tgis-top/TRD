import os
from os import listdir
from os.path import join

in_dir = r'E:\SourceCode\Python\DOTA_devkit\example\labelTxt'
out_dir = r'E:\SourceCode\Python\TRD\param'
# 这两个类别的高宽比到了45，不适合用预测范围框的方式检测目标
# 其实船也不太合适，其高宽比到20，但是船周围很单调
excludes = ('bridge','harbor')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i in os.listdir(in_dir):
    image_id,image_ext = os.path.splitext(i)
    if image_ext.lower() == '.txt':  
        out_file = open(join(out_dir,i), 'w')
        in_file = open(join(in_dir,i), 'r')
        for line in in_file:
            include = True
            parts = line.split()
            if len(parts) > 8:
                if parts[8] in excludes:
                    include = False
            if include:
                out_file.write(line)
        out_file.close()   
        in_file.close() 