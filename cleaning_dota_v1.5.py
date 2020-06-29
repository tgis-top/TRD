import os
from os import listdir
from os.path import join
from PIL import Image

def exclude_classes(in_dir,out_dir,excludes):
    for i in os.listdir(in_dir):
        _,image_ext = os.path.splitext(i)
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

def exclude_small_images(img_dir,lbl_dir,img_w, img_h,img_ext='.png'):
    for i in os.listdir(img_dir):
        image_id,image_ext = os.path.splitext(i)
        if image_ext == img_ext:
            img = Image.open(join(img_dir,i)) 
            iw, ih = img.size
            img.close()
            if iw != img_w or ih != img_h:
                os.remove(join(img_dir,i))
                os.remove(join(lbl_dir,image_id+'.txt'))

def exclude_empty_samples(img_dir,lbl_dir,img_ext='.png'):
    for i in os.listdir(lbl_dir):
        image_id,lbl_ext = os.path.splitext(i)
        if lbl_ext == ".txt":
            lbl_file = open(join(lbl_dir,i), 'r')
            smplc = 0
            for line in lbl_file:
                parts = line.split()
                if len(parts) > 6:
                    smplc = smplc+1
            lbl_file.close()
            if smplc == 0:
                os.remove(join(lbl_dir,i))
                os.remove(join(img_dir,image_id+img_ext))


if __name__ == '__main__':
    # in_dir = r'E:\SourceCode\Python\DOTA_devkit\example\labelTxt'
    # out_dir = r'E:\SourceCode\Python\TRD\param'
    # # 早些时候统计出来的宽高比不对，因为官方切图工具把范围框也给切细了
    # # 这两个类别的高宽比太大，不适合用预测范围框的方式检测目标
    # # 其实船也不太合适，但是船周围很单调
    # excludes = ('bridge','harbor')
    
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    
    # exclude_classes(in_dir,out_dir,excludes)

    # 官方切图工具切出来了小于声明尺寸的图象，需要删除
    img_dir = r''
    lbl_dir = r''
    exclude_small_images(img_dir,lbl_dir,672,672)