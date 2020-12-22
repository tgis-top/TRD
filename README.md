# TRD
The TangRui annotation method (TRA) of arbitrary-oriented bounding box and the TangRui object detector (TRD). 

## TRD.py

网络定义，没啥看头。

## train.py

训练工具，参数看代码。

## predict.py

预测工具，参考看代码。

## predict_ui.py

用wxPython实现的预测工具，安装了wxPython可以用这个。

## bbox_tr.py

倾斜范围框的核心实现代码，包含各种范围框标注方式之间转换的函数，以及plot函数。

## polyiou.py

多边形范围框IOU计算函数，从DOTA_Devkit中扒出来的。只是对C语言实现的调用，C语言实现被编译为了_polyiou.cp37-win_amd64.pyd。也就说编译结果必须放一起才能用。tra标注方式先转成多边形再计算IOU。

## polynms.py

多边形范围框NMS函数。

## convert_box_4pt.py

四点式范围框转TRA。

## convert_box_cwh.py

CWH式范围框转TRA。

## convert_box_cwha.py

CWHA式范围框转TRA。

## convert_box_tr2cwh2.py

TRA式范围框转CWH。

## PairFileDataset.py

文件对数据集，一个文件对应一个yolo式的标注文件。
一行一个目标，如“0 0.5040441 0.3867188 0.2066176 0.2324219 0 0.1171439”。第一个是目标ID，后面分别是 “中心点坐标 标注向量的两个分量模 标注向量分量积的符号 标注向量的相邻顶点向量的cos值”。具体意义参考[https://zhuanlan.zhihu.com/p/150780620]

