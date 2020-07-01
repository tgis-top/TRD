import numpy as np 
import math
import random

__all__ = (
    "bbox_4pt_2_tr",\
    "bbox_cwh_2_tr",\
    "bbox_tr_get_wh",\
    "bbox_tr_2_4pt",\
    "plot_bbox",\
    )

# # 最小二乘法
# # 最小二乘线性方程组模型的未知量为 矩形中心点坐标 中心点到任意两个相邻顶点的向量
# # 通过这几个未知量建立了计算平行四边形顶点坐标的方程（每加直角约束，所以是平行四边形）
# A = np.array(
#     [[1,0,-1,0,0,0],
#      [0,1,0,-1,0,0],
#      [1,0,0,0,-1,0],
#      [0,1,0,0,0,-1],
#      [1,0,1,0,0,0],
#      [0,1,0,1,0,0],
#      [1,0,0,0,1,0],
#      [0,1,0,0,0,1]])
# #求A.T.dot(A)的逆
# ATA_inv = np.linalg.inv(A.T.dot(A))

# AAA = ATA_inv.dot(A.T)

# # 最小二乘法求解出来的解系数矩阵就是这个
# AAA = np.array(
#     [[ 0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ],
#      [ 0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25],
#      [-0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ,  0.  ,  0.  ],
#      [ 0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ,  0.  ],
#      [ 0.  ,  0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ],
#      [ 0.  ,  0.  ,  0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ]])

def bbox_4pt_2_tr(bbox_):
    if not hasattr(bbox_4pt_2_tr,'AAA'):
        bbox_4pt_2_tr.AAA = np.array(
            [[ 0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ],
             [ 0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  ,  0.25],
             [-0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ,  0.  ,  0.  ],
             [ 0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ,  0.  ],
             [ 0.  ,  0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ],
             [ 0.  ,  0.  ,  0.  , -0.5 ,  0.  ,  0.  ,  0.  ,  0.5 ]])
    bbox = np.array(bbox_)
    # 拟合时没加约束，不一定是矩形，也就是先拟合了一个平行四边形
    # 平行四边形的参数包括中心点坐标 中心点到其中两个顶点的向量
    D = bbox_4pt_2_tr.AAA.dot(bbox)
    # 所以要缩放中心点到两个顶点的向量使其长度相等
    lp = math.sqrt(D[2]*D[2] + D[3]*D[3])
    lq = math.sqrt(D[4]*D[4] + D[5]*D[5])
    if lp == 0.0 or lq == 0.0:
        return [D[0],D[1],0.,0.,1,0]
    ll = (lp+lq)/2
    D[2] = D[2]*ll/lp
    D[3] = D[3]*ll/lp
    D[4] = D[4]*ll/lq
    D[5] = D[5]*ll/lq
    # 按照夹角大小和方向约束选择标注四边形的向量
    dp = D[2]*D[4] + D[3]*D[5]
    # 如果是正方形，人为规定取两个分量同号的顶点作为标注量
    # 另外，如果顶点在坐标轴上，选择x轴方向上的顶点作为标注量
    if dp == 0:
        if (D[2] > 0 and D[3] >= 0) or (D[2] < 0 and D[3] <= 0):
            return [D[0],D[1],2*abs(D[2]),2*abs(D[3]),1,0]
        else:
            return [D[0],D[1],2*abs(D[4]),2*abs(D[5]),1,0]
    
    cos = dp/(ll*ll)
    # 向量(D[2],D[3])在(D[4],D[5])上的投影向量与(D[4],D[5])的比例系数 就是|cos|
    if cos < 0:
        D[2] = -D[2]
        D[3] = -D[3]
        cos = -cos
    cp = D[2]*D[5] - D[3]*D[4]
    if cp < 0:
        D[2],D[4] = D[4],D[2]
        D[3],D[5] = D[5],D[3]
    # 所选向量两个分量是同号(1)还是异号(0)
    s4 = 1
    if D[4] < 0:
        D[4] = - D[4]
        s4 = -1
    s5 = 1
    if D[5] < 0:
        D[5] = - D[5]
        s5 = -1
    s = 1 if s4*s5 == 1 else 0
    # 为了和 中心点坐标+宽+高 的标注方式兼容，将所选向量扩大一倍
    bbox_tr_ = [D[0],D[1],2*D[4],2*D[5],s,cos]

    return bbox_tr_

def bbox_cwh_2_tr(bbox_):
    # 如果宽高相等，也就是正方形，人为规定取两个分量同号的顶点作为标注量
    if bbox_[2] == bbox_[3]:
        return [bbox_[0],bbox_[1],bbox_[2],bbox_[3],1,0]
    P = [bbox_[2],bbox_[3]]
    Q = [-bbox_[2],bbox_[3]]
    dp = P[0]*Q[0] + P[1]*Q[1]
    ll = math.sqrt(P[0]*P[0] + P[1]*P[1])
    cos = dp/(ll*ll)
    if cos < 0:
        Q = [bbox_[2],-bbox_[3]]
        cos = -cos
    cp = P[1]*Q[0] - P[0]*Q[1]
    if cp < 0:
        P,Q = Q,P
    s0 = 1
    if P[0] < 0:
        P[0] = - P[0]
        s0 = -1
    s1 = 1
    if P[1] < 0:
        P[1] = - P[0]
        s1 = -1
    s = 1 if s0*s1 == 1 else 0

    bbox_tr_ = [bbox_[0],bbox_[1],P[0],P[1],s,cos]

    return bbox_tr_

def bbox_tr_get_wh(bbox_):
    if bbox_[2] == 0. and bbox_[3] == 0.:
        return 0,0
    r1 = bbox_[2]/2
    r2 = bbox_[3]/2
    # 如果异号将其中一个取负
    if bbox_[4] == 0:
        r2 = -r2    
    p1 = bbox_[5]*r1
    p2 = bbox_[5]*r2
    rls = r1*r1 + r2*r2
    rl = math.sqrt(rls)
    pl = math.sqrt(p1*p1+p2*p2)
    h = math.sqrt(2*rls + 2*rl*pl)
    w = 2*rls - 2*rl*pl
    w = 0 if w <= 0. else math.sqrt(w)    

    return w,h

def bbox_tr_2_4pt(bbox_):
    bbox = [a for a in bbox_]
    bbox[2] = bbox[2]/2
    bbox[3] = bbox[3]/2
    # 如果异号将其中一个取负
    if bbox_[4] == 0:
        bbox[3] = -bbox[3]  
    A1 = bbox[2]*bbox[2]
    A2 = bbox[3]*bbox[3]
    A = A1 + A2
    B = 2*bbox[5]*bbox[3]*A
    C = bbox[5]*bbox[5]*A*A - A1*A1 - A1*A2
    n1 = (B + math.sqrt(B*B-4*A*C))/(2*A)
    if bbox[2] == 0:
        m1 = bbox[5]*bbox[3]
    else:
        m1 = (bbox[5]*A-n1*bbox[3])/bbox[2]    
    CD = [m1,n1]
    cp1 = m1*bbox[3] - n1*bbox[2]
    if cp1 < 0:
        n2 = (B - math.sqrt(B*B-4*A*C))/(2*A)
        if bbox[2] == 0:
            m2 = bbox[5]*bbox[3]
        else:
            m2 = (bbox[5]*A-n2*bbox[3])/bbox[2] 
        CD = [m2,n2]
    bbox_4pt_ = [
        bbox[0] - bbox[2],
        bbox[1] - bbox[3],
        bbox[0] - CD[0],
        bbox[1] - CD[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
        bbox[0] + CD[0],
        bbox[1] + CD[1]]
    
    return bbox_4pt_


def plot_image(img, ax=None, reverse_rgb=False):
    """Visualize image.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    Examples
    --------

    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """
    from matplotlib import pyplot as plt
    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax

def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpathes

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        bbox_4pt_ = bbox_tr_2_4pt(bbox)
        bbox_4pt_ = np.reshape(bbox_4pt_,(-1,2))
        polygon = plt.Polygon(bbox_4pt_,
                             fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(polygon)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(bbox[0], bbox[1],
                    '{:s} {:s}'.format(class_name, score),
                    # bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')
    return ax


