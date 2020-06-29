from polyiou import iou_poly
import numpy as np

p = [0, 0, 1, 0, 1, 1, 0, 1]
q = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5]

# p = np.array(p)
# q = np.array(q)

iou = iou_poly(p,q)

print(iou)