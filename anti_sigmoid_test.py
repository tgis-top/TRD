import numpy as np
import math
import matplotlib.pyplot as plt

def anti_sigmoid(x):
    return 1/(1+np.exp(100*(x-0.70)))


# x = np.arange(0,1.1,0.005)
# y = anti_sigmoid(x)
# plt.plot(x,y)
# plt.show()

angle = np.arange(0.00001,math.pi/4,0.005)

cos = np.cos(angle)
sin = np.sin(angle)
cs = np.stack([cos,sin])
td = np.max(cs,axis=0)

rd = (cos-sin)**2/(cos+sin)**2

tdas = anti_sigmoid(td)

rdas = anti_sigmoid(rd)

angle = angle*(180.0/math.pi)


plt.plot(angle,rd,label="RD")
# plt.plot(angle,td,label="TD")
plt.plot(angle,rdas,label="RDAS")
# plt.plot(angle,tdas,label="TDAS")

plt.legend(loc="upper left")
plt.show()