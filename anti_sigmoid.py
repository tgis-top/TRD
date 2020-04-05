import numpy as np
import matplotlib.pyplot as plt

def anti_sigmoid(x):
    return 1/(1+np.exp(50*(x-0.8)))

x = np.arange(-1,2,0.005)
y = anti_sigmoid(x)
plt.plot(x,y)
plt.show()