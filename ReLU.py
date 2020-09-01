import numpy as np
import matplotlib.pylab as plt

def relu(a):
    return np.maximum(0, a)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5)
plt.show()