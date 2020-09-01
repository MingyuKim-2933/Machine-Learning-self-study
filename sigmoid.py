import numpy as np
import matplotlib.pylab as plt


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
