import numpy as np
import matplotlib.pylab as plt


def step_function(a):
    return np.array(a > 0, dtype=np.int)


def sigmoid(a):
    return np.array(1 / (1 + np.exp(-x)), dtype=np.float)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1, 1.1)
plt.plot(x, y, linestyle="--", label="step_function")
y = sigmoid(x)
plt.plot(x, y, label="sigmoid")
plt.legend()
plt.show()
