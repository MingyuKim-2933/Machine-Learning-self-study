import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)

x = np.arange(-6, 6, 0.01)


def sigmoid(x):  # Sigmoid 함수 = Logistic
    return 1 / (1 + np.exp(-x))


def softplus_func(x):  # SoftPlus 함수
    return np.log(1 + np.exp(x))


ax.plot(x, sigmoid(x), color='r', linestyle='-', label="Sigmoid")
ax.plot(x, softplus_func(x), color='b', linestyle='-', label="SoftPlus")

ax.grid()
ax.legend()
plt.ylim(-1, 5)

plt.show()
