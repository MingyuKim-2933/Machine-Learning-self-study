import numpy as np


def SSE(y, t):  # 오차제곱합
    return 0.5 * np.sum((y-t) ** 2)
