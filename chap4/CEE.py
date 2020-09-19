import numpy as np


def CEE(y, t):  # 교차 엔트로피
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))