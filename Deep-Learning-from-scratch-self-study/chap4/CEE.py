import numpy as np


def CEE(y, t):  # 교차 엔트로피
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# def CEE(y, t):  # 배치용 교차 엔트로피(원-핫 인코딩)
#     delta = 1e-7

#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + delta)) / batch_size


# def CEE(y, t):  # 배치용 교차 엔트로피(숫자 레이블)
#     delta = 1e-7

#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arrange(batch_size), t] + delta)) / batch_size
