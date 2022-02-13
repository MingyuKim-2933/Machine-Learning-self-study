import numpy as np
from gradient_descent import gradient_descent


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
lr = 1e-5
while lr <= 10:
    y = gradient_descent(function_2, init_x=init_x, lr=lr, step_num=100)
    if lr >= 10:
        print("학습률 :", '%-10.4f' % lr, "   ", "최솟값 : ", y)
    else:
        print("학습률 :", '%-10.5f' % lr, "   ", "최솟값 : ", y)

    lr *= 10
