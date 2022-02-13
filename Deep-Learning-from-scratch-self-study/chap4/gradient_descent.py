from numerical_gradient import _numerical_gradient_1d


def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = _numerical_gradient_1d(f, x)
        x -= lr * grad
    return x
