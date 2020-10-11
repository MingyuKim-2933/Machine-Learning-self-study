import sys, os
sys.path.append(os.pardir)
import numpy as np
from chap3.softmax import softmax
from chap4.CEE import CEE
from chap4.numerical_gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화 randn(2, 3) : 난수로 구성된 배열를 반환합니다.(2X3 행렬을 반환)

    def predict(self, x):  # 예측을 수행한다.
        return np.dot(x, self.W)

    def loss(self, x, t):  # 손실함수의 값을 구한다.
        z = self.predict(x)
        y = softmax(z)
        loss = CEE(y, t)

        return loss


net = simpleNet()
print(net.W)  # 난수로 구성된 2X3 행렬을 반환
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))  # 최대값 반환
if np.argmax(p) == 0:
    t = np.array([1, 0, 0])
elif np.argmax(p) == 1:
    t = np.array([0, 1, 0])
else:
    t = np.array([0, 0, 1])
print(t)
print(net.loss(x, t))  # CEE(교차 엔트로피)를 활용한 손실함수의 값 구하기
print()


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)

f = lambda w: net.loss(x, t)  # 람다 기법으로 함수 정의(파이썬에서 간단한 함수는 람다 기법을 사용하면 더 편하다.)
dW = numerical_gradient(f, net.W)
print(dW)
