import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)] # 꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int)  # 부채붓꽃(Iris Setosa)인가?

# loss="perceptron", learning_rate="constant", eta0=1(학습률), penalty=None(규제 없음)인 SGDClassifier와 같습니다.
# 로지스틱 회귀와 달리 확률을 제공하지 않으며 고정된 임곗값을 기준으로 예측을 만듭니다.
per_clf = Perceptron()

per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)