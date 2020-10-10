# 나쁜 구현 예
def bad_numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

# 두 가지 개선점이 발생
# 1. 너무 작은 값을 이용하면 컴퓨터로 계산하는데 문제가 된다.
# 2. h를 무한히 0으로 좁히는 것이 불가능해 진정한 미분과 이번 구현의 값은 엄밀히 일치하지 않는다.


# 좋은 구현 예
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2 * h)
