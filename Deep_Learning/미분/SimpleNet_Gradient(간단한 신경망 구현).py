import sys, os
sys.path.append(os.pardir)
import numpy as np

# softmax 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.sum(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 교차 엔트로피 함수
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


# 기울기 함수
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h)계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원

    return grad

class simpleNet:

    #초기화
    def __init__(self):
        self.W = np.random.randn(2,3)

    #예측 수행
    def predict(self, x):
        return np.dot(x, self.W)

    #손실값
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print("가중치 매개변수 : ",net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
t = np.array([0,0,1])
print(net.loss(x, t))#손실값

