import numpy as np
import matplotlib.pyplot as plt


# 항등 함수

def identity_funtion(x):
    return x


# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 데이터 초기화 함수
# 이미 정해진 가중치와 편향
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.2, 0.3])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 순방향 연산 함수
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    #     plt.plot(x, z1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    #     plt.plot(z1, z2)
    a3 = np.dot(z2, W3) + b3
    y = identity_funtion(a3)
    #     plt.show()
    return y


network = init_network()

x = np.array([1.0, 0.5])
y = forward(network, x)

print(y)

