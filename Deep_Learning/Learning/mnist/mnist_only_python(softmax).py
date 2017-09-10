from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle as pickle

# 분류를 위한 softmax 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.sum(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y



# 이미지 show함수
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()




# MNIST 데이터 mnist.py로부터  읽어오기
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test




# 미리 설정된 편향과 가중치 함수
def init_network():
    with open("../../dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network




# 활성화 함수 ReLu
def relu(x):
    return np.maximum(0, x)

# 활성화 함수 Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))


# 신경망 함수
# b = 편향, W = 가중치 (미리 계산된 값)
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = softmax(a3)

    return y


#### MAIN ####
x, t = get_data()
network = init_network()

# 정확도 변수
accuracy_cnt = 0

# 정확도 측정
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("데이터 양 : ",len(x))
print(accuracy_cnt)
print("정확도 : "+str(float(accuracy_cnt)/len(x)))

