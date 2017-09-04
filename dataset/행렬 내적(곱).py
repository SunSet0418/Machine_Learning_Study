import numpy as np
import matplotlib.pyplot as plt

# 2차원 행렬
A = np.array([[1,2], [3,4]])
print('A Shape : ', A.shape)

# 2차원 행렬
B = np.array([[5,6], [7,8]])
print('B shape : ', B.shape)

# 행렬곱 dot 메소드
print(np.dot(A, B))