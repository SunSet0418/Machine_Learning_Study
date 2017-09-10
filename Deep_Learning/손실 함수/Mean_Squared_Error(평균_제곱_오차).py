import numpy as np

# y, t 는  numpy 배열
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)