import numpy as np
import data_constrain
def relu(x):
    return np.maximum(0,x)
def relu_derivative(Z):
    return (Z > 0).astype(np.float32)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    return exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)
def cross_entropy_loss(y_true, y_pred):
    # Cộng thêm 1e-9 để tránh lỗi toán học log(0)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / N
    return loss



