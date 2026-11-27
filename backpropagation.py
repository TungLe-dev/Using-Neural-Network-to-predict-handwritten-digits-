import numpy as np
from activated import relu,softmax,relu_derivative,cross_entropy_loss
from data_constrain import X,y_onehot,y_raw
np.random.seed(42)

W1 = np.random.randn(784, 16) * np.sqrt(2.0 / 784)
b1 = np.zeros((1, 16))

W2 = np.random.randn(16, 16) * np.sqrt(2.0 / 16)
b2 = np.zeros((1, 16))

W3 = np.random.randn(16, 10) * np.sqrt(2.0 / 16)
b3 = np.zeros((1, 10))
def forward(X_batch):
    # Lớp ẩn 1: (N, 784) dot (784, 16) -> (N, 16)
    Z1 = np.dot(X_batch, W1) + b1
    A1 = relu(Z1)
    
    # Lớp ẩn 2: (N, 16) dot (16, 16) -> (N, 16)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # Lớp đầu ra: (N, 16) dot (16, 10) -> (N, 10)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3) # Đây chính là y_pred (xác suất dự đoán)
    
    return Z1, A1, Z2, A2, Z3, A3



def backward(X_batch, y_batch, Z1, A1, Z2, A2, A3):
    m = X_batch.shape[0]

    #Lớp 3
    dZ3 = (A3 - y_batch) / m
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    #Lớp 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Đạo hàm tại lớp 1 (Hidden 1)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_batch.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

epochs = 15
batch_size = 64
learning_rate = 0.1
num_samples = X.shape[0]

for epoch in range(epochs):
    # Xáo trộn ngẫu nhiên dữ liệu đầu mỗi Epoch để mạng không học vẹt thứ tự
    indices = np.random.permutation(num_samples)
    X_shuffled = X[indices]
    y_shuffled = y_onehot[indices]
    
    # Chia 60.000 ảnh thành các batch nhỏ (mỗi batch 64 ảnh)
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]
        
        # 1. Forward pass
        Z1, A1, Z2, A2, Z3, A3 = forward(X_batch)
        
        # 2. Backward pass (tính gradient)
        dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, y_batch, Z1, A1, Z2, A2, A3)
        
        # 3. Gradient Descent (cập nhật trọng số)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
    # Đánh giá Loss và độ chính xác (Accuracy) sau mỗi Epoch
    _, _, _, _, _, y_pred_all = forward(X)
    epoch_loss = cross_entropy_loss(y_onehot, y_pred_all)
    predictions = np.argmax(y_pred_all, axis=1)
    accuracy = np.mean(predictions == y_raw) * 100
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

# Lưu tất cả các ma trận trọng số và bias vào file 'model_weights.npz'
best_acc = 0.0

for epoch in range(epochs):
    # ... (code train giữ nguyên) ...
    
    if accuracy > best_acc:
        best_acc = accuracy
        np.savez('best_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
        print(f"--> Đã lưu model tốt nhất mới (Accuracy: {best_acc:.2f}%)")




