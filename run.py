import numpy as np
import pandas as pd
from PIL import Image
import change_image
PATH = "mnist/mnist_train.csv/mnist_train.csv"
# --- BƯỚC 1: ĐỊNH NGHĨA LẠI HÀM KÍCH HOẠT & FORWARD ---
# (Bắt buộc phải giống hệt lúc bạn dùng để train)
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward(X_batch, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X_batch, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return A3

# --- BƯỚC 2: NẠP TRỌNG SỐ ĐÃ LƯU TỪ FILE .NPZ ---
weights = np.load('best_weights.npz') 
W1, b1 = weights['W1'], weights['b1']
W2, b2 = weights['W2'], weights['b2']
W3, b3 = weights['W3'], weights['b3']

IMAGE_PATH = "custom_data/image4.png"

canvas = change_image.analize(IMAGE_PATH)
X_input = canvas.reshape(1, 784) / 255.0

# --- BƯỚC 4: CHẠY DỰ ĐOÁN ---
y_pred_probs = forward(X_input, W1, b1, W2, b2, W3, b3)

predicted_digit = np.argmax(y_pred_probs)
confidence = np.max(y_pred_probs) * 100

print(f"\n========================================")
print(f"Mạng đoán đây là số: {predicted_digit} (Độ tự tin: {confidence:.2f}%)")
print(f"========================================")

for digit, prob in enumerate(y_pred_probs[0]):
    print(f"Số {digit}: {prob * 100:6.2f}%")

























# # --- BƯỚC 3: ĐỌC VÀ TIỀN XỬ LÝ FILE DỮ LIỆU TEST ---
# # Lưu ý: Kiểm tra lại tên file test trong thư mục mnist của bạn (thường là mnist_test.csv)
# test_data = pd.read_csv(PATH, header=None).to_numpy()

# y_test_raw = test_data[:, 0].astype(int)                 # Nhãn thật (10.000 số)
# X_test = test_data[:, 1:].astype(np.float32) / 255.0     # Ảnh đã chuẩn hóa [0, 1]

# # --- BƯỚC 4: CHẠY DỰ ĐOÁN VÀ TÍNH ĐỘ CHÍNH XÁC ---
# y_pred_probs = forward(X_test, W1, b1, W2, b2, W3, b3)

# # Lấy vị trí có xác suất lớn nhất làm con số dự đoán
# predictions = np.argmax(y_pred_probs, axis=1)

# # # Tính tỷ lệ đoán đúng trên toàn bộ tập test
# # accuracy = np.mean(predictions == y_test_raw) * 100
# # print(f"Tổng số ảnh test: {len(y_test_raw)}")
# # print(f"Số ảnh đoán đúng: {np.sum(predictions == y_test_raw)}")
# # print(f"Độ chính xác trên tập Test: {accuracy:.2f}%")

# # # In thử 10 kết quả đầu tiên để đối chiếu bằng mắt
# # print("\n--- KIỂM TRA THỬ 10 ẢNH ĐẦU TIÊN ---")

# # for i in range(10):
# #     status = "ĐÚNG" if predictions[i] == y_test_raw[i] else "SAI"
# #     print(f"Ảnh {i+1:02d} | Đáp án thật: {y_test_raw[i]} | Mạng đoán: {predictions[i]} -> {status}")



