from PIL import Image
import numpy as np

def analize(IMAGE_PATH):
    img = Image.open(IMAGE_PATH).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Tự động đảo màu nếu ảnh có nền trắng chữ đen
    if arr[0, 0] > 127:
        arr = 255.0 - arr

    # Cắt sát viền chữ số, thu về khung 20x20 rồi đặt vào giữa khung đen 28x28
    coords = np.argwhere(arr > 20)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = arr[y0:y1, x0:x1]
        
        h, w = cropped.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        img_20 = Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.array(img_20, dtype=np.float32)
    else:
        canvas = np.zeros((28, 28), dtype=np.float32)
    return canvas
