# IT3190-2024.2-GR24-Traffic Sign Classification

## Cài đặt:

### 1. Tạo môi trường ảo:

Sử dụng **Python 3.9** để tương thích với các phiên bản thư viện cũ như **sklearn==1.2.2**

```bash
py -3.9 -m venv skl122env
```
### 2. Kích hoạt môi trường ảo:
Trên Windows:

```bash
skl122env\Scripts\activate
```

### 3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Visualize kết quả dự đoán trên tập test:
### Random Forest Model:
```bash
python utils/visualize_results.py --model rf
```

### VGG16:
```bash
python utils/visualize_results.py --model vgg16
```

