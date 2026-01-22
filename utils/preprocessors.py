import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage import feature

def preprocess_rf_hog(img_path, target_size=(32, 32)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hog_features = feature.hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        feature_vector=True
    )
    color_features = []
    for channel in range(3): 
        hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
        color_features.extend(hist.flatten())
    small_img = cv2.resize(gray_img, (16, 16))
    pixel_features = small_img.flatten() / 255.0
    
    # Kết hợp tất cả features
    all_features = np.concatenate([hog_features, color_features, pixel_features])
    return all_features

def preprocess_vgg16(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, target_size)          
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)     
    return img_array