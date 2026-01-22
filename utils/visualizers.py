import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import cv2


def visualize_predictions(X_test, y_test, model, class_names, model_type='rf', 
                          num_images=25, class_indices=None, original_class_ids=None, image_paths=None):
    if model_type == 'rf_hog':
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        reshape_dim = None
    elif model_type == 'vgg16':
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        for i in range(min(3, len(y_pred_proba))):
            top_indices = np.argsort(y_pred_proba[i])[-3:][::-1]
            top_probs = y_pred_proba[i][top_indices]
            print(f"Image {i}:")
            for idx, prob in zip(top_indices, top_probs):
                # Mapping từ output index sang class ID dựa vào class_indices
                if class_indices:
                    # Đảo ngược class_indices để lấy string class từ index
                    class_indices_inv = {v: k for k, v in class_indices.items()}
                    if idx in class_indices_inv:
                        class_id_str = class_indices_inv[idx]
                        class_id = int(class_id_str)
                        # Lấy tên class từ dictionary hoặc list
                        if isinstance(class_names, dict):
                            class_name = class_names.get(class_id, f"Unknown class {class_id}")
                        else:
                            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown class {class_id}"
                        print(f"  Class {class_id} ({class_name}): {prob:.4f}")
                    else:
                        print(f"  Index {idx} (Unknown): {prob:.4f}")
                else:
                    # Không có mapping, sử dụng index trực tiếp
                    if isinstance(class_names, dict):
                        class_name = class_names.get(idx, f"Unknown class {idx}")
                    else:
                        class_name = class_names[idx] if idx < len(class_names) else f"Unknown class {idx}"
                    print(f"  Class {idx} ({class_name}): {prob:.4f}")
    else:
        raise ValueError(f"Không hỗ trợ model type: {model_type}")
    
    # Giới hạn số lượng ảnh
    num_display = min(num_images, len(X_test))
    
    # Sử dụng sequential indices để dễ theo dõi
    indices = list(range(num_display))
    
    # Thiết lập layout
    n_cols = 5
    n_rows = (num_display + n_cols - 1) // n_cols
    plt.figure(figsize=(20, 4 * n_rows))
    
    # Tạo mapping ngược từ index trong output layer về class ID
    class_indices_inv = None
    if class_indices:
        class_indices_inv = {v: int(k) for k, v in class_indices.items()}
    
    # Hiển thị mỗi ảnh với dự đoán
    for i, idx in enumerate(indices):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Xử lý ảnh để hiển thị
        if model_type == 'rf' or model_type == 'random_forest':
            # Random Forest dùng ảnh flatten, cần reshape
            img = X_test[idx].reshape(reshape_dim)
            # Normalize ảnh về range 0-255 để hiển thị
            if img.max() <= 1.0:
                img = img * 255
        elif model_type == 'rf_hog':
            # Load ảnh gốc từ đường dẫn
            if image_paths is not None and idx < len(image_paths):
                try:
                    img_path = image_paths[idx]
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (32, 32))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Fallback to gray placeholder
                    img = np.ones((32, 32, 3)) * 128
            else:
                # Fallback to gray placeholder
                img = np.ones((32, 32, 3)) * 128
        elif model_type == 'vgg16':
            # VGG16 dùng preprocessing_input, cần denormalize
            img = X_test[idx].copy()
            # Denormalize ảnh về range 0-255 để hiển thị
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        
        img = img.astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
        
        # Xử lý class labels và set màu cho title
        if model_type == 'vgg16':
            # Lấy ground truth class
            true_class_id = int(original_class_ids[idx])
            
            # Lấy predicted class
            pred_idx = int(y_pred[idx])
            
            # Chuyển đổi từ output index sang class ID
            if class_indices_inv and pred_idx in class_indices_inv:
                pred_class_id = class_indices_inv[pred_idx]
            else:
                pred_class_id = pred_idx
            
            # Lấy tên class từ dictionary hoặc list
            if isinstance(class_names, dict):
                true_name = class_names.get(true_class_id, f"Unknown class {true_class_id}")
                pred_name = class_names.get(pred_class_id, f"Unknown class {pred_class_id}")
            else:
                true_name = class_names[true_class_id] if true_class_id < len(class_names) else f"Unknown class {true_class_id}"
                pred_name = class_names[pred_class_id] if pred_class_id < len(class_names) else f"Unknown class {pred_class_id}"
                
            # Set màu cho title
            color = 'green' if true_class_id == pred_class_id else 'red'
                
        else:
            # Xử lý bình thường cho Random Forest (cả simple và HOG)
            true_class = int(y_test[idx])
            pred_class = int(y_pred[idx])
            
            # Lấy tên class từ dictionary hoặc list
            if isinstance(class_names, dict):
                true_name = class_names.get(true_class, f"Unknown class {true_class}")
                pred_name = class_names.get(pred_class, f"Unknown class {pred_class}")
            else:
                true_name = class_names[true_class] if true_class < len(class_names) else f"Unknown class {true_class}"
                pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Unknown class {pred_class}"
                
            # Set màu cho title
            color = 'green' if true_class == pred_class else 'red'
        
        # Hiển thị title với thông tin
        plt.title(f"Actual: {true_name}\nPredict: {pred_name}",
                  color=color, fontsize=10)
    
    # Hiển thị plot
    plt.tight_layout()
    plt.suptitle(f"{model_type.upper()} Model Predictions", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Tính accuracy
    if model_type == 'vgg16':
        # Chuyển đổi predictions về class IDs gốc
        if class_indices_inv:
            corrected_preds = []
            for pred_idx in y_pred:
                if pred_idx in class_indices_inv:
                    corrected_preds.append(class_indices_inv[pred_idx])
                else:
                    corrected_preds.append(pred_idx)
            
            accuracy = np.mean(np.array(corrected_preds) == original_class_ids)
        else:
            # Nếu không có class_indices, tính trực tiếp
            accuracy = np.mean(y_pred == original_class_ids)
        
        print(f"\nAccuracy on test data: {accuracy:.4f}")
    else:
        accuracy = np.mean(y_test == y_pred)
        print(f"\nAccuracy on test data: {accuracy:.4f}")
    
    return accuracy
