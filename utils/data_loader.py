import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessors import preprocess_vgg16, preprocess_rf_hog
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import json


def load_class_indices(models_dir):
    class_indices_path = os.path.join(models_dir, 'class_indices.json')
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print(f"Loaded class indices from {class_indices_path}")
        print(f"Class indices: {class_indices}")
        return class_indices


def load_test_data(test_csv, model_type='rf_hog', max_samples=None):
    test_data = pd.read_csv(test_csv)
    if max_samples:
        test_data = test_data.sample(min(max_samples, len(test_data)), random_state=42)
    data_dir = os.path.dirname(test_csv)
    models_dir = os.path.join(os.path.dirname(data_dir), 'models')
    test_df = test_data.copy()
    sample_path = test_df['Path'].iloc[0] if len(test_df) > 0 else ""
    
    if sample_path.startswith('Test/') or sample_path.startswith('Test\\'):
        test_df['Path'] = test_df['Path'].apply(lambda x: os.path.join(data_dir, x))
    else:
        test_dir = os.path.join(data_dir, 'Test')
        test_df['Path'] = test_df['Path'].apply(lambda x: os.path.join(test_dir, x))
    test_df['Path'] = test_df['Path'].apply(os.path.normpath)
    
    if model_type == 'rf_hog':
        # Tiền xử lý cho Random Forest với HOG features
        print(f"Processing {len(test_df)} images for Random Forest model (HOG features):")
        X_test = []
        image_paths = []
        y_test = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting HOG features"):
            try:
                img_path = row['Path']
                features = preprocess_rf_hog(img_path)
                X_test.append(features)
                image_paths.append(img_path)
                y_test.append(row['ClassId'])
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test)
        image_paths = np.array(image_paths)
        
        return X_test, y_test, image_paths
    
    elif model_type == 'vgg16':
        print(f"Using ImageDataGenerator for VGG16 with {len(test_df)} images")
        original_class_ids = test_df['ClassId'].values
        class_indices = load_class_indices(models_dir)
        test_df_copy = test_df.copy()
        test_df_copy['ClassId'] = test_df_copy['ClassId'].astype(str)
        datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        generator = datagen.flow_from_dataframe(
            dataframe=test_df_copy,
            x_col='Path',
            y_col='ClassId',
            target_size=(224, 224),
            batch_size=len(test_df_copy),
            class_mode='categorical',
            shuffle=False
        )

        X_batch, y_batch = next(generator)
        return X_batch, y_batch, class_indices, original_class_ids  
    else:
        raise ValueError(f"Không hỗ trợ model type: {model_type}")

def load_rf_data(test_df):
    X_test = []
    y_test = test_df['ClassId'].values
    
    print(f"Processing {len(test_df)} images for Random Forest model:")
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = row['Path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (32, 32)) 
        img_flat = img.flatten() / 255.0  
        X_test.append(img_flat)
    
    X_test = np.array(X_test, dtype=np.float32)
    return X_test, y_test

def load_vgg16_data(test_df):
    test_df_copy = test_df.copy()
    test_df_copy['ClassId'] = test_df_copy['ClassId'].astype(str)
    print(f"Using ImageDataGenerator for VGG16 with {len(test_df)} images")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_dataframe(
        dataframe=test_df_copy,
        x_col='Path',
        y_col='ClassId',
        target_size=(224, 224),  
        batch_size=len(test_df),  
        class_mode='categorical',
        shuffle=False
    )
    X_batch, y_batch_onehot = next(generator)
    y_batch_indices = np.argmax(y_batch_onehot, axis=1)
    class_indices = generator.class_indices
    print(f"Class indices from generator: {class_indices}")
    original_class_ids = test_df['ClassId'].values
    return X_batch, y_batch_indices, class_indices, original_class_ids
