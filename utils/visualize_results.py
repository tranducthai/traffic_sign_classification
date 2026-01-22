import os
import joblib
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_test_data
from visualizers import visualize_predictions

class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',      
    2: 'Speed limit (50km/h)',       
    3: 'Speed limit (60km/h)',      
    4: 'Speed limit (70km/h)',    
    5: 'Speed limit (80km/h)',      
    6: 'End of speed limit (80km/h)',     
    7: 'Speed limit (100km/h)',    
    8: 'Speed limit (120km/h)',     
    9: 'No passing',   
    10: 'No passing veh over 3.5 tons',     
    11: 'Right-of-way at intersection',     
    12: 'Priority road',    
    13: 'Yield',     
    14: 'Stop',       
    15: 'No vehicles',       
    16: 'Veh > 3.5 tons prohibited',       
    17: 'No entry',       
    18: 'General caution',     
    19: 'Dangerous curve left',      
    20: 'Dangerous curve right',   
    21: 'Double curve',      
    22: 'Bumpy road',     
    23: 'Slippery road',       
    24: 'Road narrows on the right',  
    25: 'Road work',    
    26: 'Traffic signals',      
    27: 'Pedestrians',     
    28: 'Children crossing',     
    29: 'Bicycles crossing',       
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',      
    32: 'End speed + passing limits',      
    33: 'Turn right ahead',     
    34: 'Turn left ahead',       
    35: 'Ahead only',      
    36: 'Go straight or right',      
    37: 'Go straight or left',      
    38: 'Keep right',     
    39: 'Keep left',      
    40: 'Roundabout mandatory',     
    41: 'End of no passing',      
    42: 'End no passing veh > 3.5 tons'
}

NUM_IMAGES = 25

def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    test_csv = os.path.join(data_dir, 'Test.csv')
    models_dir = os.path.join(project_root, 'models')
    return {
        'project_root': project_root,
        'data_dir': data_dir,
        'test_csv': test_csv,
        'models_dir': models_dir
    }

def visualize_rf_hog_model(paths):
    model_path = os.path.join(paths['models_dir'], 'rf_hog.joblib')
    try:
        model = joblib.load(model_path)
        X_test, y_test, image_paths = load_test_data(
            paths['test_csv'], 
            model_type='rf_hog',
            max_samples=NUM_IMAGES
        )
        accuracy = visualize_predictions(
            X_test, y_test, model, class_names,
            model_type='rf_hog',
            num_images=NUM_IMAGES,
            image_paths=image_paths
        )
        
        return {'name': 'random_forest_hog', 'accuracy': accuracy}
    except Exception as e:
        print(f"Error processing Random Forest HOG model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_vgg16_model(paths):
    model_path = os.path.join(paths['models_dir'], 'VGG16_new.keras')
    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()
        X_test, y_test, class_indices, original_class_ids = load_test_data(
            paths['test_csv'], 
            model_type='vgg16',
            max_samples=NUM_IMAGES
        )
        accuracy = visualize_predictions(
            X_test, y_test, model, class_names,
            model_type='vgg16',
            num_images=NUM_IMAGES,
            class_indices=class_indices,
            original_class_ids=original_class_ids
        )
        
        return {'name': 'vgg16', 'accuracy': accuracy}
    except Exception as e:
        print(f"Error processing VGG16 model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize Traffic Sign Recognition Models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['rf_hog', 'vgg16', 'all'],
                        help='Model to visualize (rf_hog, vgg16, or all)')
    args = parser.parse_args()
    paths = setup_paths()
    results = []
    
    if args.model in ['rf_hog', 'all']:
        rf_hog_result = visualize_rf_hog_model(paths)
        if rf_hog_result:
            results.append(rf_hog_result)
    
    if args.model in ['vgg16', 'all']:
        vgg16_result = visualize_vgg16_model(paths)
        if vgg16_result:
            results.append(vgg16_result)

if __name__ == "__main__":
    main()