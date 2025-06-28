# src/scripts/validation.py

import os
import time
from src.config import VALIDATION_DATA_DIR
from .prediction import predict

def validate_all():
    """
    Runs prediction on all images within the validation directory,
    processing them one by one.
    """
    if not os.path.exists(VALIDATION_DATA_DIR) or not os.listdir(VALIDATION_DATA_DIR):
        print(f"Validation directory is empty or not found at: '{VALIDATION_DATA_DIR}'")
        return

    print("--- Starting Full Validation Process ---")
    
    # Loop through each pose class in the validation folder
    for pose_class_name in sorted(os.listdir(VALIDATION_DATA_DIR)):
        class_dir = os.path.join(VALIDATION_DATA_DIR, pose_class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [f for f in sorted(os.listdir(class_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            continue

        print(f"\n===== Validating Class: {pose_class_name} =====")

        for filename in image_files:
            image_path = os.path.join(class_dir, filename)
            
            print(f"\n--- Validating: {pose_class_name}/{filename} ---")
            
            # Call the existing predict function for the image
            predict(image_path)
            
            # Pause for a couple of seconds to make output readable
            time.sleep(2)
            
    print("\n--- Validation Process Finished ---")