import os
import pickle
import numpy as np
from src.config import RAW_DATA_DIR, PROTOTYPES_DIR, SKELETONS_DIR
from src.pose_extractor import extract_landmarks

def train():
    """
    Processes all subdirectories in the raw data folder, treating each
    as a pose class. It extracts landmarks, saves individual skeletons,
    and creates a prototype vector for each pose.
    """
    print("Starting training process...")
    os.makedirs(PROTOTYPES_DIR, exist_ok=True)
    os.makedirs(SKELETONS_DIR, exist_ok=True)

    # Loop through each subdirectory
    for pose_class_name in os.listdir(RAW_DATA_DIR):
        class_dir = os.path.join(RAW_DATA_DIR, pose_class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\nProcessing pose class: '{pose_class_name}'")
        landmark_vectors = []
        
        # Create directories for processed data
        class_skeletons_dir = os.path.join(SKELETONS_DIR, pose_class_name)
        os.makedirs(class_skeletons_dir, exist_ok=True)

        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            image_path = os.path.join(class_dir, filename)
            print(f"  -> Processing image: {filename}")

            landmarks = extract_landmarks(image_path)
            if landmarks is not None:
                landmark_vectors.append(landmarks)
                
                # Save the individual skeleton vector
                skeleton_filename = f"{os.path.splitext(filename)[0]}.pkl"
                skeleton_path = os.path.join(class_skeletons_dir, skeleton_filename)
                with open(skeleton_path, 'wb') as f:
                    pickle.dump(landmarks, f)
            else:
                print(f"    -> Warning: No pose detected in {filename}.")

        if not landmark_vectors:
            print(f"Training failed for class '{pose_class_name}'. No poses detected.")
            continue

        # Calculate the mean "prototype" vector for the class
        prototype_vector = np.mean(landmark_vectors, axis=0)
        prototype_vector /= np.linalg.norm(prototype_vector) # Normalize the final prototype

        # Save the prototype vector
        prototype_path = os.path.join(PROTOTYPES_DIR, f"{pose_class_name}_prototype.pkl")
        with open(prototype_path, 'wb') as f:
            pickle.dump(prototype_vector, f)

        print(f"Successfully trained and saved prototype for '{pose_class_name}' at '{prototype_path}'.")

if __name__ == "__main__":
    train()