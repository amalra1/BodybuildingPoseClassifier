import os
import sys
import pickle
import numpy as np
import cv2
import mediapipe as mp
from src.pose_extractor import extract_landmarks
from src.config import RAW_DATA_DIR, PROTOTYPES_DIR, SKELETONS_DIR, VISUALIZATIONS_DIR

# Draws the raw MediaPipe landmarks onto the original image and saves it.
def save_skeleton_visualization(original_image_path, raw_landmarks, output_path):
    image = cv2.imread(original_image_path)
    if image is None: return

    annotated_image = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=raw_landmarks,
        connections=mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    )
    cv2.imwrite(output_path, annotated_image)

def train():
    """
    Processes training images, saves skeletons, generates visualizations,
    and creates a prototype vector for each pose class.
    """
    print("Starting training and visualization process...")
    os.makedirs(PROTOTYPES_DIR, exist_ok=True)
    os.makedirs(SKELETONS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    for pose_class_name in os.listdir(RAW_DATA_DIR):
        class_dir = os.path.join(RAW_DATA_DIR, pose_class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\nProcessing pose class: '{pose_class_name}'")
        landmark_vectors = []
        
        class_skeletons_dir = os.path.join(SKELETONS_DIR, pose_class_name)
        os.makedirs(class_skeletons_dir, exist_ok=True)
        
        # Create the visualizations folder for this class
        class_visualizations_dir = os.path.join(VISUALIZATIONS_DIR, pose_class_name)
        os.makedirs(class_visualizations_dir, exist_ok=True)

        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            image_path = os.path.join(class_dir, filename)
            print(f"  -> Processing image: {filename}")

            # Unpack the two values returned by the function
            landmark_vector, raw_landmarks = extract_landmarks(image_path)

            if landmark_vector is not None:
                # Use the 'landmark_vector' for training
                landmark_vectors.append(landmark_vector)
                skeleton_filename = f"{os.path.splitext(filename)[0]}.pkl"
                skeleton_path = os.path.join(class_skeletons_dir, skeleton_filename)
                with open(skeleton_path, 'wb') as f:
                    pickle.dump(landmark_vector, f)

                # Use the 'raw_landmarks' for visualization
                visualization_filename = f"{os.path.splitext(filename)[0]}_skeleton.jpg"
                visualization_path = os.path.join(class_visualizations_dir, visualization_filename)
                save_skeleton_visualization(image_path, raw_landmarks, visualization_path)
                print(f"    -> Visualization saved to '{visualization_path}'")
            else:
                print(f"    -> Warning: No pose detected in {filename}.")

        if not landmark_vectors:
            print(f"Training failed for class '{pose_class_name}'. No poses detected.")
            continue

        prototype_vector = np.mean(landmark_vectors, axis=0)
        prototype_vector /= np.linalg.norm(prototype_vector)
        prototype_path = os.path.join(PROTOTYPES_DIR, f"{pose_class_name}_prototype.pkl")
        with open(prototype_path, 'wb') as f:
            pickle.dump(prototype_vector, f)
            
        print(f"Successfully trained and saved prototype for '{pose_class_name}'.")

if __name__ == "__main__":
    train()