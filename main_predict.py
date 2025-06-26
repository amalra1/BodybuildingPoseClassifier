import sys
import os
import pickle
import numpy as np
from src.config import PROTOTYPES_DIR, SIMILARITY_THRESHOLD
from src.pose_extractor import extract_landmarks

def predict(image_path):
    """
    Loads all trained prototypes, extracts landmarks from the new image,
    and finds the best matching pose.
    """
    # Load all available pose prototypes
    prototypes = {}
    try:
        for filename in os.listdir(PROTOTYPES_DIR):
            if filename.endswith(".pkl"):
                pose_name = filename.replace("_prototype.pkl", "")
                with open(os.path.join(PROTOTYPES_DIR, filename), 'rb') as f:
                    prototypes[pose_name] = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Prototypes directory '{PROTOTYPES_DIR}' not found.")
        print("Please run 'main_train.py' first.")
        return

    if not prototypes:
        print("Error: No prototypes found. Please run 'main_train.py' first.")
        return

    # Extract landmarks from the test image
    test_vector = extract_landmarks(image_path)
    if test_vector is None:
        print("Could not detect a pose in the test image.")
        return

    # Calculate similarity against each prototype
    best_match = None
    max_similarity = -1

    for pose_name, prototype_vector in prototypes.items():
        cosine_similarity = np.dot(test_vector, prototype_vector)
        if cosine_similarity > max_similarity:
            max_similarity = cosine_similarity
            best_match = pose_name

    # Compare with our confidence threshold
    print("--- POSE ANALYSIS ---")
    print(f"Best match: '{best_match}'")
    print(f"Similarity Score: {max_similarity:.2%}")
    print(f"Confidence Threshold: {SIMILARITY_THRESHOLD:.2%}")

    if max_similarity >= SIMILARITY_THRESHOLD:
        print(f"\nResult: The detected pose is '{best_match}'.")
    else:
        print("\nResult: The pose does not match any known prototype with enough confidence.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_predict.py <path_to_your_image>")
    else:
        image_to_test = sys.argv[1]
        predict(image_to_test)