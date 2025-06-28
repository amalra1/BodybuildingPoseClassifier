# src/scripts/prediction.py
import os
import pickle
import numpy as np
from src.config import PROTOTYPES_DIR, SIMILARITY_THRESHOLD
from src.pose_extractor import extract_landmarks

def get_prediction(image_path):
    """
    Core logic: loads prototypes, extracts landmarks, and returns the prediction details.
    This function does NOT print, it just returns data.
    """
    prototypes = {}
    try:
        for filename in os.listdir(PROTOTYPES_DIR):
            if filename.endswith(".pkl"):
                pose_name = filename.replace("_prototype.pkl", "")
                with open(os.path.join(PROTOTYPES_DIR, filename), 'rb') as f:
                    prototypes[pose_name] = pickle.load(f)
    except FileNotFoundError:
        return None, None, "Prototypes directory not found."

    if not prototypes:
        return None, None, "No prototypes found."

    test_vector, _ = extract_landmarks(image_path)
    if test_vector is None:
        return None, None, "Could not detect a pose in the test image."

    best_match = None
    max_similarity = -1
    for pose_name, prototype_vector in prototypes.items():
        cosine_similarity = np.dot(test_vector, prototype_vector)
        if cosine_similarity > max_similarity:
            max_similarity = cosine_similarity
            best_match = pose_name

    return best_match, max_similarity, None # Return results, no error

def predict(image_path):
    """
    Wrapper function that gets a prediction and prints it to the console.
    This is used by the 'predict' command in main.py.
    """
    best_match, max_similarity, error = get_prediction(image_path)

    if error:
        print(f"Error: {error}")
        return

    print("\n--- POSE ANALYSIS ---")
    print(f"Best match: '{best_match}'")
    print(f"Similarity Score: {max_similarity:.2%}")
    print(f"Confidence Threshold: {SIMILARITY_THRESHOLD:.2%}")

    if max_similarity >= SIMILARITY_THRESHOLD:
        print(f"\nResult: The detected pose is '{best_match}'.")
    else:
        print("\nResult: The pose does not match any known prototype with enough confidence.")