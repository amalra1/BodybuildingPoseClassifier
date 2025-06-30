import cv2
import mediapipe as mp
import numpy as np
from src.config import MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE

mp_pose = mp.solutions.pose

def extract_landmarks(image_path):
    """
    Receives an image path, creates a NEW pose detector, detects the pose,
    and returns a normalized vector and the raw landmarks object.
    """
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None, None

    # 1. Prepare the normalized vector for machine learning
    landmarks_for_vector = results.pose_landmarks.landmark
    pose_vector = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks_for_vector]).flatten()
    norm_vector = pose_vector / np.linalg.norm(pose_vector)
    
    # 2. Get the raw landmarks for visualization
    raw_landmarks_for_drawing = results.pose_landmarks

    # Return both items
    return norm_vector, raw_landmarks_for_drawing