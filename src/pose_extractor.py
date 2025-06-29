import cv2
import mediapipe as mp
import numpy as np
from src.config import MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE

mp_pose = mp.solutions.pose

def extract_landmarks(image_path):
    """
    Extract normalized landmarks from an image.
    Standardizes by centering on the nose and scaling by the distance between shoulders.
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

    landmarks = results.pose_landmarks.landmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Centralize on nose (landmark 0)
    center = coords[0]
    coords -= center

    # Normalize by the distance between shoulders (landmarks 11 and 12)
    shoulder_dist = np.linalg.norm(coords[11] - coords[12])
    if shoulder_dist > 0:
        coords /= shoulder_dist

    # Flatten to a 1D vector
    norm_vector = coords.flatten()
    return norm_vector, results.pose_landmarks
