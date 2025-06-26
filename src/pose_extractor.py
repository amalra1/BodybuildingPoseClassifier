import cv2
import mediapipe as mp
import numpy as np
from src.config import MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE

# Initialize MediaPipe 
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)

def extract_landmarks(image_path):
    """
    Receives an image path, detects the pose, and returns
    a normalized landmark vector.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # MediaPipe only accepts RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    # Extract landmarks and creates a flat vector with (x, y, visibility)
    landmarks = results.pose_landmarks.landmark
    pose_vector = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks]).flatten()

    # L2 normalization to make the pose vector comparable regardless of size
    norm_vector = pose_vector / np.linalg.norm(pose_vector)

    return norm_vector