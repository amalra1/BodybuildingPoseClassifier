# src/scripts/real_time.py

import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from src.config import PROTOTYPES_DIR, SIMILARITY_THRESHOLD, MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE

# --- MediaPipe Initialization for Real-Time ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
realtime_pose_detector = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, # Usamos complexidade 1 para maior velocidade
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

def extract_and_normalize_for_comparison(pose_landmarks_object):
    """
    Takes a MediaPipe pose landmarks object and normalizes it using the
    standard L2-norm method, consistent with the training script.
    """
    pose_vector = np.array([[lm.x, lm.y, lm.visibility] for lm in pose_landmarks_object.landmark]).flatten()
    
    norm_vector = pose_vector / np.linalg.norm(pose_vector)
    
    return norm_vector

def load_prototypes():
    """Loads all trained pose prototypes from the disk."""
    prototypes = {}
    if not os.path.exists(PROTOTYPES_DIR):
        return None
    for filename in os.listdir(PROTOTYPES_DIR):
        if filename.endswith("_prototype.pkl"):
            class_name = filename.replace("_prototype.pkl", "")
            with open(os.path.join(PROTOTYPES_DIR, filename), 'rb') as f:
                prototypes[class_name] = pickle.load(f)
    return prototypes

def predict_from_frame(raw_landmarks, prototypes):
    """Runs the prediction comparison on detected landmarks."""
    test_vector = extract_and_normalize_for_comparison(raw_landmarks)

    best_match = None
    max_similarity = -1

    for class_name, prototype in prototypes.items():
        similarity = np.dot(test_vector, prototype)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name

    return best_match, max_similarity


def detect_real_time():
    """Main function to run real-time pose detection using a webcam."""
    print("Starting real-time detection...")
    prototypes = load_prototypes()
    if not prototypes:
        print("Error: No prototypes found. Please run 'main.py train' first.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detects pose on frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = realtime_pose_detector.process(image_rgb)

        display_text = "No pose detected"
        text_color = (0, 0, 255) # Red

        # 2. If pose found, compare with prototypes
        if results.pose_landmarks:
            best_match, similarity = predict_from_frame(results.pose_landmarks, prototypes)
            
            display_text = f"{best_match} ({similarity:.2%})"
            text_color = (0, 255, 0) if similarity >= SIMILARITY_THRESHOLD else (0, 165, 255)
            
            # Draws skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2),
            )

        # Pose label on top
        cv2.rectangle(frame, (0,0), (640, 40), (0,0,0), -1)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.imshow("Real-Time Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")