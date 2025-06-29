import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from src.config import PROTOTYPES_DIR, SIMILARITY_THRESHOLD

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
_pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

def extract_landmarks_fast(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _pose.process(image_rgb)

    if not result.pose_landmarks:
        return None, None

    landmarks = result.pose_landmarks.landmark

    # Obtem x, y, z como numpy array (33, 3)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Centraliza no centro do quadril (landmark 0)
    center = coords[0]
    coords -= center

    # Normaliza pela distância entre ombros (landmarks 11 e 12)
    shoulder_dist = np.linalg.norm(coords[11] - coords[12])
    if shoulder_dist > 0:
        coords /= shoulder_dist

    # Achata para vetor 1D
    vector = coords.flatten()

    return vector, result.pose_landmarks

def load_prototypes():
    prototypes = {}
    for filename in os.listdir(PROTOTYPES_DIR):
        if filename.endswith("_prototype.pkl"):
            class_name = filename.replace("_prototype.pkl", "")
            with open(os.path.join(PROTOTYPES_DIR, filename), 'rb') as f:
                prototypes[class_name] = pickle.load(f)
    return prototypes

def predict_from_frame_fast(frame, prototypes):
    test_vector, pose_landmarks = extract_landmarks_fast(frame)
    if test_vector is None:
        return None, None, "No pose detected", None

    test_vector /= np.linalg.norm(test_vector)

    best_match = None
    max_similarity = -1

    for class_name, prototype in prototypes.items():
        similarity = np.dot(test_vector, prototype)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name

    return best_match, max_similarity, None, pose_landmarks

def detect_real_time():
    """
    Wrapper function for real-time pose detection using webcam.
    This is used by the 'real_time' command in main.py.
    """

    print("Iniciando detecção em tempo real...")
    prototypes = load_prototypes()
    if not prototypes:
        print("Nenhum protótipo encontrado.")
        return

    cap = cv2.VideoCapture(0)

    last_text = "Procurando pose..."
    last_color = (0, 165, 255)
    last_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        best_match, similarity, error, pose_landmarks = predict_from_frame_fast(frame, prototypes)

        if error:
            last_text = error
            last_color = (0, 0, 255)
            last_landmarks = None
        else:
            last_text = f"{best_match} ({similarity:.2%})"
            last_color = (0, 255, 0) if similarity >= SIMILARITY_THRESHOLD else (0, 165, 255)
            last_landmarks = pose_landmarks

        # Draw landmarks if available
        if last_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                last_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2),
            )

        # Overlay text
        cv2.putText(frame, last_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_color, 2)
        cv2.imshow("Pose Detection (Realtime)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

