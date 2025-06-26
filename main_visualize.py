import sys
import cv2
import mediapipe as mp
import os
from src.config import VISUALIZATIONS_DIR, MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE

# Loads an image, detects the pose, draws the skeleton, and saves
# the result to a new image file.
def visualize_single_image(image_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    annotated_image = image.copy()
    
    if results.pose_landmarks:
        print(f"Pose successfully detected in '{image_path}'!")
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
    else:
        print(f"Warning: No pose was detected in '{image_path}'.")

    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_visualization{ext}"
    output_path = os.path.join(VISUALIZATIONS_DIR, output_filename)
    
    cv2.imwrite(output_path, annotated_image)
    print(f"\nProcessed image saved to: '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_visualize.py <path_to_your_image>")
    else:
        image_to_visualize = sys.argv[1]
        visualize_single_image(image_to_visualize)