import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, "validation")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PROTOTYPES_DIR = os.path.join(PROCESSED_DATA_DIR, "prototypes")
SKELETONS_DIR = os.path.join(PROCESSED_DATA_DIR, "skeletons")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "reports", "visualizations")
REPORTS_DIR = os.path.join(BASE_DIR, "reports") 

# --- Model Configuration ---
# Similarity threshold to consider a pose as a match.
SIMILARITY_THRESHOLD = 0.94

# MediaPipe Pose solution settings
MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.5