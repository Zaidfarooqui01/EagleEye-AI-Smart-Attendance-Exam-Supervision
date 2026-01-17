# /app/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Database Configuration ---
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'eagle_eye.db')}"

# --- Data Paths ---
ENCODINGS_DIR = os.path.join(BASE_DIR, 'data', 'face_encodings')

# --- Output Paths ---
ATTENDANCE_REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'attendance_reports')
SUPERVISION_REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'supervision_reports')
VIOLATION_SNAPSHOTS_DIR = os.path.join(SUPERVISION_REPORTS_DIR, 'violation_snapshots')

# --- Model & Processing Settings ---
FACE_TOLERANCE = 0.6
FACE_DETECTION_MODEL = 'hog'
CAMERA_INDEX = 0
EYE_AR_THRESH = 0.25   # Eye Aspect Ratio threshold for blink detection

# --- Directory Initialization ---
def initialize_directories():
    """Creates all necessary data and output directories."""
    dirs_to_create = [
        ENCODINGS_DIR,
        ATTENDANCE_REPORTS_DIR,
        VIOLATION_SNAPSHOTS_DIR
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("All necessary directories are initialized.")

initialize_directories()
