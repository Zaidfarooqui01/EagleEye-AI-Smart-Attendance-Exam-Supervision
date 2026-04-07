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
CAMERA_INDEX = 0           # Default integrated camera (0)
EYE_AR_THRESH = 0.25       # Eye Aspect Ratio threshold for blink detection

# --- AI Monitoring / Screenshot Settings ---
# Screenshots are captured ONLY on violations, with a 2-second cooldown
VIOLATION_SCREENSHOT_COOLDOWN = 2.0   # Seconds between snapshots per violation type
SNAPSHOT_ON_SEVERITY = ['high']        # Only capture for high severity

# --- App Security ---
SECRET_KEY = os.environ.get('EAGLE_EYE_SECRET', 'eagle_eye_dev_secret_2026_change_in_prod')

# --- Directory Initialization ---
def initialize_directories():
    """Creates all necessary data and output directories."""
    dirs_to_create = [
        ENCODINGS_DIR,
        ATTENDANCE_REPORTS_DIR,
        VIOLATION_SNAPSHOTS_DIR,
        os.path.join(BASE_DIR, 'data'),
        os.path.join(BASE_DIR, 'logs'),
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

initialize_directories()
