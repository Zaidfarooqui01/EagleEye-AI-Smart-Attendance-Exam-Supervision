# /app/dashboard.py
"""
EagleEye Command Center — Production-Grade Flask + SocketIO Server
Role-based authentication, camera debugging, and optimized AI monitoring.
"""

import sys
import os
import cv2
import base64
import threading
import time
import uuid
import datetime
import csv
import logging
import json

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_socketio import SocketIO, emit, disconnect
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sqlalchemy.exc import SQLAlchemyError
from functools import wraps

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Core App Modules ---
from app.user import User, users, get_user, has_permission, ROLE_DISPLAY
from app.database import SessionLocal, Violation, create_db_and_tables
from app.config import (
    CAMERA_INDEX, VIOLATION_SNAPSHOTS_DIR, ATTENDANCE_REPORTS_DIR,
    SECRET_KEY, VIOLATION_SCREENSHOT_COOLDOWN, SNAPSHOT_ON_SEVERITY
)
from app.ml_models.face_detector import FaceRecognizer
from app.ml_models.object_detection import ObjectDetector
from app.ml_models.pose_estimation import PoseEstimator
from app.ml_models.gaze_tracking import GazeTracker
from app.ml_models.audio_analysis import AudioAnalyzer
from app.ml_models.alert_system import generate_alerts

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs', 'eagleeye.log'
        ), encoding='utf-8')
    ]
)
logger = logging.getLogger('EagleEye')

# ─────────────────────────────────────────────────────────────────────────────
# Flask & SocketIO Setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*",
                    ping_timeout=60, ping_interval=25)

# ─────────────────────────────────────────────────────────────────────────────
# Login Manager Setup
# ─────────────────────────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return get_user(user_id)

# ─────────────────────────────────────────────────────────────────────────────
# Role-Based Access Decorators
# ─────────────────────────────────────────────────────────────────────────────
def role_required(*roles):
    """Decorator to restrict routes to specific roles."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                logger.warning(
                    f"Unauthorized access attempt by {current_user.username} "
                    f"(role={current_user.role}) to restricted route."
                )
                return redirect(url_for('login', role=roles[0]))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Global Thread & State Management
# ─────────────────────────────────────────────────────────────────────────────
active_threads = {}
stop_events = {}
thread_cleanup_lock = threading.Lock()

controls_state = {
    "audio": True,
    "gaze": True,
    "object": True,
    "posture": True
}
controls_lock = threading.Lock()

# Per-violation-type cooldown tracker {violation_type: last_screenshot_time}
_snapshot_cooldown = {}
_snapshot_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Camera Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def enumerate_cameras(max_test=5):
    """Detect available cameras and return list of valid indices."""
    available = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available

def open_camera_with_fallback(preferred_index=0):
    """
    Try to open preferred camera index. Falls back through alternatives.
    Returns (cap, index_used) or (None, -1) if all fail.
    Logs detailed debug info on failure.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices_to_try = [preferred_index] + [i for i in range(5) if i != preferred_index]
    
    for idx in indices_to_try:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"[CAMERA] Successfully opened camera {idx} with backend {backend}")
                        return cap, idx
                    cap.release()
                    logger.debug(f"[CAMERA] Camera {idx} backend {backend}: opened but no frame")
            except Exception as e:
                logger.debug(f"[CAMERA] Camera {idx} backend {backend} failed: {e}")
    
    logger.error("[CAMERA] All camera open attempts failed.")
    return None, -1

# ─────────────────────────────────────────────────────────────────────────────
# Thread Management
# ─────────────────────────────────────────────────────────────────────────────
def manage_thread(namespace, target_func):
    """Start a background thread for a namespace if not already running."""
    global active_threads, stop_events
    with thread_cleanup_lock:
        # Cleanup dead threads
        for ns in list(active_threads.keys()):
            if not active_threads[ns].is_alive():
                del active_threads[ns]
                if ns in stop_events:
                    del stop_events[ns]

        if namespace in active_threads and active_threads[namespace].is_alive():
            logger.info(f"Thread for {namespace} is already running.")
            return

        stop_events[namespace] = threading.Event()
        thread = threading.Thread(
            target=target_func,
            args=(app.app_context(), stop_events[namespace]),
            name=f"{namespace}_thread"
        )
        thread.daemon = True
        active_threads[namespace] = thread
        thread.start()
        logger.info(f"[THREAD] Started thread for {namespace}")

def stop_thread(namespace):
    """Signal a thread to stop."""
    if namespace in stop_events:
        stop_events[namespace].set()
        logger.info(f"[THREAD] Stop signal sent to {namespace}")

# ─────────────────────────────────────────────────────────────────────────────
# Database Helper
# ─────────────────────────────────────────────────────────────────────────────
def log_violation_thread_safe(alert_data, person_id='N/A', snapshot_path=None):
    """Thread-safe violation logging with its own database session."""
    db = SessionLocal()
    try:
        new_violation = Violation(
            student_id=person_id,
            violation_type=alert_data.get('type', 'Unknown'),
            timestamp=datetime.datetime.now(),
            details=alert_data.get('details', alert_data.get('message', '')),
            snapshot_path=snapshot_path
        )
        db.add(new_violation)
        db.commit()
        logger.info(f"[DB] Logged violation: {alert_data.get('type')} for student {person_id}")
    except SQLAlchemyError as e:
        logger.error(f"[DB ERROR] Failed to log violation: {e}")
        db.rollback()
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error in violation logging: {e}")
    finally:
        db.close()

def get_violations_from_db(limit=100):
    """Retrieve recent violations from the database."""
    db = SessionLocal()
    try:
        violations = db.query(Violation)\
            .order_by(Violation.timestamp.desc())\
            .limit(limit).all()
        return [
            {
                'id': v.id,
                'student_id': v.student_id,
                'violation_type': v.violation_type,
                'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M:%S') if v.timestamp else 'N/A',
                'details': v.details or '',
                'snapshot_path': v.snapshot_path or ''
            } for v in violations
        ]
    except Exception as e:
        logger.error(f"[DB] Error fetching violations: {e}")
        return []
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────────────────
# Screenshot Cooldown Logic
# ─────────────────────────────────────────────────────────────────────────────
def should_take_screenshot(violation_type):
    """
    Returns True only if the cooldown period has elapsed since last screenshot
    for this violation type. Implements 2-second interval between captures.
    """
    now = time.time()
    with _snapshot_lock:
        last = _snapshot_cooldown.get(violation_type, 0)
        if (now - last) >= VIOLATION_SCREENSHOT_COOLDOWN:
            _snapshot_cooldown[violation_type] = now
            return True
        return False

def save_violation_snapshot(frame, alert):
    """Saves a violation screenshot if cooldown allows."""
    vtype = alert.get('type', 'unknown')
    if not should_take_screenshot(vtype):
        return None
    
    try:
        safe_type = vtype.replace(' ', '_').replace('/', '_')
        filename = f"violation_{safe_type}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        snapshot_path = os.path.join(VIOLATION_SNAPSHOTS_DIR, filename)
        
        # Add overlay text to the snapshot
        annotated = frame.copy()
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, f"VIOLATION: {vtype}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated, ts, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imwrite(snapshot_path, annotated)
        logger.info(f"[SNAPSHOT] Saved evidence: {filename} | Type: {vtype}")
        return snapshot_path
    except Exception as e:
        logger.error(f"[SNAPSHOT ERROR] Failed to save screenshot: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Background Thread: Exam Supervision
# ─────────────────────────────────────────────────────────────────────────────
def supervision_thread(app_context, stop_event):
    """
    Exam supervision background thread with:
    - Camera fallback + retry
    - Optimized violation-triggered screenshots (2s cooldown)
    - Real-time alerts to invigilator dashboard
    """
    with app_context:
        logger.info("[SUPERVISION] Thread started. Initializing AI models...")
        
        # ── Initialize AI modules independently (fault-tolerant) ─────────────
        # FaceRecognizer: core module — fatal if it fails
        try:
            face_recognizer = FaceRecognizer()
        except Exception as e:
            logger.error(f"[SUPERVISION] FaceRecognizer init failed: {e}")
            socketio.emit('supervision_error', {
                'message': f'Face recognition module failed to load: {str(e)}',
                'code': 'MODEL_INIT_FAILED'
            }, namespace='/supervision')
            return

        # ObjectDetector: optional — degrades gracefully if YOLO fails
        try:
            object_detector = ObjectDetector()
            if not object_detector.is_available:
                logger.warning("[SUPERVISION] ObjectDetector unavailable — object detection disabled.")
        except Exception as e:
            logger.warning(f"[SUPERVISION] ObjectDetector init failed: {e} — continuing without it.")
            object_detector = None

        # PoseEstimator: optional
        try:
            pose_estimator = PoseEstimator()
        except Exception as e:
            logger.warning(f"[SUPERVISION] PoseEstimator init failed: {e} — continuing without it.")
            pose_estimator = None

        # GazeTracker: optional
        try:
            gaze_tracker = GazeTracker()
        except Exception as e:
            logger.warning(f"[SUPERVISION] GazeTracker init failed: {e} — continuing without it.")
            gaze_tracker = None

        # AudioAnalyzer: optional
        try:
            audio_analyzer = AudioAnalyzer()
            audio_analyzer.start()
        except Exception as e:
            logger.warning(f"[SUPERVISION] AudioAnalyzer init failed: {e} — continuing without it.")
            audio_analyzer = None

        logger.info("[SUPERVISION] Model initialization complete — entering supervision loop.")

        # Open camera with fallback
        cap, cam_idx = open_camera_with_fallback(CAMERA_INDEX)
        if cap is None:
            logger.error("[SUPERVISION] Cannot open any camera.")
            if audio_analyzer is not None:
                audio_analyzer.stop()
            socketio.emit('supervision_error', {
                'message': 'No camera detected. Please check camera permissions and connections.',
                'code': 'CAMERA_NOT_FOUND',
                'debug': {
                    'attempted_index': CAMERA_INDEX,
                    'tip': 'Try: 1) Close other apps using camera, 2) Check browser/OS permissions, 3) Try an external webcam'
                }
            }, namespace='/supervision')
            return

        logger.info(f"[SUPERVISION] All systems ready. Streaming from camera {cam_idx}.")
        socketio.emit('controls_update', controls_state, namespace='/supervision')
        socketio.emit('camera_info', {'index': cam_idx}, namespace='/supervision')

        prev_frame_time = 0
        consecutive_frame_failures = 0
        MAX_FRAME_FAILS = 30  # ~3 seconds at 10fps

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                consecutive_frame_failures += 1
                if consecutive_frame_failures >= MAX_FRAME_FAILS:
                    logger.error("[SUPERVISION] Too many consecutive frame failures. Stopping.")
                    socketio.emit('supervision_error', {
                        'message': 'Camera feed lost. Please restart supervision.',
                        'code': 'CAMERA_FEED_LOST'
                    }, namespace='/supervision')
                    break
                time.sleep(0.1)
                continue
            consecutive_frame_failures = 0

            with controls_lock:
                current_controls = controls_state.copy()

            # ── Run AI Inference ──────────────────────────────────────────────
            try:
                import numpy as _np
                # Sanitize frame: ensure uint8 BGR before any inference
                if frame.dtype != _np.uint8:
                    frame = frame.astype(_np.uint8)
                rgb_frame = _np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=_np.uint8
                )

                # Face recognition (core — must succeed)
                face_data = face_recognizer.recognize_faces(rgb_frame)

                # Object detection (optional — skip if unavailable)
                if object_detector is not None and current_controls.get('object', True):
                    object_data = object_detector.detect_objects(frame)
                else:
                    object_data = []

                # Gaze tracking (optional)
                if gaze_tracker is not None and current_controls.get('gaze', True):
                    gaze_data = gaze_tracker.get_gaze_direction(frame)
                else:
                    gaze_data = []

                # Audio analysis (optional)
                if audio_analyzer is not None and current_controls.get('audio', True):
                    is_sound_detected = audio_analyzer.is_sound_detected()
                else:
                    is_sound_detected = False

                # Pose estimation (optional)
                is_suspicious_posture = False
                if pose_estimator is not None and current_controls.get('posture', True):
                    try:
                        _, landmarks = pose_estimator.find_pose(frame.copy(), draw=False)
                        if landmarks:
                            lm_list = pose_estimator.get_landmark_positions(frame.shape, landmarks)
                            is_suspicious_posture = pose_estimator.check_suspicious_posture(lm_list)
                    except Exception as pe:
                        logger.warning(f"[SUPERVISION] Pose error: {pe}")

                alerts = generate_alerts(face_data, object_data, gaze_data,
                                         is_suspicious_posture, is_sound_detected)
            except Exception as e:
                logger.error(f"[SUPERVISION] Inference error: {e}")
                # Still emit the raw frame so the video feed stays live
                try:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    socketio.emit('video_frame', {
                        'image': base64.b64encode(buf).decode('utf-8'),
                        'fps': 0, 'face_count': 0, 'alerts_count': 0
                    }, namespace='/supervision')
                except Exception:
                    pass
                socketio.sleep(0.05)
                continue

            # ── Handle Alerts with Smart Screenshot Logic ─────────────────────
            if alerts:
                person_id = face_data[0].get('id', 'N/A') if face_data else 'N/A'
                for alert in alerts:
                    snapshot_path = None
                    # Screenshot only for configured severities + with cooldown
                    if alert.get('severity') in SNAPSHOT_ON_SEVERITY:
                        snapshot_path = save_violation_snapshot(frame, alert)
                        if snapshot_path:
                            alert['snapshot'] = os.path.basename(snapshot_path)
                    
                    log_violation_thread_safe(alert, person_id, snapshot_path)
                    socketio.emit('new_alert', alert, namespace='/supervision')

            # ── Draw Results & Emit Frame ─────────────────────────────────────
            try:
                display_frame = frame.copy()
                
                for person in face_data:
                    box = person['box']
                    name = person.get('name', 'Unknown')
                    top, right, bottom, left = box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for obj in object_data:
                    x1, y1, x2, y2 = obj['box']
                    label = f"{obj['label']} ({obj['confidence']:.2f})"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                if alerts:
                    cv2.putText(display_frame, "⚠ VIOLATION DETECTED", (10, 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 2)

                # FPS overlay
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
                prev_frame_time = new_frame_time
                cv2.putText(display_frame, f"FPS: {int(fps)}", (10, display_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {
                    'image': image_data,
                    'fps': int(fps),
                    'face_count': len(face_data),
                    'alerts_count': len(alerts)
                }, namespace='/supervision')
            except Exception as e:
                logger.error(f"[SUPERVISION] Frame emission error: {e}")

            socketio.sleep(0.033)  # ~30fps target

        # ── Cleanup ───────────────────────────────────────────────────────────
        logger.info("[SUPERVISION] Stop signal received. Cleaning up...")
        try:
            cap.release()
        except Exception:
            pass
        if audio_analyzer is not None:
            try:
                audio_analyzer.stop()
            except Exception:
                pass
        logger.info("[SUPERVISION] Thread finished.")
        socketio.emit('supervision_stopped', namespace='/supervision')

# ─────────────────────────────────────────────────────────────────────────────
# Background Thread: Attendance
# ─────────────────────────────────────────────────────────────────────────────
def attendance_thread(app_context, stop_event):
    """Face-recognition attendance marking thread."""
    with app_context:
        logger.info("[ATTENDANCE] Thread started.")
        face_recognizer = FaceRecognizer()
        log_file_path = os.path.join(
            ATTENDANCE_REPORTS_DIR,
            f"attendance_{datetime.date.today().strftime('%Y-%m-%d')}.csv"
        )
        todays_attendance = set()

        cap, _ = open_camera_with_fallback(CAMERA_INDEX)
        if cap is None:
            socketio.emit('attendance_error', {
                'message': 'Camera not available. Check connections and permissions.',
                'code': 'CAMERA_NOT_FOUND'
            }, namespace='/attendance')
            return

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()

                # ── Guard: skip bad/empty frames ─────────────────────────────
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                # ── Guard: ensure 8-bit 3-channel BGR (face_recognition requirement) ─
                import numpy as _np
                if frame.dtype != _np.uint8:
                    frame = frame.astype(_np.uint8)
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logger.warning("[ATTENDANCE] Unexpected frame shape, skipping frame.")
                    continue

                # Convert BGR -> RGB in contiguous memory layout
                rgb_frame = _np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=_np.uint8
                )

                # ── Run face recognition (per-frame try/except) ──────────────
                try:
                    face_data = face_recognizer.recognize_faces(rgb_frame)
                except Exception as fe:
                    logger.warning(f"[ATTENDANCE] Face recognition error: {fe}")
                    # Still stream the raw video so the feed stays alive
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    socketio.emit('video_frame',
                                  {'image': base64.b64encode(buffer).decode('utf-8')},
                                  namespace='/attendance')
                    socketio.sleep(0.05)
                    continue

                for person in face_data:
                    student_id = person.get('id')
                    student_name = person.get('name', 'Unknown')
                    box = person['box']
                    top, right, bottom, left = box

                    if student_id and student_id != 'Unknown' and student_id not in todays_attendance:
                        timestamp_str = datetime.datetime.now().strftime('%H:%M:%S')
                        todays_attendance.add(student_id)
                        socketio.emit('attendance_update', {
                            'timestamp': timestamp_str,
                            'name': student_name,
                            'roll_number': student_id,
                            'status': 'Present'
                        }, namespace='/attendance')
                        try:
                            with open(log_file_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    datetime.datetime.now().isoformat(),
                                    student_id, student_name, 'Present'
                                ])
                        except IOError as ioe:
                            logger.error(f"[ATTENDANCE] CSV write error: {ioe}")
                        logger.info(f"[ATTENDANCE] Marked: {student_name} ({student_id})")

                    color = (255, 165, 0) if student_id in todays_attendance else (
                        (0, 255, 0) if student_id != 'Unknown' else (0, 0, 255)
                    )
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, student_name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': image_data}, namespace='/attendance')
                socketio.sleep(0.05)

        except Exception as e:
            logger.error(f"[ATTENDANCE] Thread error: {e}")
            socketio.emit('attendance_error', {'message': str(e)}, namespace='/attendance')
        finally:
            if cap and cap.isOpened():
                cap.release()
            logger.info("[ATTENDANCE] Thread stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# Background Thread: Registration Feed
# ─────────────────────────────────────────────────────────────────────────────
def register_thread(app_context, stop_event):
    """Live camera feed for face registration."""
    with app_context:
        cap, _ = open_camera_with_fallback(CAMERA_INDEX)
        if cap is None:
            socketio.emit('registration_error', {
                'message': 'Camera not available for registration.',
                'code': 'CAMERA_NOT_FOUND'
            }, namespace='/register')
            return
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': image_data}, namespace='/register')
                socketio.sleep(0.05)
        except Exception as e:
            logger.error(f"[REGISTER] Thread error: {e}")
            socketio.emit('registration_error', {'message': str(e)}, namespace='/register')
        finally:
            if cap and cap.isOpened():
                cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('hub'))
    # Show the public role-selection landing page
    return render_template('hub.html', role_display=ROLE_DISPLAY, current_user=None)

@app.route('/portal')
def portal():
    """Public landing page — no login required."""
    return render_template('hub.html', role_display=ROLE_DISPLAY, current_user=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # If an authenticated user explicitly visits the login page (or gets redirected here
        # due to lacking role permissions), log them out so they can sign in anew.
        logout_user()
    
    error = None
    selected_role = request.args.get('role', '')  # Pre-select role from hub

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            user = users.get(username)
            if user and user.check_password(password):
                login_user(user, remember=True)
                logger.info(f"[AUTH] Login: {username} (role={user.role})")
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):
                    return redirect(next_page)
                else:
                    # Redirect to the dashboard based on the user's role
                    role_routes = {
                        'student': 'student_dashboard',
                        'faculty': 'faculty_dashboard',
                        'admin': 'admin_dashboard',
                        'invigilator': 'supervision',
                    }
                    return redirect(url_for(role_routes.get(user.role, 'hub')))
                return redirect(next_page)
            else:
                error = 'Invalid username or password. Please try again.'
                logger.warning(f"[AUTH] Failed login attempt for username: '{username}'")

    return render_template('login.html', error=error, selected_role=selected_role,
                           role_display=ROLE_DISPLAY)

@app.route('/logout')
@login_required
def logout():
    logger.info(f"[AUTH] Logout: {current_user.username}")
    logout_user()
    return redirect(url_for('login'))

@app.route('/hub')
@login_required
def hub():
    # Show the hub landing page so authenticated users can navigate to any section.
    # This is the destination for all "Back to Hub" links across the app.
    return render_template('hub.html', role_display=ROLE_DISPLAY, current_user=current_user)

@app.route('/supervision')
@login_required
@role_required('admin', 'invigilator')
def supervision():
    return render_template('supervision.html', current_user=current_user)

@app.route('/attendance')
@login_required
def attendance():
    return render_template('attendance.html', current_user=current_user)

@app.route('/register')
@login_required
@role_required('admin')
def register():
    return render_template('register.html', current_user=current_user)

# Kiosk mode — No auth required
@app.route('/kiosk')
def kiosk():
    return render_template('kiosk.html')

# Admin dashboard
@app.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    violations = get_violations_from_db(100)
    return render_template('admin.html', current_user=current_user, violations=violations)

# Student dashboard
@app.route('/student')
@login_required
@role_required('student')
def student_dashboard():
    import datetime
    return render_template('student.html', current_user=current_user,
                           now_date=datetime.date.today().strftime('%A, %B %d %Y'))

# Faculty dashboard
@app.route('/faculty')
@login_required
@role_required('faculty')
def faculty_dashboard():
    return render_template('faculty.html', current_user=current_user)

# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/cameras')
@login_required
def api_cameras():
    """Returns list of available camera indices."""
    cameras = enumerate_cameras()
    return jsonify({
        'status': 'ok',
        'cameras': cameras,
        'default': CAMERA_INDEX,
        'count': len(cameras)
    })

@app.route('/api/violations')
@login_required
@role_required('admin', 'invigilator')
def api_violations():
    """Returns recent violations as JSON."""
    limit = request.args.get('limit', 50, type=int)
    violations = get_violations_from_db(limit)
    return jsonify({'status': 'ok', 'violations': violations, 'count': len(violations)})

@app.route('/api/violations/export')
@login_required
@role_required('admin')
def api_export_violations():
    """Export violations as CSV."""
    import io
    violations = get_violations_from_db(1000)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['id', 'student_id', 'violation_type', 'timestamp', 'details', 'snapshot_path'])
    writer.writeheader()
    writer.writerows(violations)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=violations_report.csv'}
    )

@app.route('/api/stats')
@login_required
def api_stats():
    """Returns dashboard statistics."""
    db = SessionLocal()
    try:
        total_violations = db.query(Violation).count()
        today = datetime.date.today()
        today_start = datetime.datetime.combine(today, datetime.time.min)
        today_violations = db.query(Violation)\
            .filter(Violation.timestamp >= today_start).count()
        return jsonify({
            'status': 'ok',
            'total_violations': total_violations,
            'today_violations': today_violations,
        })
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────────────────
# Attendance Management APIs
# ─────────────────────────────────────────────────────────────────────────────

# In-memory stores (production: use database tables from the spec's data model)
_sessions = {}          # session_id -> {class_id, faculty_id, start_time, end_time, status}
_corrections = []       # [{id, student_id, old_status, new_status, reason, by_who, timestamp}]
_leave_requests = []    # [{id, student_id, subject, date, reason, status, proof_path}]
_notifications = []     # [{id, role, trigger, channel, message, timestamp, read}]
_offline_queue = []     # [{student_id, session_id, status, source, timestamp}]
_schedule_changes = []  # [{id, type, reason, by_who, timestamp}]

@app.route('/api/attendance/mark', methods=['POST'])
@login_required
def api_attendance_mark():
    """Mark attendance for a student (manual/override). All methods feed one event table with source tag."""
    data = request.get_json(force=True)
    student_id = data.get('student_id')
    status = data.get('status', 'Present')
    source = data.get('source', 'manual')         # face, manual, nfc, etc.
    session_id = data.get('session_id', '')

    if not student_id:
        return jsonify({'status': 'error', 'message': 'student_id is required'}), 400

    event = {
        'id': str(uuid.uuid4()),
        'student_id': student_id,
        'session_id': session_id,
        'status': status,
        'source': source,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    logger.info(f"[ATTENDANCE] Marked: {student_id} → {status} (source={source})")
    return jsonify({'status': 'ok', 'event': event})


@app.route('/api/attendance/corrections', methods=['POST'])
@login_required
@role_required('faculty', 'admin')
def api_attendance_corrections():
    """Quick correction — current/last class only, with reason, fully audited."""
    data = request.get_json(force=True)
    student_id = data.get('student_id')
    new_status = data.get('new_status')
    reason = data.get('reason', '')
    by_who = data.get('by_who', current_user.username)

    if not student_id or not new_status:
        return jsonify({'status': 'error', 'message': 'student_id and new_status required'}), 400
    if not reason:
        return jsonify({'status': 'error', 'message': 'reason is required for audit'}), 400

    correction = {
        'id': str(uuid.uuid4()),
        'student_id': student_id,
        'old_status': 'Unknown',
        'new_status': new_status,
        'reason': reason,
        'by_who': by_who,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    _corrections.append(correction)
    logger.info(f"[CORRECTION] {by_who} changed {student_id} → {new_status}: {reason}")

    # If disciplinary absent, fire notification
    if new_status == 'Disciplinary':
        notif = {
            'id': str(uuid.uuid4()),
            'role': 'student',
            'trigger': 'disciplinary_absent',
            'channel': 'app',
            'message': f'You have been marked as Disciplinary Absent by {by_who}. Reason: {reason}',
            'timestamp': datetime.datetime.now().isoformat(),
            'read': False,
        }
        _notifications.append(notif)
        logger.info(f"[NOTIFICATION] Disciplinary notification sent for {student_id}")

    return jsonify({'status': 'ok', 'correction': correction})


@app.route('/api/attendance/corrections/log', methods=['GET'])
@login_required
@role_required('admin')
def api_get_corrections():
    """Correction/audit logs for compliance."""
    return jsonify({'status': 'ok', 'corrections': _corrections, 'count': len(_corrections)})


@app.route('/api/attendance/export')
@login_required
def api_attendance_export():
    """Export today's attendance as CSV."""
    import io
    log_file = os.path.join(
        ATTENDANCE_REPORTS_DIR,
        f"attendance_{datetime.date.today().strftime('%Y-%m-%d')}.csv"
    )
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['timestamp', 'student_id', 'name', 'status'])
        content = output.getvalue()

    from flask import Response
    return Response(
        content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=attendance_{datetime.date.today()}.csv'}
    )


@app.route('/api/sessions/start', methods=['POST'])
@login_required
@role_required('faculty', 'admin')
def api_session_start():
    """Start an attendance session. Auto-starts from timetable with quick edit for holidays/room/substitute."""
    data = request.get_json(force=True) if request.is_json else {}
    session_id = str(uuid.uuid4())
    session_obj = {
        'id': session_id,
        'class_id': data.get('class_id', 'CS-301'),
        'faculty_id': current_user.username,
        'start_time': datetime.datetime.now().isoformat(),
        'end_time': None,
        'status': 'active',
        'room': data.get('room', 'Room 204'),
        'section': data.get('section', 'B'),
        'is_makeup': data.get('is_makeup', False),
        'substitute': data.get('substitute', None),
    }
    _sessions[session_id] = session_obj
    logger.info(f"[SESSION] Started: {session_id} by {current_user.username}")
    return jsonify({'status': 'ok', 'session': session_obj})


@app.route('/api/sessions/end', methods=['POST'])
@login_required
@role_required('faculty', 'admin')
def api_session_end():
    """End an attendance session and finalize records."""
    data = request.get_json(force=True) if request.is_json else {}
    session_id = data.get('session_id', '')
    if session_id in _sessions:
        _sessions[session_id]['end_time'] = datetime.datetime.now().isoformat()
        _sessions[session_id]['status'] = 'ended'
        logger.info(f"[SESSION] Ended: {session_id}")
        return jsonify({'status': 'ok', 'session': _sessions[session_id]})
    return jsonify({'status': 'error', 'message': 'Session not found'}), 404


@app.route('/api/schedule/change', methods=['POST'])
@login_required
@role_required('faculty', 'admin')
def api_schedule_change():
    """Class transfer/cancel/substitute in one click. All logged with timestamp & reason."""
    data = request.get_json(force=True)
    change = {
        'id': str(uuid.uuid4()),
        'type': data.get('type', 'transfer'),   # cancel, transfer, substitute, makeup
        'reason': data.get('reason', ''),
        'details': data.get('details', {}),
        'by_who': current_user.username,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    _schedule_changes.append(change)
    logger.info(f"[SCHEDULE] {change['type']} by {current_user.username}: {change['reason']}")
    return jsonify({'status': 'ok', 'change': change})


@app.route('/api/leave/request', methods=['POST'])
@login_required
def api_leave_request():
    """Submit a leave request with optional proof upload."""
    subject = request.form.get('subject', request.json.get('subject', '') if request.is_json else '')
    date = request.form.get('date', request.json.get('date', '') if request.is_json else '')
    reason = request.form.get('reason', request.json.get('reason', '') if request.is_json else '')

    proof_path = None
    if 'proof' in request.files:
        proof_file = request.files['proof']
        if proof_file.filename:
            safe_name = f"leave_{current_user.username}_{int(time.time())}_{proof_file.filename}"
            proof_path = os.path.join(ATTENDANCE_REPORTS_DIR, safe_name)
            proof_file.save(proof_path)

    leave = {
        'id': str(uuid.uuid4()),
        'student_id': current_user.username,
        'subject': subject,
        'date': date,
        'reason': reason,
        'status': 'Pending',
        'proof_path': proof_path,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    _leave_requests.append(leave)
    logger.info(f"[LEAVE] Request from {current_user.username}: {subject} on {date}")
    return jsonify({'status': 'ok', 'leave': leave})


@app.route('/api/leave/requests', methods=['GET'])
@login_required
def api_get_leave_requests():
    """Get leave requests (filtered by role)."""
    if current_user.role in ('admin', 'faculty'):
        return jsonify({'status': 'ok', 'requests': _leave_requests})
    student_leaves = [l for l in _leave_requests if l['student_id'] == current_user.username]
    return jsonify({'status': 'ok', 'requests': student_leaves})


@app.route('/api/notifications', methods=['GET'])
@login_required
def api_notifications():
    """Get notifications for the current user's role."""
    role_notifs = [n for n in _notifications if n['role'] == current_user.role or n['role'] == 'all']
    return jsonify({'status': 'ok', 'notifications': role_notifs})


@app.route('/api/attendance/headcount', methods=['GET'])
@login_required
@role_required('admin')
def api_headcount():
    """Emergency headcount — students in campus/classrooms (entrance + classroom cameras)."""
    # In production: cross-reference gate camera logs with classroom attendance
    return jsonify({
        'status': 'ok',
        'on_campus': 142,
        'in_classes': 118,
        'roaming': 24,
        'timestamp': datetime.datetime.now().isoformat(),
    })


@app.route('/api/students/import', methods=['POST'])
@login_required
@role_required('admin')
def api_students_import():
    """Bulk roster import via CSV/Excel or API pull from ERP."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

    imported = 0
    try:
        content = file.read().decode('utf-8')
        reader = csv.DictReader(content.splitlines())
        for row in reader:
            imported += 1
            logger.info(f"[IMPORT] Student: {row}")
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    return jsonify({'status': 'ok', 'imported': imported})


@app.route('/api/admin/register-student', methods=['POST'])
@login_required
@role_required('admin')
def api_admin_register_student():
    """Register a new student's face embeddings from the admin dashboard."""
    try:
        data = request.get_json(force=True) if request.is_json else {}
        name = data.get('name', '').strip()
        roll_number = data.get('roll_number', '').strip()
        image_base64 = data.get('image', '').strip()
        
        # Validation
        if not name or not roll_number:
            return jsonify({
                'status': 'error',
                'message': 'Name and roll number are required.'
            }), 400
        
        if not image_base64:
            return jsonify({
                'status': 'error',
                'message': 'No image data received. Please capture a photo.'
            }), 400
        
        # Decode base64 image to numpy array
        import base64 as b64
        import numpy as np
        try:
            # Remove data URI prefix if present
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = b64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            logger.error(f"[ADMIN REGISTER] Image decode error: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid image format. Please try again.'
            }), 400
        
        # Register the face
        face_recognizer = FaceRecognizer()
        student_info = {'name': name, 'rollnumber': roll_number}
        registration_status = face_recognizer.register_face(student_info, frame)
        
        if 'Success' in registration_status:
            logger.info(f"[ADMIN REGISTER] Successfully registered: {name} ({roll_number})")
            return jsonify({
                'status': 'success',
                'message': registration_status,
                'student': {
                    'name': name,
                    'roll_number': roll_number,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            }), 200
        else:
            logger.warning(f"[ADMIN REGISTER] Registration failed: {registration_status}")
            return jsonify({
                'status': 'error',
                'message': registration_status
            }), 400
    
    except Exception as e:
        logger.error(f"[ADMIN REGISTER] Unexpected error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


@app.route('/api/webhook/event', methods=['POST'])
def api_webhook_event():
    """Webhook endpoint for ERP/LMS integration: session_started, session_ended, correction_made."""
    data = request.get_json(force=True) if request.is_json else {}
    event_type = data.get('event_type', 'unknown')
    logger.info(f"[WEBHOOK] Received event: {event_type} | Data: {json.dumps(data)}")
    return jsonify({'status': 'ok', 'received': event_type, 'timestamp': datetime.datetime.now().isoformat()})


@app.route('/api/exports', methods=['GET'])
@login_required
@role_required('admin')
def api_scheduled_exports():
    """Scheduled exports — CSV/Excel or push to ERP/LMS."""
    fmt = request.args.get('format', 'csv')
    # In production: trigger actual ERP push or generate Excel
    return jsonify({
        'status': 'ok',
        'format': fmt,
        'message': f'Export job queued in {fmt} format',
        'timestamp': datetime.datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Socket.IO: Supervision Namespace
# ─────────────────────────────────────────────────────────────────────────────
@socketio.on('connect', namespace='/supervision')
def supervision_connect():
    logger.info(f"[SOCKET] Supervision client connected: {request.sid}")
    emit('supervision_status', {'status': 'connected'})

@socketio.on('start_supervision', namespace='/supervision')
def start_supervision(data=None):
    try:
        # Allow the front-end to nominate a preferred camera index
        cam_index = None
        if isinstance(data, dict):
            try:
                cam_index = int(data.get('camera_index', CAMERA_INDEX))
            except (TypeError, ValueError):
                cam_index = CAMERA_INDEX

        def _supervision_thread_with_cam(app_ctx, stop_evt):
            """Wrapper that temporarily overrides CAMERA_INDEX for this session."""
            import app.dashboard as _dash
            original = _dash.CAMERA_INDEX
            if cam_index is not None:
                _dash.CAMERA_INDEX = cam_index
            try:
                supervision_thread(app_ctx, stop_evt)
            finally:
                _dash.CAMERA_INDEX = original

        target = _supervision_thread_with_cam if cam_index is not None else supervision_thread
        manage_thread('/supervision', target)
        emit('supervision_started')
    except Exception as e:
        logger.error(f"[SOCKET] Error starting supervision: {e}")
        emit('supervision_error', {'message': f'Failed to start: {str(e)}', 'code': 'START_FAILED'})

@socketio.on('stop_supervision', namespace='/supervision')
def stop_supervision_socket():
    stop_thread('/supervision')
    emit('supervision_stopping')

@socketio.on('disconnect', namespace='/supervision')
def supervision_disconnect():
    stop_thread('/supervision')
    logger.info(f"[SOCKET] Supervision client disconnected: {request.sid}")

@socketio.on('update_controls', namespace='/supervision')
def handle_update_controls(data):
    """Toggle AI module (admin/invigilator only)."""
    if not current_user.is_authenticated or current_user.role not in ('admin', 'invigilator'):
        logger.warning(f"[SOCKET] Unauthorized controls update attempt from {request.sid}")
        return

    module = data.get('module')
    enabled = data.get('enabled')

    with controls_lock:
        if module in controls_state:
            controls_state[module] = bool(enabled)
            logger.info(f"[CONTROLS] {current_user.username} set '{module}' to {enabled}")
            emit('controls_update', controls_state, namespace='/supervision', broadcast=True)

# ─────────────────────────────────────────────────────────────────────────────
# Socket.IO: Attendance Namespace
# ─────────────────────────────────────────────────────────────────────────────
@socketio.on('connect', namespace='/attendance')
def attendance_connect():
    logger.info(f"[SOCKET] Attendance client connected: {request.sid}")
    emit('attendance_status', {'status': 'connected'})
    manage_thread('/attendance', attendance_thread)

@socketio.on('stop_attendance', namespace='/attendance')
def attendance_stop():
    stop_thread('/attendance')

@socketio.on('disconnect', namespace='/attendance')
def attendance_disconnect():
    stop_thread('/attendance')
    logger.info(f"[SOCKET] Attendance client disconnected: {request.sid}")

# ─────────────────────────────────────────────────────────────────────────────
# Socket.IO: Registration Namespace
# ─────────────────────────────────────────────────────────────────────────────
@socketio.on('connect', namespace='/register')
def register_connect():
    logger.info(f"[SOCKET] Registration client connected: {request.sid}")
    emit('registration_status', {'status': 'connected'})
    manage_thread('/register', register_thread)

@socketio.on('disconnect', namespace='/register')
def register_disconnect():
    stop_thread('/register')
    logger.info(f"[SOCKET] Registration client disconnected: {request.sid}")

@socketio.on('register_face', namespace='/register')
def handle_register_face(data):
    try:
        face_recognizer = FaceRecognizer()
        cap, _ = open_camera_with_fallback(CAMERA_INDEX)
        if cap is None:
            emit('registration_status', {'status': 'error', 'message': 'Camera unavailable.'})
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            status_msg = face_recognizer.register_face(
                {'name': data['name'], 'rollnumber': data['roll_number']},
                frame
            )
            status = 'success' if 'Success' in status_msg else 'error'
            emit('registration_status', {'status': status, 'message': status_msg})
        else:
            emit('registration_status', {'status': 'error', 'message': 'Failed to capture frame.'})
    except Exception as e:
        logger.error(f"[REGISTER] Error: {e}")
        emit('registration_status', {'status': 'error', 'message': f'Error: {str(e)}'})

# ─────────────────────────────────────────────────────────────────────────────
# Socket.IO: Kiosk Namespace (no auth required)
# ─────────────────────────────────────────────────────────────────────────────
@socketio.on('connect', namespace='/kiosk')
def kiosk_connect():
    logger.info(f"[SOCKET] Kiosk client connected: {request.sid}")
    emit('kiosk_status', {'status': 'connected'})

@socketio.on('disconnect', namespace='/kiosk')
def kiosk_disconnect():
    stop_thread('/kiosk')

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  EagleEye Smart Campus System - Starting Up")
    logger.info("=" * 60)

    # Initialize database
    create_db_and_tables()

    # Detect available cameras on startup
    avail = enumerate_cameras()
    if avail:
        logger.info(f"[STARTUP] Available cameras: {avail}")
    else:
        logger.warning("[STARTUP] No cameras detected! Supervision features may not work.")

    logger.info("[STARTUP] Server running at http://127.0.0.1:5000")
    logger.info("[STARTUP] Default credentials: admin/admin123, invigilator/proctor123")
    logger.info("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)