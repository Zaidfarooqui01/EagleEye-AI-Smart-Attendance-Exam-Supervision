# /app/dashboard.py 

import sys
import os
import cv2
import base64
import threading
import time
import uuid
import datetime
import csv
 
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sqlalchemy.exc import SQLAlchemyError

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Core App Modules ---
from app.user import User, users, get_user
from app.database import SessionLocal, Violation
from app.config import CAMERA_INDEX, VIOLATION_SNAPSHOTS_DIR, ATTENDANCE_REPORTS_DIR
from run_supervision import SupervisionSystem
# Import all model classes, not just SupervisionSystem
from app.ml_models.face_detector import FaceRecognizer
from app.ml_models.object_detection import ObjectDetector
from app.ml_models.pose_estimation import PoseEstimator
from app.ml_models.gaze_tracking import GazeTracker
from app.ml_models.audio_analysis import AudioAnalyzer
from app.ml_models.alert_system import generate_alerts

def log_violation_thread_safe(alert_data, person_id='N/A', snapshot_path=None):
    """
    Thread-safe violation logging with its own database session.
    Place this function near the top of dashboard.py, after the imports.
    """
    db = SessionLocal()
    try:
        # Use current time for consistency
        timestamp = datetime.datetime.now()
        
        new_violation = Violation(
            student_id=person_id,
            violation_type=alert_data.get('type', 'Unknown'),
            timestamp=timestamp,
            details=alert_data.get('details', alert_data.get('message', '')),
            snapshot_path=snapshot_path
        )
        db.add(new_violation)
        db.commit()
        print(f"[DB] Logged violation: {alert_data.get('type')} for {person_id}")
    except SQLAlchemyError as e:
        print(f"[DB ERROR] Failed to log violation: {e}")
        db.rollback()
    except Exception as e:
        print(f"[ERROR] Unexpected error in violation logging: {e}")
    finally:
        db.close()
        
# --- Flask & SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_truly_secret_key_for_eagle_eye'
socketio = SocketIO(app, async_mode='threading')

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return get_user(user_id)

# --- Global Thread Management ---
active_threads = {}
stop_events = {}
thread_cleanup_lock = threading.Lock()

### NEW: Global State for Admin Controls ###
controls_state = {
    "audio": True,
    "gaze": True,
    "object": True,
    "posture": True
}
controls_lock = threading.Lock()

# --- Helper Functions ---

def log_violation_thread_safe(alert_data, person_id='N/A', snapshot_path=None):
    """Thread-safe violation logging with its own database session"""
    db = SessionLocal()
    try:
        timestamp = datetime.datetime.now()
        
        new_violation = Violation(
            student_id=person_id,
            violation_type=alert_data.get('type', 'Unknown'),
            timestamp=timestamp,
            details=alert_data.get('details', alert_data.get('message', '')),
            snapshot_path=snapshot_path
        )
        db.add(new_violation)
        db.commit()
        print(f"[DB] Logged violation: {alert_data.get('type')} for {person_id}")
    except SQLAlchemyError as e:
        print(f"[DB ERROR] Failed to log violation: {e}")
        db.rollback()
    except Exception as e:
        print(f"[ERROR] Unexpected error in violation logging: {e}")
    finally:
        db.close()

def manage_thread(namespace, target_func):
    global active_threads, stop_events
    with thread_cleanup_lock:
        for ns in list(active_threads.keys()):
            if not active_threads[ns].is_alive():
                del active_threads[ns]
                if ns in stop_events:
                    del stop_events[ns]
        
        if namespace in active_threads and active_threads[namespace].is_alive():
            print(f"Thread for {namespace} is already running.")
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
        print(f"Started thread for {namespace}")

# --- Background Thread Logic ---

def supervision_thread(app_context, stop_event):
    """
    The main logic for the exam supervision system.
    """
    with app_context:
        print("[DASHBOARD] Supervision thread started. Initializing systems...")
        
        # Initialize all models needed for supervision
        face_recognizer, object_detector, pose_estimator, gaze_tracker, audio_analyzer = FaceRecognizer(), ObjectDetector(), PoseEstimator(), GazeTracker(), AudioAnalyzer()
        audio_analyzer.start()
        
        # Open camera
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[DASHBOARD-ERROR] Cannot open camera in thread."); audio_analyzer.stop()
            socketio.emit('supervision_error', {'message': 'Cannot open camera'}, namespace='/supervision'); return

        print("[DASHBOARD] All systems initialized. Starting main loop.")
        # Emit the initial state of controls to the client
        socketio.emit('controls_update', controls_state, namespace='/supervision')

        prev_frame_time = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with controls_lock:
                current_controls = controls_state.copy()
            
            # --- Conditionally run modules based on controls ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_data = face_recognizer.recognize_faces(rgb_frame)
            object_data = object_detector.detect_objects(frame) if current_controls['object'] else []
            gaze_data = gaze_tracker.get_gaze_direction(frame) if current_controls['gaze'] else []
            is_sound_detected = audio_analyzer.is_sound_detected() if current_controls['audio'] else False
            is_suspicious_posture = False
            if current_controls['posture']:
                _, landmarks = pose_estimator.find_pose(frame.copy(), draw=False)
                if landmarks:
                    lm_list = pose_estimator.get_landmark_positions(frame.shape, landmarks)
                    is_suspicious_posture = pose_estimator.check_suspicious_posture(lm_list)
            
            alerts = generate_alerts(face_data, object_data, gaze_data, is_suspicious_posture, is_sound_detected)
            
            # Handle alerts and database logging
            if alerts:
                person_id = face_data[0]['id'] if face_data else 'N/A'
                for alert in alerts:
                    snapshot_path = None
                    if alert.get('severity') == 'high':
                        filename = f"violation_{alert['type']}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
                        snapshot_path = os.path.join(VIOLATION_SNAPSHOTS_DIR, filename)
                        try:
                            cv2.imwrite(snapshot_path, frame)
                            print(f"[EVIDENCE] Saved snapshot to {snapshot_path}")
                            alert['details'] = alert.get('details', '') + f" | Evidence: {os.path.basename(snapshot_path)}"
                        except Exception as e:
                            print(f"[SNAPSHOT ERROR] Failed to save evidence: {e}")
                    
                    # Log violation to database (thread-safe)
                    log_violation_thread_safe(alert, person_id, None)
                    socketio.emit('new_alert', alert, namespace='/supervision')

            # Draw results and emit frame
            try:
                # Create a local copy to draw on
                display_frame = frame.copy()
                if face_data:
                    # Draw face boxes
                    for person in face_data:
                        box = person['box']
                        name = person.get('name', 'Unknown')
                        top, right, bottom, left = box
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if object_data:
                     for obj in object_data:
                        box = obj['box']
                        label = f"{obj['label']}"
                        x1, y1, x2, y2 = box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if alerts:
                    cv2.putText(display_frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

                _, buffer = cv2.imencode('.jpg', display_frame)
                image_data = base64.b64encode(buffer).decode('utf-8')
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time)>0 else 0
                prev_frame_time = new_frame_time
                socketio.emit('video_frame', {
                    'image': image_data, 'fps': int(fps), 'face_count': len(face_data),
                    'alerts_count': len(alerts)
                }, namespace='/supervision')
            except Exception as e:
                print(f"[FRAME EMISSION ERROR] {e}")
            socketio.sleep(0.03)

        print("[DASHBOARD] Stop signal received. Cleaning up supervision resources...")
        cap.release()
        audio_analyzer.stop()
        print("[DASHBOARD] Supervision thread finished.")
        socketio.emit('supervision_stopped', namespace='/supervision')

def attendance_thread(app_context, stop_event):
    """Background thread for marking attendance."""
    with app_context:
        face_recognizer = FaceRecognizer()
        log_file_path = os.path.join(ATTENDANCE_REPORTS_DIR, f"attendance_{datetime.date.today()}.csv")
        todays_attendance = set()
        
        cap = None
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret: 
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_data = face_recognizer.recognize_faces(rgb_frame)
                
                for person in face_data:
                    student_id = person.get('id')
                    if student_id != 'Unknown' and student_id not in todays_attendance:
                        timestamp_str = datetime.datetime.now().strftime('%H:%M:%S')
                        student_name = person.get('name')
                        todays_attendance.add(student_id)
                        socketio.emit('attendance_update', {
                            'timestamp': timestamp_str, 
                            'name': student_name, 
                            'roll_number': student_id
                        }, namespace='/attendance')
                        with open(log_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.datetime.now().isoformat(), student_id, student_name])
                    
                    box = person['box']
                    top, right, bottom, left = box
                    color = (255,165,0) if student_id in todays_attendance else ((0,255,0) if student_id != 'Unknown' else (0,0,255))
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                _, buffer = cv2.imencode('.jpg', frame)
                image_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': image_data}, namespace='/attendance')
                socketio.sleep(0.05)
        except Exception as e:
            print(f"Attendance thread error: {e}")
            socketio.emit('attendance_error', {'message': str(e)}, namespace='/attendance')
        finally:
            if cap and cap.isOpened():
                cap.release()
            print("Attendance thread stopped.")

def register_thread(app_context, stop_event):
    """Background thread for the registration video feed."""
    with app_context:
        cap = None
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret: 
                    continue
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': image_data}, namespace='/register')
                socketio.sleep(0.05)
        except Exception as e:
            print(f"Registration thread error: {e}")
            socketio.emit('registration_error', {'message': str(e)}, namespace='/register')
        finally:
            if cap and cap.isOpened():
                cap.release()
            print("Registration feed thread stopped.")

# --- Flask Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('hub'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/hub')
@login_required
def hub():
    return render_template('hub.html')

@app.route('/supervision')
@login_required
def supervision():
    return render_template('supervision.html')

@app.route('/attendance')
@login_required
def attendance():
    return render_template('attendance.html')

@app.route('/register')
@login_required
def register():
    if current_user.role != 'admin':
        return redirect(url_for('hub'))
    return render_template('register.html')

@app.route('/')
@login_required
def home():
    return redirect(url_for('hub'))

# --- Socket.IO Namespaces & Handlers ---

# Supervision Namespace
@socketio.on('connect', namespace='/supervision')
def supervision_connect():
    print("Supervision client connected.")
    emit('supervision_status', {'status': 'connected'})

@socketio.on('start_supervision', namespace='/supervision')
def start_supervision():
    try:
        manage_thread('/supervision', supervision_thread)
        emit('supervision_started', namespace='/supervision')
    except Exception as e:
        print(f"Error starting supervision: {e}")
        emit('supervision_error', {'message': 'Failed to start supervision'}, namespace='/supervision')

@socketio.on('stop_supervision', namespace='/supervision')
def stop_supervision():
    if '/supervision' in stop_events:
        stop_events['/supervision'].set()
        emit('supervision_stopping', namespace='/supervision')

@socketio.on('disconnect', namespace='/supervision')
def supervision_disconnect():
    if '/supervision' in stop_events:
        stop_events['/supervision'].set()
    print("Supervision client disconnected.")

@socketio.on('update_controls', namespace='/supervision')
def handle_update_controls(data):
    """Handles requests from an admin to toggle a detection module."""
    if not current_user.is_authenticated or current_user.role != 'admin':
        print(f"Unauthorized control update attempt by {request.sid}")
        return

    module = data.get('module')
    enabled = data.get('enabled')

    with controls_lock:
        if module in controls_state:
            controls_state[module] = bool(enabled)
            print(f"[CONTROLS] Admin '{current_user.username}' updated '{module}' to {enabled}")
            # Broadcast the new state to ALL connected supervision clients
            emit('controls_update', controls_state, namespace='/supervision', broadcast=True)    

# Attendance Namespace
@socketio.on('connect', namespace='/attendance')
def attendance_connect():
    print("Attendance client connected.")
    emit('attendance_status', {'status': 'connected'})
    manage_thread('/attendance', attendance_thread)

@socketio.on('disconnect', namespace='/attendance')
def attendance_disconnect(): 
    if '/attendance' in stop_events:
        stop_events['/attendance'].set()
    print("Attendance client disconnected.")

# Registration Namespace
@socketio.on('connect', namespace='/register')
def register_connect():
    print("Registration client connected.")
    emit('registration_status', {'status': 'connected'})
    manage_thread('/register', register_thread)

@socketio.on('disconnect', namespace='/register')
def register_disconnect(): 
    if '/register' in stop_events:
        stop_events['/register'].set()
    print("Registration client disconnected.")

### NEW: Add this new handler for admin controls ###
@socketio.on('update_controls', namespace='/supervision')
def handle_update_controls(data):
    """Handles requests from an admin to toggle a detection module."""
    if not current_user.is_authenticated or current_user.role != 'admin':
        print(f"Unauthorized control update attempt by {request.sid}")
        return

    module = data.get('module')
    enabled = data.get('enabled')

    with controls_lock:
        if module in controls_state:
            controls_state[module] = bool(enabled)
            print(f"[CONTROLS] Admin '{current_user.username}' updated '{module}' to {enabled}")
            # Broadcast the new state to ALL connected supervision clients
            emit('controls_update', controls_state, namespace='/supervision', broadcast=True)
### END NEW ###

@socketio.on('register_face', namespace='/register')
def handle_register_face(data):
    try:
        face_recognizer = FaceRecognizer()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
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
            emit('registration_status', {'status': 'error', 'message': 'Failed to capture frame from camera.'})
    except Exception as e:
        print(f"Error registering face: {e}")
        emit('registration_status', {'status': 'error', 'message': f'Error: {str(e)}'})

# --- Main Entry Point ---
if __name__ == '__main__':
    print("[INFO] Starting Eagle Eye Command Center Server...")
    
    os.makedirs(VIOLATION_SNAPSHOTS_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_REPORTS_DIR, exist_ok=True)
    print(f"[INFO] Violation snapshots directory: {VIOLATION_SNAPSHOTS_DIR}")
    print(f"[INFO] Attendance reports directory: {ATTENDANCE_REPORTS_DIR}")
    
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)