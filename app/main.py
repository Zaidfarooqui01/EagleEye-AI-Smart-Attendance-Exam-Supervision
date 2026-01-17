# /app/main.py

import cv2
import time

# Import our custom ML modules
from .ml_models.face_detector import FaceRecognizer
from .ml_models.object_detection import ObjectDetector
from .ml_models.pose_estimation import PoseEstimator
from .ml_models.gaze_tracking import GazeTracker
from .ml_models.audio_analysis import AudioAnalyzer
from .ml_models.alert_system import generate_alerts

# Import settings from our config file
from .config import CAMERA_INDEX

# --- Main Application Class ---

class EagleEyeApp:
    """
    The main application class that orchestrates all the different
    computer vision and audio modules for the Eagle Eye system.
    """
    def __init__(self):
        """Initializes all the necessary ML models and starts audio analysis."""
        print("[INFO] Initializing Eagle Eye System...")
        self.face_recognizer = FaceRecognizer()
        self.object_detector = ObjectDetector()
        self.pose_estimator = PoseEstimator()
        self.gaze_tracker = GazeTracker()
        self.audio_analyzer = AudioAnalyzer()
        
        # Start the audio analyzer in a separate thread
        self.audio_analyzer.start()
        
        print("[INFO] All models initialized successfully.")

    def run(self):
        """
        Starts the main application loop for video and audio processing.
        """
        print("[INFO] Opening camera...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera with index {CAMERA_INDEX}. Exiting.")
            self.audio_analyzer.stop() # Ensure audio thread is stopped
            return

        prev_frame_time = 0

        print("[INFO] Starting live monitoring. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame. Exiting.")
                break

            # --- Run all AI/ML Inference ---
            
            # 1. Vision Modules
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_data = self.face_recognizer.recognize_faces(rgb_frame)
            object_data = self.object_detector.detect_objects(frame)
            pose_frame, landmarks = self.pose_estimator.find_pose(frame.copy(), draw=False)
            lm_list = self.pose_estimator.get_landmark_positions(frame.shape, landmarks)
            is_suspicious_posture = self.pose_estimator.check_suspicious_posture(lm_list)
            gaze_data = self.gaze_tracker.get_gaze_direction(frame)
            
            # 2. Audio Module
            is_sound_detected = self.audio_analyzer.is_sound_detected()
            
            # 3. Alert Generation
            alerts = generate_alerts(face_data, object_data, gaze_data, is_suspicious_posture, is_sound_detected)
            
            # For now, just print alerts to the console
            if alerts:
                print(f"[{time.strftime('%H:%M:%S')}] ALERTS DETECTED: {alerts}")

            # --- Drawing and Display ---
            display_frame = self.draw_all_results(frame, face_data, object_data, gaze_data, alerts)

            # Calculate and display FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            cv2.imshow("Eagle Eye - Live Monitoring", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        print("[INFO] Shutting down system.")
        self.audio_analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()

    def draw_all_results(self, frame, face_data, object_data, gaze_data, alerts):
        """
        A centralized function to draw all results and alerts onto the frame.
        """
        # Draw Face Recognition Results
        for person in face_data:
            box = person['box']
            name = person['name']
            top, right, bottom, left = box
            color = (0, 0, 255) # Red for Unknown
            label = "Unknown Face"
            if name != "Unknown":
                color = (0, 255, 0)
                label = f"{name} ({person['id']})"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Object Detection Results
        for obj in object_data:
            box = obj['box']
            label = f"{obj['label']} ({obj['confidence']:.2f})"
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw a general alert status on the top left
        if alerts:
            alert_messages = [f"{alert['type']}: {alert['message']}" for alert in alerts]
            # Display first alert prominently
            cv2.putText(frame, "ALERT!", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, alert_messages[0], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(alert_messages) > 1:
                cv2.putText(frame, f"+ {len(alert_messages)-1} more", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

# This allows us to run the application directly from the command line
if __name__ == '__main__':
    app = EagleEyeApp()
    app.run()
