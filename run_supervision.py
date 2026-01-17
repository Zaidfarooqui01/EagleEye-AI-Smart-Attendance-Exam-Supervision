# /run_supervision.py - FINAL VERSION

import cv2
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ml_models.face_detector import FaceRecognizer
from app.ml_models.object_detection import ObjectDetector
from app.ml_models.pose_estimation import PoseEstimator
from app.ml_models.gaze_tracking import GazeTracker
from app.ml_models.audio_analysis import AudioAnalyzer
from app.ml_models.alert_system import generate_alerts
from app.config import CAMERA_INDEX

class SupervisionSystem:
    def __init__(self):
        print("[INFO] Initializing Exam Supervision System...")
        self.face_recognizer = FaceRecognizer()
        self.object_detector = ObjectDetector()
        self.pose_estimator = PoseEstimator()
        self.gaze_tracker = GazeTracker()
        self.audio_analyzer = AudioAnalyzer()
        self.alert_system = generate_alerts
        print("[INFO] All models initialized successfully.")

    def run_standalone(self):
        """
        Main execution method for standalone mode
        """
        print("[INFO] Starting Exam Supervision (Standalone Mode). Press 'q' to quit.")
        
        # Start audio for standalone execution
        self.audio_analyzer.start()
        
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        time.sleep(2.0)

        if not cap.isOpened():
            print(f"[ERROR] Camera at index {CAMERA_INDEX} is not available.")
            self.audio_analyzer.stop()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Blank frame received. Skipping...")
                time.sleep(0.1)
                continue

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_data = self.face_recognizer.recognize_faces(rgb_frame)
            object_data = self.object_detector.detect_objects(frame)
            _, landmarks = self.pose_estimator.find_pose(frame.copy(), draw=False)
            lm_list = self.pose_estimator.get_landmark_positions(frame.shape, landmarks) if landmarks else []
            is_suspicious_posture = self.pose_estimator.check_suspicious_posture(lm_list) if lm_list else False
            gaze_data = self.gaze_tracker.get_gaze_direction(frame)
            is_sound_detected = self.audio_analyzer.is_sound_detected()
            
            alerts = self.alert_system(face_data, object_data, gaze_data, is_suspicious_posture, is_sound_detected)
            
            if alerts:
                print(f"[{time.strftime('%H:%M:%S')}] ALERTS: {alerts}")

            display_frame = self.draw_results(frame, face_data, object_data, alerts)
            cv2.imshow("Exam Supervision System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("[INFO] Shutting down supervision system...")
        self.audio_analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()

    def draw_results(self, frame, face_data, object_data, alerts):
        """Draws results on frame"""
        # ... your existing draw_results implementation ...
        return frame

if __name__ == "__main__":
    system = SupervisionSystem()
    system.run_standalone()  # Changed from system.run()