# /run_attendance.py

import cv2
import time
import datetime
import os
import sys
import csv

# Add the 'app' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ml_models.face_detector import FaceRecognizer
from app.config import CAMERA_INDEX, ATTENDANCE_REPORTS_DIR

class AttendanceSystem:
    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.log_file_path = os.path.join(ATTENDANCE_REPORTS_DIR, f"attendance_{datetime.date.today()}.csv")
        self.todays_attendance = self._load_todays_attendance()
        print(f"[INFO] Attendance System initialized. Logging to {self.log_file_path}")

    def _load_todays_attendance(self):
        """Loads today's attendance log to avoid duplicate entries."""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'RollNumber', 'Name'])
            return set()
        
        with open(self.log_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            return {row[1] for row in reader}

    def mark_attendance(self, student_id, student_name):
        """Marks attendance for a student if not already marked today."""
        if student_id not in self.todays_attendance:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, student_id, student_name])
            self.todays_attendance.add(student_id)
            print(f"[ATTENDANCE] Marked: {student_name} ({student_id}) at {timestamp}")

    def run(self):
        print("[INFO] Starting Attendance System. Press 'q' to quit.")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        time.sleep(2.0) # Give the camera 2 seconds to initialize
        if not cap.isOpened():
            print(f"[ERROR] Camera at index {CAMERA_INDEX} is not available. Retrying...")
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[FATAL] Cannot open camera. Please check camera drivers and ensure it is not in use by another application.")
                return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Add a check here for empty frames which cause the blank screen
                print("[WARNING] Blank frame received. Skipping...")
                time.sleep(0.1)
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_data = self.recognizer.recognize_faces(rgb_frame)
            
            for person in face_data:
                student_id = person.get('id')
                if student_id != 'Unknown':
                    self.mark_attendance(student_id, person.get('name'))

                # Drawing logic
                box = person['box']
                name = person['name']
                top, right, bottom, left = box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({student_id})" if name != "Unknown" else "Unknown"
                if student_id in self.todays_attendance:
                    label += " (Marked)"
                    color = (255, 165, 0) # Orange for already marked
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Daily Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Attendance System stopped.")
        

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()

