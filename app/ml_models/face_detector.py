# /app/ml_models/face_detector.py

import json
import os
import base64
import numpy as np
import face_recognition
from scipy.spatial import distance as dist

# Import settings from our centralized config file
from ..config import ENCODINGS_DIR, FACE_DETECTION_MODEL, FACE_TOLERANCE, EYE_AR_THRESH

# Define the path for the embeddings JSON file using the config
EMBED_FILE = os.path.join(ENCODINGS_DIR, "embeddings.json")

# --- Utility Functions ---

def encode_embedding(emb):
    """Encodes a face embedding into a base64 string for JSON storage."""
    arr = np.array(emb, dtype=np.float32).tobytes()
    return base64.b64encode(arr).decode("utf-8")

def decode_embedding(s):
    """Decodes a base64 string back into a numpy array for comparison."""
    arr = np.frombuffer(base64.b64decode(s), dtype=np.float32)
    return arr

def eye_aspect_ratio(eye):
    """Computes the eye aspect ratio (EAR) to determine if an eye is closed."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Main Recognizer Class ---

class FaceRecognizer:
    """
    A class to handle all face recognition, liveness detection, and database management.
    """
    def __init__(self):
        """Initializes the recognizer by loading the known faces from the JSON file."""
        self.db = self._load_db()
        print(f"[INFO] FaceRecognizer initialized. {len(self.db)} known faces loaded.")

    def _load_db(self):
        """Loads the student database from the embeddings JSON file."""
        if not os.path.exists(EMBED_FILE):
            return {}
        with open(EMBED_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_db(self):
        """Saves the current student database to the embeddings JSON file."""
        with open(EMBED_FILE, "w") as f:
            json.dump(self.db, f, indent=2)

    def register_face(self, student_info, face_img):
        """
        Registers a new student's face. This is called by the registration utility.
        """
        face_locations = face_recognition.face_locations(face_img, number_of_times_to_upsample=2, model=FACE_DETECTION_MODEL)
        
        if not face_locations:
            return "Registration failed: No face could be detected in the image."
        if len(face_locations) > 1:
            return "Registration failed: Multiple faces detected. Please show only one face."

        face_encoding = face_recognition.face_encodings(face_img, known_face_locations=face_locations)[0]

        for entry in self.db.values():
            known_encoding = decode_embedding(entry['embedding'])
            if face_recognition.compare_faces([known_encoding], face_encoding, tolerance=FACE_TOLERANCE)[0]:
                return f"Registration failed: Face is too similar to {entry['name']} ({entry['rollnumber']})."

        sid = student_info["rollnumber"]
        self.db[sid] = {
            "name": student_info["name"],
            "rollnumber": sid,
            "embedding": encode_embedding(face_encoding),
        }
        self._save_db()
        return f"Success! {student_info['name']} has been registered."

    def recognize_faces(self, frame):
        """
        Recognizes all known faces in a given frame and performs liveness detection.
        """
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model=FACE_DETECTION_MODEL)
        
        if not face_locations or not self.db:
            return []

        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
        
        known_encodings = [decode_embedding(entry['embedding']) for entry in self.db.values()]
        known_student_ids = list(self.db.keys())

        recognized_students = []
        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=FACE_TOLERANCE)
            
            student_id = "Unknown"
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                student_id = known_student_ids[first_match_index]
                name = self.db[student_id]['name']

            recognized_students.append({
                'id': student_id,
                'name': name,
                'box': face_locations[i]
            })

        return recognized_students
    
    def list_students(self):
        """Returns a list of all registered students."""
        return list(self.db.values())

    def delete_student(self, rollnumber):
        """Deletes a student from the database by their roll number."""
        if rollnumber in self.db:
            self.db.pop(rollnumber)
            self._save_db()
            return True
        return False


# ----------just for testing independently----------------
'''

if __name__ == "__main__":
    import cv2
    
    print("Testing FaceRecognizer module...")
    recognizer = FaceRecognizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Recognize faces
        faces = recognizer.recognize_faces(rgb_frame)
        
        # Draw results on frame
        for face in faces:
            top, right, bottom, left = face['box']
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{face['name']} ({face['id']})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition Test', frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
'''
    #------end here-----------