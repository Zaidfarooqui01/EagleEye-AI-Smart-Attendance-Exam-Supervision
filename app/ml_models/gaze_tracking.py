# /app/ml_models/gaze_tracking.py

import cv2
import numpy as np

# We can reuse the face_recognition library to get landmarks, as it's already a dependency.
import face_recognition

class GazeTracker:
    """
    A class to estimate gaze direction based on facial landmarks.
    This provides a simple, lightweight way to detect if a person is looking away.
    """
    def __init__(self):
        """Initializes the GazeTracker."""
        print("[INFO] GazeTracker initialized.")

    def get_gaze_direction(self, frame):
        """
        Estimates the gaze direction for all faces in a frame.

        Args:
            frame: The image frame (as a numpy array) to process.

        Returns:
            A list of dictionaries, one for each detected face, containing the gaze direction.
            Example: [{'box': (x1, y1, x2, y2), 'direction': 'Center'}]
        """
        # First, find all faces and their landmarks in the frame.
        # This is a bit redundant if main.py already does it, but it keeps the module independent.
        # We can optimize this later by passing landmarks directly.
        face_locations = face_recognition.face_locations(frame)
        face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)

        gaze_results = []

        for i, face_landmarks in enumerate(face_landmarks_list):
            # Get the coordinates for the left and right eyes, and the nose tip.
            # These landmarks are used to estimate the head's horizontal orientation.
            left_eye = np.array(face_landmarks['left_eye'])
            right_eye = np.array(face_landmarks['right_eye'])
            nose_tip = np.array(face_landmarks['nose_bridge'][-1]) # The bottom of the nose bridge

            # Calculate the center point of each eye
            left_eye_center = left_eye.mean(axis=0).astype(int)
            right_eye_center = right_eye.mean(axis=0).astype(int)

            # Calculate the horizontal distance from the nose tip to the center of each eye.
            dist_nose_to_left_eye = np.linalg.norm(nose_tip - left_eye_center)
            dist_nose_to_right_eye = np.linalg.norm(nose_tip - right_eye_center)

            # Calculate the ratio of the distances.
            # A ratio of ~1.0 means looking forward.
            # A ratio > 1.0 means turning right (left eye appears closer to nose).
            # A ratio < 1.0 means turning left (right eye appears closer to nose).
            if dist_nose_to_right_eye > 0:
                ratio = dist_nose_to_left_eye / dist_nose_to_right_eye
            else:
                ratio = 1.0 # Avoid division by zero

            direction = "Center"
            # These thresholds can be tuned for sensitivity.
            if ratio > 1.35:
                direction = "Looking Right"
            elif ratio < 0.75:
                direction = "Looking Left"

            # Get the bounding box for this face
            top, right, bottom, left = face_locations[i]

            gaze_results.append({
                'box': (top, right, bottom, left),
                'direction': direction
            })

        return gaze_results

# --- Example Usage (for testing this module directly) ---
'''
if __name__ == '__main__':
    # Initialize the tracker
    gaze_tracker = GazeTracker()

    # Open a connection to the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        exit()
        
    print("[INFO] Running gaze tracking test. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get gaze results
        # Process a smaller frame for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gaze_data = gaze_tracker.get_gaze_direction(small_frame)

        # Draw results on the original frame
        for result in gaze_data:
            top, right, bottom, left = result['box']
            # Scale coordinates back up
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            
            direction = result['direction']
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, direction, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gaze Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

'''
    #-----end here ---------
