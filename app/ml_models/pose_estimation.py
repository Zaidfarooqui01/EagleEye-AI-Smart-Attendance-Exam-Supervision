# /app/ml_models/pose_estimation.py

import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    """
    A class to handle pose estimation using MediaPipe.
    """
    def __init__(self, static_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the pose estimator with MediaPipe.

        Args:
            static_mode (bool): If True, treats the input images as a batch of static, possibly unrelated images.
                                If False, treats them as a video stream.
            model_complexity (int): Complexity of the pose landmark model: 0, 1, or 2.
                                    We use 1 as a balance between accuracy and speed for our CPU.
            min_detection_confidence (float): Minimum confidence value for the detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value for the landmark tracking to be considered successful.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("[INFO] PoseEstimator initialized with MediaPipe.")

    def find_pose(self, frame, draw=True):
        """
        Finds the pose landmarks in a given frame.

        Args:
            frame: The image frame (as a numpy array) to process.
            draw (bool): If True, draws the landmarks and connections on the frame.

        Returns:
            A tuple containing:
            - The frame with landmarks drawn on it (if draw=True).
            - The list of detected landmarks.
        """
        # MediaPipe works with RGB images, so we convert from BGR (OpenCV's default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find the pose
        self.results = self.pose.process(frame_rgb)
        
        # Draw the pose annotation on the image
        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            
        return frame, self.results.pose_landmarks

    def get_landmark_positions(self, frame_shape, landmarks):
        """
        Converts normalized landmark coordinates to pixel coordinates.

        Args:
            frame_shape: The shape of the frame (height, width).
            landmarks: The pose landmarks detected by MediaPipe.

        Returns:
            A list of tuples, where each tuple is the (x, y) pixel coordinate of a landmark.
        """
        lm_list = []
        if landmarks:
            h, w, _ = frame_shape
            for id, lm in enumerate(landmarks.landmark):
                # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
        return lm_list
    
    def check_suspicious_posture(self, landmarks_list):
        """
        Analyzes landmark positions to detect suspicious postures.
        This is a simple example: checks if the head is tilted significantly.

        Args:
            landmarks_list: A list of (x, y) coordinates for all landmarks.

        Returns:
            A boolean indicating if a suspicious posture was detected.
        """
        if len(landmarks_list) == 0:
            return False
            
        # Example check: Head tilt. We compare the y-coordinates of the ears.
        # Landmark indices for left and right ears are 7 and 8.
        left_ear_y = landmarks_list[self.mp_pose.PoseLandmark.LEFT_EAR.value][1]
        right_ear_y = landmarks_list[self.mp_pose.PoseLandmark.RIGHT_EAR.value][1]
        
        # A simple threshold for vertical ear difference. This can be tuned.
        ear_y_difference_threshold = 25 
        
        if abs(left_ear_y - right_ear_y) > ear_y_difference_threshold:
            # This indicates a significant head tilt, which could be suspicious.
            return True
            
        return False

# --- Example Usage (for testing this module directly) ---
'''
if __name__ == '__main__':
    import time
    
    # Initialize the estimator
    estimator = PoseEstimator()
    
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)
    
    pTime = 0
    print("[INFO] Running pose estimation test. Press 'q' to quit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Find and draw the pose
        frame, landmarks = estimator.find_pose(frame, draw=True)
        
        # Get landmark positions
        lm_list = estimator.get_landmark_positions(frame.shape, landmarks)
        
        # Check for suspicious posture
        is_suspicious = estimator.check_suspicious_posture(lm_list)
        
        if is_suspicious:
            cv2.putText(frame, "Suspicious Posture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        cv2.imshow("Pose Estimation Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    '''
    
#------------ends here----------------
