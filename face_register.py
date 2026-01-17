# /face_register.py

import cv2
import os
import sys
import time

# Add the 'app' directory to the Python path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Now we can import from our app package
from app.ml_models.face_detector import FaceRecognizer
from app.config import CAMERA_INDEX # We'll use a default, but allow override

# A simple global to manage camera mode, similar to your original code
CAMERA_MODE_INDEX = CAMERA_INDEX

def register_from_webcam(recognizer):
    """
    Handles the interactive process of registering a new student via webcam.
    """
    print(f"\n[INFO] Opening camera with index: {CAMERA_MODE_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    time.sleep(2.0) # Give the camera 2 seconds to initialize
    if not cap.isOpened():
        print(f"[ERROR] Camera at index {CAMERA_INDEX} is not available. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[FATAL] Cannot open camera. Please check camera drivers and ensure it is not in use by another application.")
            return

    print("[INFO] Please provide the new student's details.")
    name = input("Enter Name: ").strip()
    roll_number = input("Enter Roll Number: ").strip()

    if not name or not roll_number:
        print("[ERROR] Name and Roll Number cannot be empty.")
        cap.release()
        return

    if roll_number in recognizer.db:
        print(f"[ERROR] A student with Roll Number '{roll_number}' already exists.")
        cap.release()
        return

    student_info = {"name": name, "rollnumber": roll_number}
    
    print("\n[INFO] Look at the camera. Press [SPACE] to capture your face.")
    print("[INFO] Press [Q] to cancel registration.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Add a check here for empty frames which cause the blank screen
            print("[WARNING] Blank frame received. Skipping...")
            time.sleep(0.1)
            continue
        
        status_msg = ""
        # Create a copy to draw on
        display_frame = frame.copy()
        
        # We can draw a simple box to guide the user
        h, w, _ = display_frame.shape
        box_x1, box_y1 = int(w*0.3), int(h*0.2)
        box_x2, box_y2 = int(w*0.7), int(h*0.8)
        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
        cv2.putText(display_frame, "Position your face in the box", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Register Student", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): # Spacebar to capture
            # Use the original, clean frame for registration
            registration_status = recognizer.register_face(student_info, frame)
            
            print(f"\n[STATUS] {registration_status}")
            
            # Show the status on the frame for a moment
            cv2.putText(display_frame, registration_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Success" in registration_status else (0,0,255), 2)
            cv2.imshow("Register Student", display_frame)
            cv2.waitKey(2000) # Wait 2 seconds
            break # Exit after attempting registration
            
        elif key == ord('q'): # 'q' to quit
            print("[INFO] Registration cancelled by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    """
    The main menu for the registration utility.
    """
    global CAMERA_MODE_INDEX
    # We initialize the recognizer here to load the DB once
    recognizer = FaceRecognizer()
    
    while True:
        print("\n========== Face Registration Utility ==========")
        print(f"Current Camera Index: {CAMERA_MODE_INDEX}")
        print("---------------------------------------------")
        print("1. Register New Student")
        print("2. List Registered Students")
        print("3. Delete a Student")
        print("4. Change Camera Index (e.g., for Phone/External Webcam)")
        print("5. Quit")
        print("---------------------------------------------")
        
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            register_from_webcam(recognizer)
        elif choice == '2':
            students = recognizer.list_students()
            if not students:
                print("\n[INFO] No students are registered yet.")
            else:
                print("\n--- Registered Students ---")
                for student in students:
                    print(f"  - Name: {student['name']}, Roll No: {student['rollnumber']}")
        elif choice == '3':
            roll = input("Enter the Roll Number of the student to delete: ").strip()
            if recognizer.delete_student(roll):
                print(f"Successfully deleted student with Roll No: {roll}")
            else:
                print(f"Could not find a student with Roll No: {roll}")
        elif choice == '4':
            try:
                new_index = int(input("Enter new camera index (0, 1, etc.): "))
                
                CAMERA_MODE_INDEX = new_index
                print(f"[INFO] Camera index set to {CAMERA_MODE_INDEX}.")
            except ValueError:
                print("[ERROR] Invalid index. Please enter a number.")
        elif choice == '5':
            print("[INFO] Exiting Registration Utility.")
            break
        else:
            print("[WARNING] Invalid choice. Please try again.")


if __name__ == "__main__":
    main_menu()

