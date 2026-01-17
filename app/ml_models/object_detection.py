# /app/ml_models/object_detection.py

from ultralytics import YOLO

# --- Constants ---
# We are using the 'nano' version of YOLOv8, which is optimized for speed on CPU.

YOLO_MODEL_NAME = 'yolov8n.pt' 

TARGET_CLASSES = {67: 'cell phone'}

class ObjectDetector:
    """
    A class to handle object detection using the YOLOv8 model.
    """
    def __init__(self, model_name=YOLO_MODEL_NAME):
        """
        Initializes the detector by loading the YOLOv8 model.
        """
        self.model = YOLO(model_name)
        print(f"[INFO] ObjectDetector initialized with model: {model_name}")

    def detect_objects(self, frame, confidence_threshold=0.5):
        """
        Detects target objects (like cell phones) in a given frame.

        Args:
            frame: The image frame (as a numpy array) to perform detection on.
            confidence_threshold (float): The minimum confidence score to consider a detection valid.

        Returns:
            A list of dictionaries, where each dictionary represents a detected object.
            Example: [{'label': 'cell phone', 'confidence': 0.85, 'box': (x1, y1, x2, y2)}]
        """
        # The 'predict' method runs the inference. 'verbose=False' silences the log output for each frame.
        results = self.model.predict(frame, verbose=False)
        
        detected_objects = []
        
        # The result object contains all the detections. We loop through them.
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the class ID (e.g., 67 for 'cell phone')
                class_id = int(box.cls[0])
                
                # Check if the detected class is one of our targets
                if class_id in TARGET_CLASSES:
                    confidence = float(box.conf[0])
                    
                    # Check if the confidence is above our threshold
                    if confidence > confidence_threshold:
                        # Get the bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detected_objects.append({
                            'label': TARGET_CLASSES[class_id],
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2)
                        })
                        
        return detected_objects

# --- Example Usage (for testing this module directly) ---
'''
if __name__ == '__main__':
    import cv2
    
    # Initialize the detector
    detector = ObjectDetector()
    
    # Open a connection to the camera
    cap = cv2.VideoCapture(0) # Use your default camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        exit()
        
    print("[INFO] Running object detection test. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects in the frame
        objects = detector.detect_objects(frame)
        
        # Draw boxes around detected objects
        for obj in objects:
            label = f"{obj['label']}: {obj['confidence']:.2f}"
            box = obj['box']
            x1, y1, x2, y2 = box
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw the label background and text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        cv2.imshow("Object Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

'''
# -------end here----------