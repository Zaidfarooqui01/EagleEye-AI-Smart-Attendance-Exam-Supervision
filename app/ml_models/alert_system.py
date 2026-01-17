# /app/ml_models/alert_system.py

import datetime

def generate_alerts(face_data, object_data, gaze_data, is_suspicious_posture, is_sound_detected):
    """
    Analyzes the data from all detection modules and generates a list of alerts.

    Args:
        face_data (list): Data from FaceRecognizer.
        object_data (list): Data from ObjectDetector.
        gaze_data (list): Data from GazeTracker.
        is_suspicious_posture (bool): Flag from PoseEstimator.
        is_sound_detected (bool): Flag from AudioAnalyzer.

    Returns:
        list: A list of alert dictionaries. Each dictionary contains the
              type of violation and relevant details.
              Returns an empty list if no violations are found.
    """
    alerts = []
    timestamp = datetime.datetime.now().isoformat()

    # Rule 1: No face detected or multiple faces
    # This is a high-priority alert.
    if not face_data:
        alerts.append({
            'timestamp': timestamp,
            'type': 'Identity Alert',
            'message': 'No person detected in the frame.',
            'severity': 'high'
        })
    elif len(face_data) > 1:
        alerts.append({
            'timestamp': timestamp,
            'type': 'Identity Alert',
            'message': f'Multiple people ({len(face_data)}) detected in the frame.',
            'severity': 'high'
        })

    # Rules applied per person if at least one face is detected
    if face_data:
        # For simplicity, we'll check general alerts against the first detected person.
        # A more advanced system could link objects/sounds to the closest person.
        person_id = face_data[0].get('id', 'Unknown')

        # Rule 2: Unknown person detected
        if person_id == 'Unknown':
            alerts.append({
                'timestamp': timestamp,
                'type': 'Identity Alert',
                'message': 'An unknown person has been detected.',
                'severity': 'high'
            })

        # Rule 3: Prohibited object detected
        if object_data:
            for obj in object_data:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'Object Alert',
                    'message': f"Prohibited object detected: {obj['label']}",
                    'details': f"Associated with person: {person_id}",
                    'severity': 'high'
                })

        # Rule 4: Suspicious gaze detected
        # We can check if any detected person is looking away.
        for gaze in gaze_data:
            if gaze['direction'] != 'Center':
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'Behavior Alert',
                    'message': f"Suspicious gaze detected: {gaze['direction']}",
                    'details': f"Associated with person: {person_id}",
                    'severity': 'medium'
                })
                break # One gaze alert is enough

        # Rule 5: Suspicious posture detected
        if is_suspicious_posture:
            alerts.append({
                'timestamp': timestamp,
                'type': 'Behavior Alert',
                'message': 'Suspicious posture (e.g., head tilt) detected.',
                'details': f"Associated with person: {person_id}",
                'severity': 'medium'
            })

    # Rule 6: Continuous sound detected (General alert)
    if is_sound_detected:
        alerts.append({
            'timestamp': timestamp,
            'type': 'Audio Alert',
            'message': 'Potential conversation or whisper detected.',
            'severity': 'low'
        })
        
    return alerts

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    # --- Simulate different scenarios ---
    
    print("--- SCENARIO 1: Everything is normal ---")
    alerts = generate_alerts(
        face_data=[{'id': '123', 'name': 'John Doe'}], 
        object_data=[], 
        gaze_data=[{'direction': 'Center'}], 
        is_suspicious_posture=False, 
        is_sound_detected=False
    )
    print(f"Alerts: {alerts}\n") # Expected: []

    print("--- SCENARIO 2: Phone detected ---")
    alerts = generate_alerts(
        face_data=[{'id': '123', 'name': 'John Doe'}], 
        object_data=[{'label': 'cell phone'}], 
        gaze_data=[{'direction': 'Center'}], 
        is_suspicious_posture=False, 
        is_sound_detected=False
    )
    print(f"Alerts: {alerts}\n") # Expected: [{'type': 'Object Alert', ...}]

    print("--- SCENARIO 3: Multiple violations ---")
    alerts = generate_alerts(
        face_data=[{'id': '123', 'name': 'John Doe'}], 
        object_data=[], 
        gaze_data=[{'direction': 'Looking Left'}], 
        is_suspicious_posture=True, 
        is_sound_detected=True
    )
    print(f"Alerts: {alerts}\n") # Expected: 3 alerts (Gaze, Posture, Audio)
    
    print("--- SCENARIO 4: Unknown person ---")
    alerts = generate_alerts(
        face_data=[{'id': 'Unknown', 'name': 'Unknown'}], 
        object_data=[], 
        gaze_data=[], 
        is_suspicious_posture=False, 
        is_sound_detected=False
    )
    print(f"Alerts: {alerts}\n") # Expected: [{'type': 'Identity Alert', ...}]
