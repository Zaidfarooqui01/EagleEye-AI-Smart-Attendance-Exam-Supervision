# EagleEye Smart Campus - Implementation Guide
## Register Student & Real-Time Exam Violations Monitoring

### 🎯 What Was Implemented

#### 1. **Register New Student Functionality**
Admin dashboard now includes a complete face embedding registration system:

**Features:**
- Student name and roll number input fields
- Live camera capture using browser camera API
- Image preview before registration
- Real-time validation and error messages
- Color-coded status feedback (success/error/info)
- Seamless integration with existing face recognition backend

**How to Use:**
1. Login as Admin (username: `admin`, password: `admin123`)
2. Navigate to Admin Dashboard (`/admin`)
3. Scroll to "Register New Student" section
4. Enter student name (e.g., "Ali Hassan")
5. Enter roll number (e.g., "CS-042")
6. Click "Capture Photo" → browser camera opens
7. Position face in camera view and click "Capture"
8. Review image preview
9. Click "Register Student" to save face embeddings
10. Success message confirms registration with base64-encoded face embedding stored in `data/face_encodings/embeddings.json`

**Backend Integration:**
- Uses existing `FaceRecognizer.register_face()` method
- Stores embeddings as base64 in JSON format (same as login backend)
- Validates:
  - Face detection (must be exactly 1 face)
  - No duplicate faces in database
  - Proper image formatting


#### 2. **Real-Time Exam Violations Monitoring**
Admin dashboard now displays live violations from exam supervision:

**Features:**
- Real-time WebSocket connection to supervision system
- Live status indicator (● Live/Disconnected)
- Violation list shows last 20 incidents
- Each violation displays:
  - Type (Identity Alert, Object Alert, Behavior Alert, Audio Alert)
  - Detailed message
  - Additional details
  - Exact timestamp
  - Severity badge (High/Medium/Low)
- Color-coded severity indicators:
  - 🔴 **High** (red): Identity/Object violations
  - 🟡 **Medium** (yellow): Gaze/Posture violations
  - 🔵 **Low** (cyan): Audio alerts
- KPI card auto-updates with today's violation count
- Auto-scroll shows most recent violations at top
- Hover effects for better interactivity

**How It Works:**
1. Admin is on Dashboard (`/admin`)
2. Invigilator on Supervision page (`/supervision`) clicks "Start Supervision"
3. Supervision thread starts ML models and monitoring
4. When violations detected, WebSocket emits alert to `/supervision` namespace
5. Admin dashboard receives alert in real-time
6. Violation appears in "Real-Time Exam Violations" panel
7. KPI card updates automatically
8. Dashboard shows "● Live" status when connected

---

### 🔧 Technical Architecture

#### API Endpoints

**Student Registration:**
```
POST /api/admin/register-student
Content-Type: application/json

Request Body:
{
  "name": "Ali Hassan",
  "roll_number": "CS-042",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}

Response (Success):
{
  "status": "success",
  "message": "Success! Ali Hassan has been registered.",
  "student": {
    "name": "Ali Hassan",
    "roll_number": "CS-042",
    "timestamp": "2026-03-31T14:30:45.123456"
  }
}

Response (Error):
{
  "status": "error",
  "message": "Face is too similar to existing student"
}
```

#### WebSocket Events

**Namespace:** `/supervision`

**Events Received by Admin Dashboard:**
```javascript
// New violation detected
socket.on('new_alert', function(alert) {
  {
    "timestamp": "2026-03-31T14:32:15.234567",
    "type": "Object Alert",
    "message": "Prohibited object detected: Mobile Phone",
    "details": "Associated with person: CS-042",
    "severity": "high",
    "snapshot": "violation_object_alert_1711900335_a1b2c3.jpg"
  }
});

// Status updates
socket.on('connect') {}    // Connected to supervision
socket.on('disconnect') {} // Disconnected from supervision
```

---

### 📊 Data Flow Diagrams

#### Registration Flow:
```
Admin Dashboard
    ↓
1. Capture Photo (Browser Camera API)
    ↓
2. Send Base64 Image
    ↓ POST /api/admin/register-student
    ↓
3. Dashboard Backend
    ├─ Decode Base64 to Image
    ├─ Detect Faces
    ├─ Generate Embeddings
    └─ Save to embeddings.json
    ↓
4. Success/Error Message
    ↓
Admin Sees Confirmation
```

#### Real-Time Violations Flow:
```
Exam Supervision
    ↓
ML Models Process Frame
    ├─ Face Recognition
    ├─ Object Detection
    ├─ Pose Estimation
    ├─ Gaze Tracking
    └─ Audio Analysis
    ↓
4. Generate Alerts
    ↓ WebSocket emit('new_alert')
    ↓
5. Admin Dashboard
    ├─ Receives Alert (Socket.IO)
    ├─ Adds to Violations List
    ├─ Updates KPI Card
    └─ Updates Status Indicator
    ↓
Admin Sees Real-Time Update
```

---

### ⚙️ Configuration & Requirements

**Frontend Requirements:**
- Browser with WebRTC support (for camera API)
- Socket.IO 4.5.4+ (already included via CDN)
- JavaScript ES6+ support

**Backend Requirements:**
- Python face_recognition library
- Flask + Flask-SocketIO
- OpenCV (cv2)
- NumPy

**Database:**
- SQLite/SQLAlchemy for violations logging
- JSON file for face embeddings (base64 encoded)

---

### 🧪 Testing Guide

#### Test 1: Register a Student
1. Go to `/admin` (login as admin)
2. Scroll to "Register New Student" section
3. Enter name: "Test Student"
4. Enter roll number: "TEST-001"
5. Click "Capture Photo" → camera opens
6. Click "Capture" → image captured
7. Click "Register Student"
8. Expected: ✓ Success message appears, form clears
9. Verify: Check `data/face_encodings/embeddings.json` has new entry

#### Test 2: Duplicate Detection
1. Try to register same student again
2. Expected: ✓ Error message: "Face is too similar to Test Student"

#### Test 3: Real-Time Violations (Multi-User Test)
1. **User 1** (Invigilator): Go to `/supervision`
   - Click "Start Supervision"
   - Manually move away from camera / take object in view
2. **User 2** (Admin): Stay on `/admin`
   - Observe "● Live" status appears
   - Violations appear in real-time
3. **User 1**: Click "Stop Supervision"
4. **User 2**: Status changes to "● Disconnected"
5. Expected: ✓ Real-time updates work smoothly

#### Test 4: Violation History
1. Admin refreshes page → old violations still visible in table below
2. New violations appear in top panel in real-time
3. KPI card shows total count

---

### 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Camera access denied"** | Browser permission issue. Check browser settings → allow camera access for this site |
| **"No face detected"** | Ensure adequate lighting, face clearly visible, centered in frame |
| **"Multiple faces detected"** | Only one person should be in frame |
| **"Violations not appearing"** | Check if supervision is running → Check `/supervision` page → Verify admin is on `/admin` |
| **Real-time updates lag** | Check internet connection → Reload page → Check browser console for errors |
| **Registration fails silently** | Check browser console (F12) → Check network tab for failed requests → Check server logs |

---

### 📝 Code Changes Summary

**Modified Files:**
1. **app/dashboard.py** (Lines 1093-1159)
   - Added new POST endpoint `/api/admin/register-student`
   - Image decoding and validation
   - Integration with FaceRecognizer

2. **app/templates/admin.html**
   - Registration form section (lines 165-210)
   - Real-time violations panel (lines 212-222)
   - JavaScript for camera capture (lines 339-495)
   - WebSocket event handlers (lines 498-570)
   - Input styling in CSS (lines 93-97)

**No Breaking Changes:**
- Existing routes unchanged
- Existing database schema compatible
- Backward compatible with existing registrations

---

### 🚀 Future Enhancements

Possible improvements:
1. **Batch Registration**: CSV/Excel upload for bulk student registration
2. **Camera Settings**: Customize camera index, resolution, FPS
3. **Alert Filtering**: Filter violations by type/severity
4. **Export Reports**: Download real-time violation logs
5. **Email Notifications**: Send alerts to admins via email
6. **Biometric Verification**: Add liveness detection for registration
7. **Mobile App**: Native mobile client for admins

---

### 📞 Support

For issues or questions:
1. Check server logs: `logs/eagleeye.log`
2. Check browser console: F12 → Console tab
3. Check network requests: F12 → Network tab
4. Review implementation: See IMPLEMENTATION_GUIDE.md (this file)

---

**Implementation Date:** March 31, 2026  
**Status:** ✅ Complete and Ready for Production Testing
