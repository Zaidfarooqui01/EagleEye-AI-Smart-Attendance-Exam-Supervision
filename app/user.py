# /app/user.py
from flask_login import UserMixin
import hashlib

class User(UserMixin):
    """A user class with roles for Flask-Login."""
    def __init__(self, id, username, password, role, name=None, department=None):
        self.id = id
        self.username = username
        self.password = password  # Stored as plain-text for demo; use hashing in production
        self.role = role  # 'student', 'faculty', 'admin', 'invigilator'
        self.name = name or username.title()
        self.department = department or "Computer Science"

    def check_password(self, password):
        return self.password == password

# Hardcoded demo users for all four roles
# In production: use hashed passwords and a proper database
users = {
    "student1": User(
        id="1",
        username="student1",
        password="student123",
        role="student",
        name="Ali Hassan",
        department="Computer Science"
    ),
    "faculty1": User(
        id="2",
        username="faculty1",
        password="faculty123",
        role="faculty",
        name="Dr. Sarah Khan",
        department="Computer Science"
    ),
    "admin": User(
        id="3",
        username="admin",
        password="admin123",
        role="admin",
        name="Prof. Ahmed (HOD)",
        department="Administration"
    ),
    "invigilator": User(
        id="4",
        username="invigilator",
        password="proctor123",
        role="invigilator",
        name="Mr. Zaid Farooqui",
        department="Examination Cell"
    ),
    # Legacy fallback
    "demo": User(
        id="5",
        username="demo",
        password="demo123",
        role="invigilator",
        name="Demo User",
        department="Demo"
    ),
}

# Helper function to get a user by their ID
def get_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None

# Role permission helpers
ROLE_PERMISSIONS = {
    "student":     ["view_attendance", "view_own_records"],
    "faculty":     ["view_attendance", "mark_attendance", "view_reports", "view_sessions"],
    "admin":       ["view_attendance", "mark_attendance", "view_reports", "view_sessions",
                    "manage_students", "view_violations", "toggle_modules", "export_reports",
                    "view_kpis", "audit_logs"],
    "invigilator": ["view_supervision", "view_violations", "toggle_modules_supervision"],
}

def has_permission(user, permission):
    if not user or not user.is_authenticated:
        return False
    return permission in ROLE_PERMISSIONS.get(user.role, [])

# Role display info for the hub page
ROLE_DISPLAY = {
    "student": {
        "label": "Student",
        "icon": "fa-graduation-cap",
        "color": "#818cf8",
        "bg": "rgba(129,140,248,0.15)",
        "features": ["Attendance Charts", "Leave Requests", "Risk Alerts"],
        "description": "View attendance, ERP analytics & leave requests"
    },
    "faculty": {
        "label": "Faculty",
        "icon": "fa-chalkboard-teacher",
        "color": "#22d3ee",
        "bg": "rgba(34,211,238,0.15)",
        "features": ["Live Sessions", "Quick Corrections", "Export Reports"],
        "description": "Manage sessions, mark attendance & view reports"
    },
    "admin": {
        "label": "Admin / HOD",
        "icon": "fa-shield-halved",
        "color": "#10b981",
        "bg": "rgba(16,185,129,0.15)",
        "features": ["Institute KPIs", "Dept Analytics", "Audit Logs"],
        "description": "Institute KPIs, defaulter lists & audit logs"
    },
    "invigilator": {
        "label": "Invigilator",
        "icon": "fa-eye",
        "color": "#f43f5e",
        "bg": "rgba(244,63,94,0.15)",
        "features": ["Live Monitoring", "Violation Alerts", "Exam Reports"],
        "description": "Monitor exams with AI proctoring & violation alerts"
    },
}
