# /app/user.py
from flask_login import UserMixin

class User(UserMixin):
    """A user class with roles for Flask-Login."""
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role # Admin or Invigilator

# For this project, we'll use a simple hardcoded user dictionary.
# A real-world application would store hashed passwords in the database.
users = {
    "invigilator": User(id="1", username="invigilator", password="password123", role="invigilator"),
    "admin": User(id="2", username="admin", password="adminpassword", role="admin")
}

# Helper function to get a user by their ID
def get_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None
