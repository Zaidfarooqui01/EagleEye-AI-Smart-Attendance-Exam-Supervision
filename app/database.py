# /app/database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import DATABASE_URL
import datetime

# Create a connection to the database. The 'check_same_thread=False'
# is needed because we'll be accessing the DB from different parts of our app (Flask and main loop).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# A session is our "handle" to the database, allowing us to query it.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is the base class our database models will inherit from.
Base = declarative_base()

# --- Database Table Models ---

class Student(Base):
    """Represents a student in the database."""
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)

class Attendance(Base):
    """Represents an attendance record."""
    __tablename__ = "attendance"
    id = Column(
    Integer, 
    primary_key=True, 
    index=True)

    student_id = Column(
    String,
    index=True, 
    nullable=False)

    timestamp = Column(
    DateTime, 
    default=datetime.datetime.utcnow, 
    nullable=False
)
    location = Column(
    String, 
    default="Room A101")

class Violation(Base):
    """Represents a malpractice violation record."""
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    violation_type = Column(String, nullable=False) # e.g., 'Phone Detected', 'Suspicious Gaze'
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    snapshot_path = Column(String) # Path to the evidence image
    details = Column(Text) # Extra details, like transcribed text

def create_db_and_tables():
    """
    Creates the database and all the tables defined above.
    This function should be called once when the application starts.
    """
    print("Creating database and tables...")
    # The 'metadata.create_all' command connects to the DB and creates any tables that don't exist.
    Base.metadata.create_all(bind=engine)
    print("Database and tables created successfully.")

# To get a database session
def get_db():
    """Dependency to get a DB session. Ensures the session is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_violation(session, alert_data, person_id='N/A', snapshot_path=None):
    """
    Logs a detected violation to the database.

    Args:
        session: The SQLAlchemy database session.
        alert_data (dict): The alert dictionary.
        person_id (str): The ID of the student.
        snapshot_path (str, optional): The file path to the evidence snapshot.
    """
    # Use current time instead of trying to parse from alert_data
    timestamp = datetime.datetime.now()
    if 'timestamp' in alert_data and isinstance(alert_data['timestamp'], str):
        try:
            timestamp = datetime.datetime.fromisoformat(alert_data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            pass  # Use current time if parsing fails
    
    new_violation = Violation(
        student_id=person_id,
        violation_type=alert_data.get('type', 'Unknown'),
        timestamp=timestamp,
        details=alert_data.get('details', alert_data.get('message', '')),
        snapshot_path=snapshot_path # Save the path to the DB
    )
    session.add(new_violation)
    session.commit()

# --- Example of how to use it (for testing later) ---
if __name__ == "__main__":
    # If you run this file directly, it will create the database and tables.
    create_db_and_tables()
    print("Database setup complete. You can find 'eagle_eye.db' in the 'data' folder.")