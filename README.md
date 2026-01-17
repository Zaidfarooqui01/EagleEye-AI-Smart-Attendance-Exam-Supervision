# 🦅 Eagle Eye: AI-Powered Exam Supervision System

Eagle Eye is a comprehensive, modular proctoring solution designed to ensure academic integrity during examinations. It leverages a suite of modern computer vision and machine learning models to detect a wide range of potential malpractice activities in real-time. The system is split into two primary applications: a lightweight daily attendance system and a full-featured exam supervision system, both managed through an intuitive web-based dashboard.

---

## 🔑 Key Features

*   **Automated Attendance System:** Uses facial recognition to log student attendance automatically, saving reports in CSV format.
*   **Multi-Modal Malpractice Detection:**
    *   **Face Recognition:** Identifies registered vs. unknown individuals.
    *   **Object Detection (YOLOv8n):** Detects prohibited items like cell phones.
    *   **Pose Estimation (MediaPipe):** Flags suspicious body postures (e.g., significant head tilting).
    *   **Gaze Tracking:** Estimates gaze direction to detect when a student is looking away.
    *   **Audio Analysis:** Monitors microphone volume to detect potential whispering or talking.
*   **Centralized Alert System:** An intelligent core that consolidates flags from all modules to generate high-level alerts with severity ratings.
*   **Professional Web Dashboard:** A real-time control panel built with Flask and Socket.IO to start/stop supervision, view the live video feed, and monitor alerts as they happen.
*   **Secure & Persistent:** Features a user login system to protect the dashboard and saves all violation data to a persistent SQLite database for auditing.
*   **Automated Evidence Capture:** Automatically saves image snapshots for high-severity violations.
*   **Modular Architecture:** Cleanly separated modules for registration, attendance, and supervision, allowing for independent use and easy maintenance.

---

## 🛠️ System Architecture

The project is built on a robust, multi-tier architecture designed for scalability and maintainability.

*   **Perception Layer:** Utilizes OpenCV and PyAudio to capture raw video and audio streams from hardware devices.
*   **Processing Layer (The "Engine"):** A collection of independent AI/ML modules located in `app/ml_models/`. Each module is responsible for a single detection task. The `alert_system` acts as the brain of this layer, making decisions based on inputs from other modules.
*   **Application Layer:** Consists of three primary user-facing entry points:
    1.  `face_register.py`: A command-line utility for the administrative task of enrolling students.
    2.  `run_attendance.py`: A lightweight script for daily attendance marking.
    3.  `dashboard.py`: A Flask and Socket.IO-powered web server that provides a rich, interactive UI for the high-intensity exam supervision.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   A C++ compiler (Visual Studio Build Tools with "Desktop development with C++" workload on Windows)
*   A working webcam and microphone

### Installation Guide

1.  **Clone the Repository:**
    ```
    git clone https://github.com/your-username/EagleEye.git
    cd EagleEye
    ```

2.  **Create and Activate a Virtual Environment:** It is highly recommended to use a virtual environment to manage dependencies.
    ```
    # Create the environment
    python -m venv .venv

    # Activate on Windows
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:** The core `dlib` library can be tricky. The recommended method is to install it via a pre-compiled wheel file.

    *   **Install `dlib`:**
        Download the wheel file corresponding to your Python version (e.g., `cp310` for Python 3.10) from a reliable source. A common source is [this GitHub repository](https://github.com/z-mahmud22/Dlib_Windows_Python3.x). Then, install it directly:
        ```
        pip install "path/to/your/downloaded/dlib-19.24.4-cp310-cp310-win_amd64.whl"
        ```

    *   **Install Remaining Requirements:** Once `dlib` is installed, install the rest of the packages from `requirements.txt`.
        ```
        pip install -r requirements.txt
        ```

### Usage Workflow

The system is designed to be used in a specific order:

**Step 1: Register Students**
Run the registration utility from the command line. Follow the on-screen menu to enroll students by capturing their faces via webcam.

```
python face_register.py
```

**Step 2: Choose an Operational Mode**

You have two options for running the system:

*   **Option A: Daily Attendance Marking**
    To run the simple, standalone attendance logger:
    ```
    python run_attendance.py
    ```
    A window will appear, and attendance for recognized students will be saved to a CSV file in the `outputs/attendance_reports/` directory.

*   **Option B: Live Exam Supervision (Recommended)**
    To launch the full-featured system with the web dashboard:
    ```
    python -m app.dashboard
    ```
    This will start the Flask web server. Open your web browser and navigate to `http://127.0.0.1:5000`. Log in with the default credentials (`invigilator` / `password123`) and click "Start Supervision" to begin monitoring.

---

## 📁 Project Structure

```
EagleEye/
├── app/                  # Core Flask application and ML engine
│   ├── ml_models/        # All individual AI/ML modules
│   ├── static/           # CSS and JavaScript for the dashboard
│   └── templates/        # HTML templates for the dashboard
├── data/                 # Persistent data (database, face encodings)
├── outputs/              # All generated reports and evidence
│   ├── attendance_reports/
│   └── supervision_reports/
├── face_register.py      # Standalone utility to enroll students
├── run_attendance.py     # Entry point for the attendance system
├── run_supervision.py    # Standalone entry point for supervision (no UI)
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
```

***

### **Task 2 & 3: Final Diagrams and Code Review**

This `README.md` already contains a high-level description of the architecture. For the detailed diagrams (ERD, DFD, etc.), these are typically created using external tools. I can provide you with the structured information needed to create them easily using any diagramming software (like Lucidchart, draw.io, or even PowerPoint).

Let's generate the information for each diagram now.

**1. Entity-Relationship Diagram (ERD)**

This describes our database structure.
*   **Entities:** `Student`, `Attendance`, `Violation`.
*   **Student Attributes:** `id` (Primary Key), `student_id` (Unique), `name`, `image_filename`.
*   **Attendance Attributes:** `id` (Primary Key), `student_id` (Foreign Key to Student), `timestamp`, `location`.
*   **Violation Attributes:** `id` (Primary Key), `student_id` (Foreign Key to Student), `violation_type`, `timestamp`, `snapshot_path`, `details`.
*   **Relationships:**
    *   A `Student` can have many `Attendance` records.
    *   A `Student` can have many `Violation` records.

**2. Data Flow Diagram (DFD - Level 0)**

This shows the overall system context.
*   **External Entities:** `Student`, `Invigilator`.
*   **Process:** `0. Eagle Eye System`.
*   **Data Flows:**
    *   `Student` -> provides `Face & Audio Data` -> `System`.
    *   `Invigilator` -> provides `Control Commands` (Start/Stop) -> `System`.
    *   `System` -> provides `Live Feed & Alerts` -> `Invigilator`.
    *   `System` -> provides `Violation & Attendance Reports` -> (Data Store).

**3. Block Diagram**

This shows the high-level components and their interaction.
*   **Input Block:** Webcam, Microphone.
*   **Processing Block (Eagle Eye Core):**
    *   Sub-Block: `Face Recognition`
    *   Sub-Block: `Object Detection`
    *   Sub-Block: `Pose/Gaze Estimation`
    *   Sub-Block: `Audio Analysis`
    *   Sub-Block: `Alert System`
    *   Sub-Block: `Database Logger`
*   **Output Block:**
    *   `Web Dashboard (UI)`
    *   `Alerts Log`
    *   `Evidence Snapshots (Files)`
    *   `Attendance Reports (CSV)`



