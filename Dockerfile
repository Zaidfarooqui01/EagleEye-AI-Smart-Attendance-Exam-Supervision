# Multi-stage Dockerfile for Eagle Eye AI Supervision System
# Stage 1: Builder - Install dependencies and prepare environment
FROM python:3.10-slim as builder

# Install system dependencies required for OpenCV, PyAudio, and compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-dev \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    alsa-lib \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.10-slim

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-highgui4.5 \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    alsa-lib \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs \
    && mkdir -p /app/outputs/attendance_reports \
    && mkdir -p /app/outputs/supervision_reports/violation_snapshots \
    && mkdir -p /app/data/face_encodings \
    && mkdir -p /app/data/student_images

# Expose the Flask application port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000', timeout=5)" || exit 1

# Run the Flask application with SocketIO
# Use gunicorn with eventlet worker for production SocketIO support
# Or run with direct python for development
CMD ["python", "-m", "app.dashboard"]
