# Eagle Eye Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Eagle Eye AI Supervision System using Docker and Docker Compose.

## 📋 Prerequisites

- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 1.29+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **For GPU Support**: NVIDIA Container Toolkit (optional)

## 🚀 Quick Start

### 1. Development Deployment with Docker Compose

The easiest way to get started with full debugging capabilities:

```bash
# Build and start the container
docker-compose up --build

# In another terminal, you can access logs
docker-compose logs -f
```

The application will be available at `http://localhost:5000`

### 2. Production Deployment

For production environments, use the optimized Dockerfile:

```bash
# Build the production image
docker build -f Dockerfile.prod -t eagleeye:latest .

# Run the container
docker run -d \
  --name eagleeye-prod \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  --device /dev/video0:/dev/video0 \
  --device /dev/snd:/dev/snd \
  -e EAGLE_EYE_SECRET="your-production-secret" \
  eagleeye:latest
```

## 🎥 Camera and Audio Configuration

### Linux/Ubuntu

The Docker Compose file includes device mappings for camera and audio:

```yaml
devices:
  - /dev/video0:/dev/video0    # Webcam
  - /dev/snd:/dev/snd          # Audio
```

To use a different camera or audio device:

```bash
# List available video devices
ls -la /dev/video*

# List audio devices
arecord -l

# Update docker-compose.yml with the correct device path
```

### macOS

macOS doesn't support device pass-through to Docker containers directly. For camera/audio on Mac:

```bash
# Option 1: Use Docker Desktop with camera sharing
# Enable in Docker Desktop preferences → Resources → File Sharing

# Option 2: Use host network mode
docker run -d \
  --network host \
  -v $(pwd)/data:/app/data \
  --name eagleeye \
  eagleeye:latest
```

### Windows (Docker Desktop)

Windows Docker Desktop with WSL2 backend:

```bash
# Check available cameras in Windows
powershell -Command "Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -match 'camera|video'}"

# Update the host Windows registry or use UsbipDialog for USB device passthrough
# Or run the application with Windows host network:
docker run -d ^
  --network host ^
  -v %cd%\data:/app/data ^
  --name eagleeye ^
  eagleeye:latest
```

## 🔐 Environment Configuration

### Production Secrets

Create a `.env` file in the project root:

```env
EAGLE_EYE_SECRET=your-super-secret-key-change-this-in-production
FLASK_ENV=production
CAMERA_INDEX=0
```

Load the environment file:

```bash
docker-compose --env-file .env up
```

### Custom Configuration

Modify environment variables in `docker-compose.yml` or pass them at runtime:

```bash
docker run -d \
  -e CAMERA_INDEX=1 \
  -e FLASK_ENV=production \
  -e EAGLE_EYE_SECRET="custom-secret" \
  eagleeye:latest
```

## 📊 Volume Management

### Data Volumes

The Docker setup uses volumes for persistent data:

| Volume | Purpose | Mount Point |
|--------|---------|------------|
| `./data` | Database, embeddings, student images | `/app/data` |
| `./outputs` | Attendance & violation reports | `/app/outputs` |
| `./logs` | Application logs | `/app/logs` |

### Named Volumes (Alternative)

For production, use named volumes for better management:

```yaml
volumes:
  eagleeye-data:
  eagleeye-logs:

services:
  eagleeye:
    volumes:
      - eagleeye-data:/app/data
      - eagleeye-logs:/app/logs
```

List and inspect volumes:

```bash
docker volume ls
docker volume inspect eagleeye-data
```

## 🐳 Docker Commands Reference

### Build

```bash
# Development image
docker build -t eagleeye:dev .

# Production image
docker build -f Dockerfile.prod -t eagleeye:latest .

# With custom tag
docker build -t your-registry/eagleeye:1.0 .
```

### Run

```bash
# Interactive mode (for debugging)
docker run -it --rm \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  --device /dev/video0:/dev/video0 \
  eagleeye:dev

# Detached mode (production)
docker run -d \
  --name eagleeye \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  eagleeye:latest
```

### Management

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# Stop container
docker stop eagleeye

# Start container
docker start eagleeye

# Remove container
docker rm eagleeye

# View logs
docker logs eagleeye
docker logs -f eagleeye  # Follow logs

# Execute command in container
docker exec -it eagleeye bash
docker exec -it eagleeye python app.dashboard
```

## 📈 Docker Compose Commands

```bash
# Start services
docker-compose up

# Start services in background
docker-compose up -d

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down

# Stop services and remove volumes
docker-compose down -v

# View logs
docker-compose logs
docker-compose logs -f eagleeye

# Execute command
docker-compose exec eagleeye bash

# Scale services (if applicable)
docker-compose up -d --scale eagleeye=3
```

## 🌐 Networking

### Access the Application

- **Local**: `http://localhost:5000`
- **Remote (if exposed)**: `http://<docker-host-ip>:5000`

### Port Mapping

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8000:5000"  # Access via http://localhost:8000
```

### With Reverse Proxy (Nginx)

For production deployment behind Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 📦 Registry and Image Management

### Push to Docker Registry

```bash
# Login to registry
docker login your-registry

# Tag image
docker tag eagleeye:latest your-registry/eagleeye:1.0

# Push image
docker push your-registry/eagleeye:1.0

# Pull image
docker pull your-registry/eagleeye:1.0
```

### Cleanup

```bash
# Remove unused images
docker image prune

# Remove all unused images (force)
docker image prune -a

# Remove image
docker rmi eagleeye:dev
```

## 🔍 Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker logs -f eagleeye
```

Common causes:
- Missing environment variables
- Port already in use
- Database connection issues

### Camera/Audio Not Working

```bash
# Verify device access inside container
docker exec eagleeye ls -la /dev/video0
docker exec eagleeye arecord -l

# Check device permissions
docker run --privileged -it eagleeye bash
```

### Permission Denied Errors

Run with elevated privileges:
```bash
docker-compose down
sudo docker-compose up
```

Or add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Out of Memory

Increase Docker memory limit:
```bash
docker run -d \
  -m 4g \
  --memory-swap 4g \
  eagleeye:latest
```

Or update `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## 🚀 Advanced Deployments

### Kubernetes Deployment

You can extend this to Kubernetes using:
```bash
docker save eagleeye:latest | gzip > eagleeye.tar.gz
# Transfer to K8s cluster and load
```

### Docker Swarm

```bash
# Initialize Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml eagleeye

# List services
docker service ls
```

### CI/CD Integration

Example GitHub Actions workflow:
```yaml
name: Build and Deploy to Docker
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: docker/build-push-action@v2
        with:
          context: .
          push: true
          tags: your-registry/eagleeye:latest
```

## 📝 Health Monitoring

The Docker containers include health checks. View health status:

```bash
docker ps  # Check STATUS column

# View health check logs
docker inspect eagleeye | grep -A 10 -i health
```

## 🔒 Security Considerations

1. **Use strong secrets**: Change `EAGLE_EYE_SECRET` in production
2. **Non-root user**: Consider adding a non-root user to the Dockerfile
3. **Network isolation**: Use custom networks instead of host mode
4. **Image scanning**: Scan images for vulnerabilities:
   ```bash
   docker scan eagleeye:latest
   ```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/dev-best-practices/dockerfile_best-practices/)
