# Docker Quick Start Guide for Eagle Eye

## 🚀 Fastest Way to Deploy

### Option 1: Development (Easy Setup)
```bash
# Clone and navigate to project
cd EagleEye

# Start with Docker Compose
docker-compose up --build

# Access at http://localhost:5000
```

### Option 2: Production (Optimized)
```bash
# Set up environment
cp .env.example .env
# Edit .env with your settings

# Start with production compose
docker-compose -f docker-compose.prod.yml up -d

# Access at http://localhost:5000
```

## 📋 What Each File Does

| File | Purpose |
|------|---------|
| `Dockerfile` | Development image with Python and dependencies |
| `Dockerfile.prod` | Production image with Gunicorn + Eventlet |
| `docker-compose.yml` | Development setup with camera/audio passthrough |
| `docker-compose.prod.yml` | Production setup with volumes, logging, optional Nginx |
| `.dockerignore` | Exclude unnecessary files from image |
| `.env.example` | Configuration template (copy to `.env`) |
| `nginx.conf` | Reverse proxy configuration (optional) |
| `DEPLOYMENT.md` | Comprehensive deployment guide |

## 🎯 Common Tasks

### Build the Image
```bash
# Development
docker build -t eagleeye:dev .

# Production
docker build -f Dockerfile.prod -t eagleeye:latest .
```

### Run Container
```bash
# Development (interactive)
docker-compose up

# Production (background)
docker-compose -f docker-compose.prod.yml up -d
```

### View Logs
```bash
# Docker Compose
docker-compose logs -f

# Single container
docker logs -f eagleeye
```

### Stop & Clean Up
```bash
docker-compose down      # Stop and remove
docker-compose down -v   # Remove volumes too
```

### Execute Commands in Container
```bash
docker-compose exec eagleeye bash
docker-compose exec eagleeye python -c "import app; print(app)"
```

## 🎥 Camera & Audio Configuration

### Linux/Mac
Add device paths to `docker-compose.yml`:
```yaml
devices:
  - /dev/video0:/dev/video0  # Change to /dev/video1, etc. for different cameras
  - /dev/snd:/dev/snd         # Audio device
```

### Windows
Use host networking mode or Docker Desktop with WSL2:
```yaml
network_mode: host
```

## 🔐 Security Setup

1. **Generate Secret Key:**
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Create .env file:**
   ```bash
   cp .env.example .env
   # Edit .env with values:
   # EAGLE_EYE_SECRET=<generated-key>
   # FLASK_ENV=production
   ```

3. **Run with SSL (optional):**
   - Place certificates in `./ssl/` directory
   - Use `docker-compose.prod.yml` with Nginx

## 📊 Data Management

### View Data Files
```bash
docker-compose exec eagleeye ls -la /app/data
docker-compose exec eagleeye ls -la /app/outputs
```

### Backup Data
```bash
docker run --rm -v eagleeye-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/eagleeye-backup.tar.gz -C /data .
```

### Restore Data
```bash
docker run --rm -v eagleeye-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/eagleeye-backup.tar.gz -C /data
```

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs eagleeye

# Common issues:
# - Port 5000 already in use: Change port in docker-compose.yml
# - Camera not found: Verify /dev/video0 exists and is accessible
# - Missing .env: Copy .env.example to .env
```

### Camera/Audio Issues
```bash
# Check device availability
ls -la /dev/video*
arecord -l

# Test inside container
docker-compose exec eagleeye bash
# Inside container:
ls -la /dev/video0
arecord -d 5 test.wav
```

### Permission Errors
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker-compose up
```

### Memory Issues
Update `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase as needed
```

## 📈 Performance Optimization

### Enable GPU (NVIDIA)
```yaml
services:
  eagleeye:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

## 🌐 Expose to Network

### Local Network
Change port forwarding in `docker-compose.yml`:
```yaml
ports:
  - "0.0.0.0:5000:5000"  # Accessible from any interface
```

### Using Reverse Proxy (Nginx)
```bash
docker-compose -f docker-compose.prod.yml up -d
# Nginx runs on port 80/443
```

## 📝 Advanced: Push to Registry

```bash
# Tag image
docker tag eagleeye:latest myregistry/eagleeye:1.0

# Login and push
docker login myregistry
docker push myregistry/eagleeye:1.0

# Pull and run
docker pull myregistry/eagleeye:1.0
docker run -d -p 5000:5000 myregistry/eagleeye:1.0
```

## 🔗 Useful Links

- [Docker Docs](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Full Deployment Guide](./DEPLOYMENT.md)
- [Project README](./README.md)

---

**Need Help?** Check `DEPLOYMENT.md` for comprehensive documentation.
