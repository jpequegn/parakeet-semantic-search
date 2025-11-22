# Deployment Guide

Complete instructions for deploying Parakeet Semantic Search in various environments.

**Version**: 1.0.0
**Last Updated**: November 22, 2025

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration](#configuration)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.12+
- pip or uv package manager
- Virtual environment tool (venv)
- 4GB RAM minimum
- 2GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/jpequegn/parakeet-semantic-search.git
cd parakeet-semantic-search

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
parakeet-search --version
```

### Setup Data

```bash
# Create vector store from podcast data
python scripts/create_vector_store.py \
  --p3-db /path/to/p3.duckdb \
  --output-dir ./data

# Verify data loaded
ls -lh ./data/parakeet.lance
```

### Running Services

**CLI Only**:
```bash
parakeet-search search "machine learning"
```

**Streamlit Web UI**:
```bash
streamlit run apps/streamlit_app.py
# Opens at http://localhost:8501
```

**REST API**:
```bash
uvicorn apps.fastapi_app:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

**All Services**:
```bash
# Terminal 1: API
uvicorn apps.fastapi_app:app --port 8000

# Terminal 2: Web UI
streamlit run apps/streamlit_app.py

# Terminal 3: Use CLI
parakeet-search search "query"
```

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+ (optional)
- 4GB RAM available to Docker
- 5GB free disk space

### Single Container Deployment

```bash
# Build image
docker build -t parakeet-search:latest .

# Run container
docker run -d \
  --name parakeet \
  -p 8000:8000 \
  -p 8501:8501 \
  -v parakeet-data:/app/data \
  parakeet-search:latest

# Check status
docker logs parakeet
docker ps
```

**Access Services**:
- REST API: http://localhost:8000/docs
- Web UI: http://localhost:8501
- Health: http://localhost:8000/health

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.9'

services:
  parakeet:
    build: .
    container_name: parakeet-search
    ports:
      - "8000:8000"  # REST API
      - "8501:8501"  # Streamlit
    volumes:
      - parakeet-data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=info
      - CACHE_ENABLED=true
      - CACHE_TTL=3600
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

volumes:
  parakeet-data:
    driver: local
```

**Deploy with Compose**:
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f parakeet

# Stop services
docker-compose down
```

### Build Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY apps/ apps/
COPY scripts/ scripts/
COPY data/ data/

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API by default (Streamlit can be run separately)
CMD ["uvicorn", "apps.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Production Deployment

### Using Gunicorn + Uvicorn

```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn apps.fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

### Using Systemd

Create `/etc/systemd/system/parakeet.service`:

```ini
[Unit]
Description=Parakeet Semantic Search API
After=network.target

[Service]
Type=notify
User=parakeet
WorkingDirectory=/opt/parakeet
ExecStart=/opt/parakeet/venv/bin/uvicorn apps.fastapi_app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
Restart=always
RestartSec=10

# Resource limits
MemoryMax=4G
CPUQuota=300%

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable parakeet
sudo systemctl start parakeet
sudo systemctl status parakeet
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/parakeet`:

```nginx
upstream parakeet_api {
    server localhost:8000;
}

upstream parakeet_web {
    server localhost:8501;
}

server {
    listen 80;
    listen [::]:80;
    server_name parakeet.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name parakeet.example.com;

    # SSL certificates (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/parakeet.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/parakeet.example.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    # API proxy
    location /api/ {
        proxy_pass http://parakeet_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Streamlit proxy
    location / {
        proxy_pass http://parakeet_web;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streamlit-specific headers
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check
    location /health {
        proxy_pass http://parakeet_api;
        access_log off;
    }
}
```

**Enable site**:
```bash
sudo ln -s /etc/nginx/sites-available/parakeet /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Cloud Deployment

### AWS EC2

**Launch Instance**:
```bash
# Instance specs
- OS: Ubuntu 22.04 LTS
- Type: t3.large (2 vCPU, 8GB RAM)
- Storage: 20GB gp3
- Security: Enable ports 22, 80, 443
```

**Setup Script**:
```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.12 python3.12-venv python3-pip
sudo apt install -y nginx curl git

# Clone repository
cd /opt
sudo git clone https://github.com/jpequegn/parakeet-semantic-search.git parakeet
cd parakeet

# Setup virtual environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# Create systemd service
sudo cp docs/parakeet.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable parakeet
sudo systemctl start parakeet

# Setup Nginx
sudo cp docs/nginx.conf /etc/nginx/sites-available/parakeet
sudo ln -s /etc/nginx/sites-available/parakeet /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

echo "Deployment complete!"
```

### Heroku

```bash
# Create Procfile
echo "web: uvicorn apps.fastapi_app:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create parakeet-search
git push heroku main

# View logs
heroku logs --tail
```

### DigitalOcean App Platform

Create `app.yaml`:
```yaml
name: parakeet-search
services:
  - name: api
    github:
      repo: jpequegn/parakeet-semantic-search
      branch: main
    build_command: pip install -r requirements.txt
    run_command: uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8080
    http_port: 8080
    source_dir: /
```

---

## Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=info                  # debug, info, warning, error
LOG_FILE=/var/log/parakeet.log

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL=3600                 # 1 hour
CACHE_MAX_SIZE=1000

# Database
VECTOR_STORE_PATH=/app/data/parakeet.lance

# Security
ALLOW_CORS=false
JWT_SECRET=your-secret-key
```

### Configuration File

Create `config.yaml`:
```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

cache:
  enabled: true
  ttl_seconds: 3600
  max_size: 1000

database:
  path: /app/data/parakeet.lance

logging:
  level: info
  file: /var/log/parakeet.log

security:
  allow_cors: false
  jwt_secret: your-secret-key
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-22T12:34:56Z",
  "uptime_seconds": 3600.5
}
```

### Logs Monitoring

```bash
# Follow logs in real-time
journalctl -u parakeet -f

# Search logs
journalctl -u parakeet --grep="error"

# Last 100 lines
journalctl -u parakeet -n 100

# Today's logs
journalctl -u parakeet --since today
```

### Performance Monitoring

```bash
# Check system resources
docker stats parakeet

# Monitor API metrics
curl http://localhost:8000/cache/stats

# Monitor response times
ab -n 100 -c 10 http://localhost:8000/health
```

### Backup Strategy

```bash
# Daily backup
0 2 * * * tar -czf /backups/parakeet-$(date +%Y%m%d).tar.gz /opt/parakeet/data/

# Weekly full backup
0 3 * * 0 tar -czf /backups/parakeet-full-$(date +%Y%m%d).tar.gz /opt/parakeet/

# Restore from backup
tar -xzf /backups/parakeet-20251122.tar.gz -C /opt/parakeet/
```

---

## Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn apps.fastapi_app:app --port 8001
```

**Out of Memory**:
```bash
# Check memory usage
docker stats

# Increase memory limit
docker update --memory 8g parakeet

# Clear cache
curl -X POST http://localhost:8000/cache/clear
```

**Database Connection Error**:
```bash
# Check database exists
ls -lh /app/data/parakeet.lance

# Recreate database
python scripts/create_vector_store.py --force
```

**Slow Performance**:
```bash
# Check cache hit rate
curl http://localhost:8000/cache/stats

# Monitor system resources
top -p $(pgrep -f uvicorn)

# Check log for errors
journalctl -u parakeet -p err -n 50
```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=debug uvicorn apps.fastapi_app:app --reload

# Enable verbose API output
PYTHONUNBUFFERED=1 LOG_LEVEL=debug python -m uvicorn apps.fastapi_app:app

# Use Python debugger
python -m pdb -m uvicorn apps.fastapi_app:app
```

---

## Scaling Strategies

### Vertical Scaling

Increase resources on single instance:
- More CPU cores
- More RAM (8GB → 16GB → 32GB)
- Faster disk (SSD preferred)
- GPU acceleration (optional)

### Horizontal Scaling

Deploy multiple instances:
```
Load Balancer (Nginx/HAProxy)
├─ Instance 1 (API + Cache)
├─ Instance 2 (API + Cache)
└─ Instance 3 (API + Cache)
        ↓
Shared Database (Volumes/Cloud Storage)
```

### Caching Strategy

For distributed setup, consider:
- Redis for shared cache
- Memcached for distributed memory
- Database-level query caching

---

## Maintenance

### Regular Tasks

**Daily**:
- Monitor logs for errors
- Check disk space
- Verify API availability

**Weekly**:
- Review performance metrics
- Check cache hit rate
- Backup data

**Monthly**:
- Update dependencies
- Security patches
- Performance analysis

**Quarterly**:
- Database optimization
- Capacity planning
- Feature updates

---

## Support & Resources

- **Documentation**: See [README.md](../README.md)
- **Architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Issues**: [GitHub Issues](https://github.com/jpequegn/parakeet-semantic-search/issues)

---

**Last Updated**: November 22, 2025
**Status**: Production Ready
