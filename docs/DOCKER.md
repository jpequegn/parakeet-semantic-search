# Docker Deployment Guide

Complete guide for deploying Parakeet Semantic Search using Docker.

**Version**: 1.0.0
**Last Updated**: November 22, 2025

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Single Container Deployment](#single-container-deployment)
3. [Multi-Service with Docker Compose](#multi-service-with-docker-compose)
4. [Configuration](#configuration)
5. [Building Custom Images](#building-custom-images)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- 4GB RAM available for Docker
- 5GB free disk space

### Start All Services

```bash
# Clone repository
git clone https://github.com/jpequegn/parakeet-semantic-search.git
cd parakeet-semantic-search

# Start services with Docker Compose
docker-compose up -d

# Services will be available at:
# - REST API: http://localhost:8000
# - Web UI: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### Access Services

```bash
# API health check
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
docker-compose logs -f web

# Stop services
docker-compose down
```

---

## Single Container Deployment

### Build Image

```bash
# Build Docker image
docker build -t parakeet-search:latest .

# Build with custom tag
docker build -t parakeet-search:v1.0.0 .

# View image info
docker images parakeet-search
```

### Run REST API Only

```bash
# Run REST API container
docker run -d \
  --name parakeet-api \
  -p 8000:8000 \
  -v parakeet-data:/app/data \
  -e LOG_LEVEL=info \
  parakeet-search:latest

# Access API
curl http://localhost:8000/health

# View logs
docker logs -f parakeet-api

# Stop container
docker stop parakeet-api
docker rm parakeet-api
```

### Run Web UI Only

```bash
# Run Streamlit container
docker run -d \
  --name parakeet-web \
  -p 8501:8501 \
  -v parakeet-data:/app/data \
  parakeet-search:latest \
  streamlit run apps/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Access Web UI
# Open http://localhost:8501 in browser

# View logs
docker logs -f parakeet-web

# Stop container
docker stop parakeet-web
docker rm parakeet-web
```

### Interactive Mode

```bash
# Run container interactively
docker run -it \
  -v parakeet-data:/app/data \
  parakeet-search:latest \
  /bin/bash

# Inside container, you can:
parakeet-search search "machine learning"
python scripts/create_vector_store.py
pytest tests/ -v
```

---

## Multi-Service with Docker Compose

### Start Services

```bash
# Start all services in background
docker-compose up -d

# Start with output in foreground (Ctrl+C to stop)
docker-compose up

# Start specific service
docker-compose up -d api
docker-compose up -d web
```

### Service Architecture

```
docker-compose.yml
├── api (REST API on port 8000)
│   ├── FastAPI application
│   ├── Port: 8000
│   ├── Health check: /health
│   └── Data volume: /app/data
│
├── web (Streamlit on port 8501)
│   ├── Streamlit application
│   ├── Port: 8501
│   ├── Depends on: api (healthy)
│   └── Data volume: /app/data
│
└── volumes
    ├── parakeet-data (shared data)
    └── parakeet-logs (shared logs)
```

### Service Management

```bash
# View running services
docker-compose ps

# View service logs
docker-compose logs                    # All services
docker-compose logs -f api            # API only
docker-compose logs -f web            # Web only
docker-compose logs --tail=100 api    # Last 100 lines

# Restart services
docker-compose restart                # All
docker-compose restart api            # API only

# Scale services
docker-compose up -d --scale api=2    # Multiple API instances

# Remove all services and volumes
docker-compose down                   # Keep volumes
docker-compose down -v                # Remove volumes too
```

### Health Checks

```bash
# Check service health
docker-compose ps

# Expected output:
# NAME              STATUS              PORTS
# parakeet-api      Up 1 minute         0.0.0.0:8000->8000/tcp
# parakeet-web      Up 1 minute         0.0.0.0:8501->8501/tcp

# Manual health check
curl http://localhost:8000/health
# Response: {"status": "healthy", "version": "1.0.0", ...}
```

---

## Configuration

### Environment Variables

```yaml
# In docker-compose.yml or docker run -e
LOG_LEVEL=info              # debug, info, warning, error
CACHE_ENABLED=true          # Enable/disable caching
CACHE_TTL=3600              # Cache time-to-live (seconds)
CACHE_MAX_SIZE=1000         # Max cached queries
API_HOST=0.0.0.0            # API listen address
API_PORT=8000               # API port
```

### Volume Configuration

```bash
# Named volumes (recommended)
docker run -v parakeet-data:/app/data parakeet-search

# Bind mounts (local directory)
docker run -v /path/to/data:/app/data parakeet-search

# Read-only volume
docker run -v /path/to/data:/app/data:ro parakeet-search
```

### Custom Entrypoint

```bash
# Override default command
docker run parakeet-search \
  streamlit run apps/streamlit_app.py

# Run CLI commands
docker run -it parakeet-search \
  parakeet-search search "query"

# Run tests
docker run parakeet-search \
  pytest tests/ -v
```

---

## Building Custom Images

### Production-Optimized Build

```bash
# Build with specific version
docker build -t parakeet-search:v1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) .

# Build for specific Python version
docker build -t parakeet-search:py3.12 .
```

### Image Information

```bash
# Show image layers
docker history parakeet-search:latest

# Inspect image
docker inspect parakeet-search:latest

# Get image size
docker images parakeet-search --human-readable
```

### Push to Registry

```bash
# Tag for Docker Hub
docker tag parakeet-search:latest myusername/parakeet-search:latest

# Login to Docker Hub
docker login

# Push image
docker push myusername/parakeet-search:latest

# Pull from registry
docker pull myusername/parakeet-search:latest
```

---

## Production Deployment

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: parakeet-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: parakeet-api
  template:
    metadata:
      labels:
        app: parakeet-api
    spec:
      containers:
      - name: parakeet-api
        image: parakeet-search:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: CACHE_ENABLED
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: parakeet-api-service
spec:
  selector:
    app: parakeet-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy to Kubernetes**:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get deployments
kubectl get services
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml parakeet

# View stack
docker stack ls
docker stack services parakeet

# Remove stack
docker stack rm parakeet
```

### Environment Setup for Production

Create `.env.production`:

```
LOG_LEVEL=warning
CACHE_ENABLED=true
CACHE_TTL=7200
CACHE_MAX_SIZE=5000
API_WORKERS=4
```

Deploy with environment file:

```bash
docker run --env-file .env.production parakeet-search
# Or with compose:
docker-compose --env-file .env.production up -d
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker logs parakeet-api

# Run in interactive mode to debug
docker run -it parakeet-search /bin/bash

# Check container status
docker ps -a
docker inspect parakeet-api
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Use different port
docker run -p 8001:8000 parakeet-search

# Or in docker-compose.yml:
# ports:
#   - "8001:8000"
```

### Out of Memory

```bash
# Check container memory usage
docker stats parakeet-api

# Limit memory in run command
docker run -m 4g parakeet-search

# Increase Docker memory allocation
# Docker Desktop: Preferences → Resources → Memory
```

### Database Connection Error

```bash
# Ensure data volume exists
docker volume ls
docker volume inspect parakeet-data

# Check data directory
docker exec parakeet-api ls -lh /app/data

# Create database
docker exec parakeet-api python scripts/create_vector_store.py
```

### Slow Performance

```bash
# Check resource usage
docker stats

# Monitor logs for errors
docker logs -f parakeet-api

# Clear cache
curl -X POST http://localhost:8000/cache/clear

# Restart service
docker-compose restart api
```

### Network Issues

```bash
# Check network
docker network ls
docker network inspect parakeet-network

# Test connectivity between containers
docker exec parakeet-api \
  curl http://localhost:8000/health

# Check DNS
docker exec parakeet-api \
  cat /etc/resolv.conf
```

---

## Performance Optimization

### Image Size Optimization

**Strategies**:
- Multi-stage build (reduces final image size)
- Use slim Python images
- Remove build tools from final stage
- Clean package manager cache

**Current Image Size**:
- Uncompressed: ~1.2GB
- Compressed: ~400MB

### Runtime Optimization

```bash
# Run with resource limits
docker run \
  --memory=4g \
  --cpus=2 \
  parakeet-search

# Use network host mode (Linux only)
docker run --network host parakeet-search
```

### Caching Strategy

```bash
# Docker layer caching during build
# Good: Stable dependencies first
# Better: Only add changed code

# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build .

# Export cache
docker build --cache-from type=registry,ref=myregistry/parakeet:buildcache .
```

---

## Monitoring & Logging

### Container Monitoring

```bash
# Real-time stats
docker stats parakeet-api

# Historical logs
docker logs --timestamps parakeet-api
docker logs --since 2025-11-22 parakeet-api
docker logs --until 1m parakeet-api

# Log aggregation
docker logs parakeet-api 2>&1 | tee app.log
```

### Health Check Monitoring

```bash
# Check health status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl http://localhost:8000/health
curl http://localhost:8501/health  # Streamlit doesn't have health endpoint

# Metrics
curl http://localhost:8000/cache/stats
```

---

## Development with Docker

### Development Container

```bash
# Run with source code mounted
docker run -it \
  -v $(pwd):/app \
  -v /app/venv \
  parakeet-search \
  /bin/bash

# Inside container:
pip install -e .
pytest tests/ -v
```

### Docker Compose for Development

Add to `docker-compose.yml`:

```yaml
dev:
  build: .
  volumes:
    - .:/app
    - /app/venv
  environment:
    - LOG_LEVEL=debug
  command: /bin/bash
  stdin_open: true
  tty: true
```

Start development:

```bash
docker-compose run dev
# Inside: pip install -e . && bash
```

---

## Cleanup

### Remove Images

```bash
# Remove image
docker rmi parakeet-search:latest

# Remove all dangling images
docker image prune

# Remove all images
docker rmi $(docker images -q)
```

### Remove Containers

```bash
# Remove stopped containers
docker container prune

# Remove all containers
docker rm $(docker ps -aq)
```

### Remove Volumes

```bash
# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm parakeet-data

# Remove all volumes
docker volume rm $(docker volume ls -q)
```

---

## Best Practices

✅ **Do**:
- Use multi-stage builds
- Pin base image versions
- Use .dockerignore
- Run as non-root user
- Include health checks
- Use specific tags, not `latest`
- Use environment variables
- Store data in volumes
- Monitor container resources

❌ **Don't**:
- Run as root
- Use `latest` tag in production
- Include test/dev files in image
- Hardcode configuration
- Run unnecessary services
- Ignore security warnings
- Store data in container
- Overcommit resources

---

## Support & Resources

- **Documentation**: See [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Docker Docs**: https://docs.docker.com/
- **Docker Hub**: https://hub.docker.com/
- **Issues**: [GitHub Issues](https://github.com/jpequegn/parakeet-semantic-search/issues)

---

**Last Updated**: November 22, 2025
**Status**: Production Ready
