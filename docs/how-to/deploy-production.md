# Deploy to Production

A comprehensive guide for deploying Enterprise MCP Bridge to production environments.

## Production Checklist

Before deploying to production:

- [ ] Security hardening completed
- [ ] TLS/HTTPS configured
- [ ] Authentication/authorization in place
- [ ] Resource limits set
- [ ] Monitoring and logging configured
- [ ] Backup strategy defined
- [ ] Disaster recovery plan ready
- [ ] Load testing completed

## Deployment Options

### 1. Docker Container

The recommended approach for most deployments.

#### Basic Docker Deployment

```bash
docker run -d \
  --name enterprise-mcp-bridge \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MCP_SERVER_COMMAND="your-mcp-command" \
  -e MCP_BASE_PATH="/api/mcp" \
  -v /data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

#### Production Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    container_name: enterprise-mcp-bridge
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-memory /data/memory.json
      - MCP_BASE_PATH=/api/mcp
      - LOG_LEVEL=info
      - WORKERS=4
    volumes:
      - ./data:/data
      - ./logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mcp-network

  # Optional: Redis for session management
  redis:
    image: redis:7-alpine
    container_name: mcp-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - mcp-network

volumes:
  redis-data:

networks:
  mcp-network:
    driver: bridge
```

Deploy:

```bash
docker-compose up -d
```

### 2. Kubernetes Deployment

#### Basic Kubernetes Manifest

Create `mcp-bridge-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enterprise-mcp-bridge
  labels:
    app: mcp-bridge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-bridge
  template:
    metadata:
      labels:
        app: mcp-bridge
    spec:
      containers:
      - name: mcp-bridge
        image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MCP_SERVER_COMMAND
          value: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
        - name: MCP_BASE_PATH
          value: "/api/mcp"
        - name: SESSION_MANAGER_TYPE
          value: "redis"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: redis-url
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: mcp-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-bridge-service
spec:
  selector:
    app: mcp-bridge
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mcp-data-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

Deploy:

```bash
kubectl apply -f mcp-bridge-deployment.yaml
```

#### Helm Chart Deployment

Create `values.yaml`:

```yaml
replicaCount: 3

image:
  repository: ghcr.io/inxm-ai/enterprise-mcp-bridge
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: mcp.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mcp-tls
      hosts:
        - mcp.example.com

env:
  MCP_SERVER_COMMAND: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
  MCP_BASE_PATH: "/api/mcp"
  SESSION_MANAGER_TYPE: "redis"
  LOG_LEVEL: "info"

secrets:
  redis_url: "redis://redis:6379"

resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"

persistence:
  enabled: true
  size: 10Gi
  storageClass: "standard"

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: false

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

### 3. Cloud Platform Deployments

#### AWS ECS

```json
{
  "family": "enterprise-mcp-bridge",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "mcp-bridge",
      "image": "ghcr.io/inxm-ai/enterprise-mcp-bridge:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MCP_SERVER_COMMAND",
          "value": "npx -y @modelcontextprotocol/server-memory /data/memory.json"
        }
      ],
      "secrets": [
        {
          "name": "REDIS_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:redis-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/enterprise-mcp-bridge",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```bash
gcloud run deploy enterprise-mcp-bridge \
  --image=ghcr.io/inxm-ai/enterprise-mcp-bridge:latest \
  --platform=managed \
  --region=us-central1 \
  --set-env-vars="MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-memory /data/memory.json" \
  --set-env-vars="MCP_BASE_PATH=/api/mcp" \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=10 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
az container create \
  --resource-group mcp-resources \
  --name enterprise-mcp-bridge \
  --image ghcr.io/inxm-ai/enterprise-mcp-bridge:latest \
  --ports 8000 \
  --environment-variables \
    MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" \
    MCP_BASE_PATH="/api/mcp" \
  --cpu 1 \
  --memory 1.5 \
  --restart-policy Always
```

## Security Configuration

### 1. TLS/HTTPS Setup

#### Using Nginx Reverse Proxy

Create `nginx.conf`:

```nginx
upstream mcp_bridge {
    server localhost:8000;
}

server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://mcp_bridge;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name mcp.example.com;
    return 301 https://$server_name$request_uri;
}
```

### 2. Authentication Proxy

#### OAuth2 Proxy

```yaml
# docker-compose.yml with OAuth2 Proxy
services:
  oauth2-proxy:
    image: quay.io/oauth2-proxy/oauth2-proxy:latest
    command:
      - --http-address=0.0.0.0:4180
      - --upstream=http://mcp-bridge:8000
      - --provider=keycloak-oidc
      - --client-id=mcp-bridge-client
      - --client-secret=YOUR_CLIENT_SECRET
      - --oidc-issuer-url=https://keycloak.example.com/realms/mcp
      - --cookie-secret=RANDOM_SECRET_HERE
      - --email-domain=*
    ports:
      - "4180:4180"
    depends_on:
      - mcp-bridge
```

### 3. API Key Authentication

Set up API key middleware:

```python
# Add to your deployment
API_KEYS="key1,key2,key3"
```

Or use external auth service.

## Environment Configuration

### Required Variables

```bash
# MCP Server Configuration
MCP_SERVER_COMMAND="your-mcp-server-command"

# Optional: Base path for API
MCP_BASE_PATH="/api/mcp"

# Session Management (for multi-instance)
SESSION_MANAGER_TYPE="redis"
REDIS_URL="redis://redis:6379/0"

# Logging
LOG_LEVEL="info"

# OAuth (if using)
OAUTH_ISSUER_URL="https://auth.example.com"
OAUTH_CLIENT_ID="mcp-bridge"
OAUTH_CLIENT_SECRET="secret"
```

### Security Variables

```bash
# CORS settings
CORS_ORIGINS="https://app.example.com,https://admin.example.com"

# Session timeout
SESSION_TIMEOUT_SECONDS=1800
```

## Monitoring and Observability

### Prometheus Metrics

The bridge exposes Prometheus metrics at `/metrics`:

```yaml
scrape_configs:
  - job_name: 'mcp-bridge'
    static_configs:
      - targets: ['mcp-bridge:8000']
```

### Logging Configuration

Configure structured logging:

```bash
LOG_LEVEL=info
LOG_FORMAT=json
LOG_OUTPUT=/logs/app.log
```

### Health Checks

Use `/health` endpoint:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.4.2",
  "uptime": "2h15m30s"
}
```

## Backup and Recovery

### Data Backup

```bash
# Backup data directory
docker run --rm \
  -v mcp-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/mcp-data-$(date +%Y%m%d).tar.gz /data

# Automated backup with cron
0 2 * * * /path/to/backup-script.sh
```

### Disaster Recovery

1. **Session Recovery:** Use Redis persistence
2. **Data Recovery:** Restore from backups
3. **Configuration:** Store in version control
4. **Secrets:** Use secret management (Vault, AWS Secrets Manager)

## Performance Tuning

### Worker Configuration

```bash
# Adjust number of workers based on CPU cores
WORKERS=4

# Set worker class
WORKER_CLASS="uvicorn.workers.UvicornWorker"
```

### Resource Limits

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Caching

Enable caching for tool schemas:

```bash
ENABLE_SCHEMA_CACHE=true
SCHEMA_CACHE_TTL=3600
```

## Scaling Strategies

### Horizontal Scaling

1. **Shared Session Storage:** Use Redis
2. **Load Balancer:** Distribute traffic
3. **Auto-scaling:** Based on CPU/memory metrics

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-bridge-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: enterprise-mcp-bridge
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

Increase resources per instance:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

## Testing Production Deployment

### Load Testing

```bash
# Install k6
brew install k6  # macOS
# or download from https://k6.io

# Create load test script (load-test.js)
cat > load-test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  let response = http.get('https://mcp.example.com/tools');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
EOF

# Run load test
k6 run load-test.js
```

### Smoke Testing

```bash
#!/bin/bash
# smoke-test.sh

BASE_URL="https://mcp.example.com"

# Test health endpoint
curl -f $BASE_URL/health || exit 1

# Test tools listing
curl -f $BASE_URL/tools || exit 1

# Test session creation
SESSION_ID="test-$(date +%s)"
curl -f -X POST $BASE_URL/session/start \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\"}" || exit 1

# Clean up
curl -X POST $BASE_URL/session/$SESSION_ID/close

echo "All smoke tests passed!"
```

## Troubleshooting Production Issues

### Common Issues

**High Memory Usage**
```bash
# Check container memory
docker stats enterprise-mcp-bridge

# Solution: Increase memory limits or reduce workers
```

**Session Timeouts**
```bash
# Increase timeout
SESSION_TIMEOUT_SECONDS=3600
```

**Slow Response Times**
```bash
# Enable caching
ENABLE_SCHEMA_CACHE=true

# Increase workers
WORKERS=8
```

## Security Checklist

- [ ] HTTPS/TLS enabled
- [ ] Authentication configured
- [ ] API keys or OAuth in place
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Secrets stored securely
- [ ] Regular security updates
- [ ] Audit logging enabled
- [ ] Network policies configured
- [ ] Container security scanning

## Post-Deployment

### Monitoring

Set up alerts for:
- High error rates
- Slow response times
- High memory/CPU usage
- Session creation failures

### Maintenance

Schedule regular:
- Security updates
- Dependency updates
- Log rotation
- Backup verification
- Performance reviews

## Summary

You now know how to:

✅ Deploy to Docker, Kubernetes, and cloud platforms  
✅ Configure security and authentication  
✅ Set up monitoring and logging  
✅ Implement backup and recovery  
✅ Scale horizontally and vertically  
✅ Test production deployments  

## Next Steps

- [Monitor Your Deployment](monitoring.md)
- [Configure OAuth](configure-oauth.md)
- [Security Best Practices](../explanation/security.md)

## Resources

- [Configuration Reference](../reference/configuration.md)
- [Example Deployments](../reference/examples.md)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
