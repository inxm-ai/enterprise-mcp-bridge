# Deploy to Kubernetes

Guide for deploying Enterprise MCP Bridge to Kubernetes clusters.

## Quick Start

### Basic Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enterprise-mcp-bridge
  labels:
    app: mcp-bridge
spec:
  replicas: 2
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
        emptyDir: {}
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
  type: ClusterIP
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Production Deployment

### With Persistent Storage

```yaml
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
  storageClassName: standard
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enterprise-mcp-bridge
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
        env:
        - name: MCP_SERVER_COMMAND
          value: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
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
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: mcp-data-pvc
```

### With ConfigMap

Create `configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-bridge-config
data:
  MCP_BASE_PATH: "/api/mcp"
  LOG_LEVEL: "info"
  LOG_FORMAT: "json"
  SESSION_MANAGER_TYPE: "redis"
```

Reference in deployment:

```yaml
spec:
  containers:
  - name: mcp-bridge
    envFrom:
    - configMapRef:
        name: mcp-bridge-config
```

### With Secrets

Create secrets:

```bash
kubectl create secret generic mcp-secrets \
  --from-literal=redis-url=redis://redis:6379 \
  --from-literal=oauth-client-secret=your-secret
```

Reference in deployment:

```yaml
spec:
  containers:
  - name: mcp-bridge
    env:
    - name: REDIS_URL
      valueFrom:
        secretKeyRef:
          name: mcp-secrets
          key: redis-url
    - name: OAUTH_CLIENT_SECRET
      valueFrom:
        secretKeyRef:
          name: mcp-secrets
          key: oauth-client-secret
```

## Ingress

### NGINX Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-bridge-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - mcp.example.com
    secretName: mcp-tls
  rules:
  - host: mcp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcp-bridge-service
            port:
              number: 80
```

## Auto-Scaling

### Horizontal Pod Autoscaler

```yaml
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Redis for Session Management

### Deploy Redis

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

## Complete Example

Full production-ready configuration:

```yaml
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: mcp-bridge
---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-bridge-config
  namespace: mcp-bridge
data:
  MCP_BASE_PATH: "/api/mcp"
  LOG_LEVEL: "info"
  SESSION_MANAGER_TYPE: "redis"
---
# Secrets
apiVersion: v1
kind: Secret
metadata:
  name: mcp-secrets
  namespace: mcp-bridge
type: Opaque
stringData:
  redis-url: "redis://redis:6379"
  oauth-client-secret: "your-secret-here"
---
# PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mcp-data-pvc
  namespace: mcp-bridge
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
---
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enterprise-mcp-bridge
  namespace: mcp-bridge
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
        envFrom:
        - configMapRef:
            name: mcp-bridge-config
        env:
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
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
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
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: mcp-data-pvc
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: mcp-bridge-service
  namespace: mcp-bridge
spec:
  selector:
    app: mcp-bridge
  ports:
  - port: 80
    targetPort: 8000
---
# HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-bridge-hpa
  namespace: mcp-bridge
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

Deploy:
```bash
kubectl apply -f production.yaml
```

## Monitoring

### ServiceMonitor for Prometheus

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mcp-bridge-metrics
  namespace: mcp-bridge
spec:
  selector:
    matchLabels:
      app: mcp-bridge
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n mcp-bridge
kubectl describe pod <pod-name> -n mcp-bridge
```

### View Logs

```bash
kubectl logs -f deployment/enterprise-mcp-bridge -n mcp-bridge
```

### Check Events

```bash
kubectl get events -n mcp-bridge --sort-by='.lastTimestamp'
```

## Next Steps

- [Deploy to Production](deploy-production.md)
- [Run in Docker](docker.md)
- [Configuration Reference](../reference/configuration.md)
