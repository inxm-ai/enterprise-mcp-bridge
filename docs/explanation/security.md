# Security Model

Understanding the security architecture and best practices for Enterprise MCP Bridge.

## Overview

Enterprise MCP Bridge implements multiple layers of security to protect:
- User data and sessions
- API access
- Downstream service credentials
- MCP server processes

## Security Layers

### 1. Transport Security

**TLS/HTTPS**

All production deployments should use HTTPS:

```
Client → HTTPS → Load Balancer → HTTP → Bridge → MCP Server
```

- Encrypts data in transit
- Prevents man-in-the-middle attacks
- Validates server certificates

**Best Practices:**
- Use TLS 1.2 or higher
- Strong cipher suites only
- Valid SSL certificates (Let's Encrypt, commercial CA)
- HSTS headers enabled

### 2. Authentication

Multiple authentication strategies supported:

#### OAuth2 / OpenID Connect

Primary authentication method for production:

```
User → OAuth Provider (Keycloak) → Bridge
         ↓
   Access Token (JWT)
         ↓
   Token Validation
         ↓
   User Identity + Groups
```

**Features:**
- Industry-standard OAuth2 flows
- JWT token validation
- Group membership extraction
- Token expiration handling

#### Bearer Token

For service-to-service authentication:

```http
Authorization: Bearer SERVICE_TOKEN
```

**Use Cases:**
- Internal services
- CI/CD pipelines
- Monitoring systems

#### API Keys

Simple authentication for trusted clients:

```http
X-API-Key: YOUR_API_KEY
```

**Limitations:**
- No user context
- No fine-grained permissions
- Suited for internal use only

### 3. Authorization

#### User Isolation

Each user has isolated resources:

```
User A → Session A → MCP Server A → Data A
User B → Session B → MCP Server B → Data B
```

- Sessions are user-scoped
- No cross-user access
- Enforced at session manager level

#### Group-Based Access Control

Users inherit permissions from groups:

```yaml
User: john@example.com
Groups: [engineering, full-time]

Access:
  - /data/engineering/*
  - /data/full-time/*
  - /data/john@example.com/*
```

**Implementation:**
```python
# Group extracted from OAuth token
user_groups = token.get('groups', [])

# Data path resolution
if '{group_id}' in path:
    path = path.replace('{group_id}', user_groups[0])
```

### 4. Token Management

#### Token Exchange

Secure token exchange for downstream services:

```
User Token → Bridge → OAuth Provider → Service Token
                ↓
         Inject into MCP call
                ↓
         MCP Server → External API
```

**Security Benefits:**
- User tokens never exposed to MCP servers
- Service tokens scoped to specific APIs
- Automatic token refresh
- Token revocation support

#### Token Storage

Tokens stored securely:

- **In-Memory:** Development only
- **Redis:** Production with encryption
- **Encrypted at rest:** Using AES-256
- **Never logged:** Tokens redacted in logs

#### Token Lifecycle

```
1. User authenticates → Access token issued
2. Token validated on each request
3. Token refreshed before expiration
4. Token revoked on logout
5. Session cleaned up
```

### 5. Process Isolation

#### MCP Server Processes

Each session runs isolated MCP server:

```
Session 1 → MCP Process 1 (PID 1234)
Session 2 → MCP Process 2 (PID 1235)
Session 3 → MCP Process 3 (PID 1236)
```

**Isolation:**
- Separate process per session
- Independent memory space
- No shared state
- Resource limits enforced

#### Resource Limits

Prevent resource exhaustion:

```python
# CPU limit
cpu_limit = "500m"  # 0.5 CPU cores

# Memory limit
memory_limit = "512Mi"

# Timeout
timeout = 300  # 5 minutes
```

### 6. Input Validation

#### Request Validation

All inputs validated:

```python
# Pydantic models
class ToolCallRequest(BaseModel):
    param1: str = Field(..., max_length=1000)
    param2: int = Field(..., ge=0, le=1000000)

# Automatic validation
@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: ToolCallRequest):
    # request is validated
    pass
```

#### Injection Prevention

**SQL Injection:** The bridge itself doesn't use SQL databases directly. However, MCP servers you connect may use databases, so ensure they follow secure coding practices.

**Command Injection:** Mitigated by:
- No shell execution of user input
- Parameterized commands
- Input sanitization

```python
# Safe
subprocess.run(["python", "server.py", user_input])

# Unsafe (never done)
os.system(f"python server.py {user_input}")
```

**Path Traversal:** Prevented by:
- Path normalization
- Whitelist validation
- Boundary checks

```python
# Prevent directory traversal
safe_path = os.path.normpath(os.path.join(base_dir, user_path))
if not safe_path.startswith(base_dir):
    raise SecurityError("Invalid path")
```

### 7. Session Security

#### Session Hijacking Prevention

- Secure session IDs (cryptographically random)
- Session binding to user identity
- IP address validation (optional)
- Session timeout enforcement

#### Session Fixation Prevention

- New session ID after authentication
- Old session invalidated
- Token rotation on sensitive operations

#### CSRF Protection

- State parameter in OAuth flow
- Origin validation
- SameSite cookies

### 8. Secrets Management

#### Environment Variables

Secrets stored in environment:

```bash
# Good - from secret manager
OAUTH_CLIENT_SECRET="${SECRET_FROM_VAULT}"

# Bad - hardcoded
OAUTH_CLIENT_SECRET="hardcoded-secret"
```

#### Secret Files

Mount secrets as files:

```yaml
# Kubernetes
volumes:
  - name: secrets
    secret:
      secretName: mcp-secrets

volumeMounts:
  - name: secrets
    mountPath: /run/secrets
    readOnly: true
```

```bash
# Read from file
OAUTH_CLIENT_SECRET_FILE=/run/secrets/oauth-secret
```

#### Never Log Secrets

Automatic redaction:

```python
# Logs show: "Authorization: Bearer [REDACTED]"
logger.info(f"Request headers: {redact_sensitive(headers)}")
```

## Security Best Practices

### Development

- [ ] Use HTTPS in development (self-signed certs OK)
- [ ] Never commit secrets to git
- [ ] Use environment variables for config
- [ ] Enable debug logging only in development
- [ ] Test with security scanners

### Staging

- [ ] Use production-like security
- [ ] Real OAuth provider
- [ ] Valid TLS certificates
- [ ] Security scanning in CI/CD
- [ ] Penetration testing

### Production

- [ ] HTTPS with valid certificates
- [ ] OAuth2 authentication enabled
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] Secrets in secret manager
- [ ] Audit logging enabled
- [ ] Security monitoring active
- [ ] Regular security updates
- [ ] Incident response plan

## Key Security Benefits

### Centralized MCP Management

One of the primary security advantages of the Enterprise MCP Bridge is **centralized credential and OAuth app management**:

✅ **No Individual OAuth Apps** - Users don't need to create their own GitHub/M365/Atlassian OAuth applications  
✅ **Centralized Key Management** - API keys and secrets managed in one secure location  
✅ **Consistent Security Policy** - Apply security controls uniformly across all users  
✅ **Simplified Auditing** - Single point for monitoring and compliance  
✅ **Reduced Attack Surface** - Fewer OAuth apps means fewer potential compromises  

Instead of each user managing credentials:
```
❌ Without Bridge:
User A → Own GitHub OAuth app → Manage secrets
User B → Own M365 OAuth app → Manage secrets
User C → Own Atlassian OAuth app → Manage secrets
(Multiple credential stores, inconsistent policies)

✅ With Bridge:
All Users → Bridge → Centralized OAuth → Token exchange
(Single credential store, consistent policies, easier rotation)
```

## Common Vulnerabilities

### Prevented by Design

⚠️ **SQL Injection** - Bridge doesn't use SQL databases directly (MCP servers may)  
⚠️ **XSS** - Mitigated in REST API; UI generation feature requires review  
✅ **CSRF** - Stateless API with token auth  
⚠️ **Command Injection** - Mitigated via parameterized commands (MCP servers may vary)  
✅ **Path Traversal** - Validated paths only  

### Require Configuration

⚠️ **Weak Authentication** - Configure OAuth2  
⚠️ **Unencrypted Transport** - Enable HTTPS  
⚠️ **Token Leakage** - Use secret management  
⚠️ **DoS Attacks** - Implement infrastructure-level protection  
⚠️ **XSS in UI Generation** - Review generated web applications (see [UI Generation Security](#ui-generation-security))

### Ongoing Monitoring

🔍 **Dependency Vulnerabilities** - Regular updates  
🔍 **Configuration Drift** - Infrastructure as code  
🔍 **Access Pattern Anomalies** - Security monitoring  

## UI Generation Security

The bridge includes an AI-powered UI generation feature that creates web applications from natural language prompts. **Security considerations:**

### XSS Risks

Generated HTML applications may contain:
- User input rendered without escaping
- Dynamic content from MCP tool responses
- JavaScript code generated by LLM

**Mitigation strategies:**
1. **Review Generated Code** - Always review AI-generated HTML before deployment
2. **Content Security Policy** - Apply strict CSP headers
3. **Input Sanitization** - Ensure generated code properly escapes user input
4. **Limited Scope** - Deploy generated UIs in isolated environments

### Generated Code Review

Before deploying generated applications:

```bash
# Review the generated HTML
curl http://localhost:8000/apps/{app_id}/html

# Check for:
# - innerHTML usage without sanitization
# - eval() or Function() calls
# - User input concatenated into DOM
# - Missing CSRF protection
```

### Isolation

Generated applications should:
- Run in separate subdomains with different origins
- Have independent CSP policies
- Use SameSite cookies
- Implement proper authentication

See [UI Generation Documentation](../how-to/ui-generation.md) for details.

## Security Headers

Recommended HTTP security headers:

```python
# Add to responses
headers = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
```

## Audit Logging

Security-relevant events logged:

```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "event": "session_created",
  "user_id": "user-123",
  "session_id": "session-456",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "success": true
}
```

**Logged Events:**
- Authentication attempts (success/failure)
- Session creation/deletion
- Tool invocations
- Permission denials
- Configuration changes
- Error conditions

## Compliance

### GDPR Considerations

- User data minimization
- Privacy by design
- Audit logging

### SOC 2 Controls

- Access control
- Encryption in transit and at rest
- Audit logging
- Change management
- Incident response

## Security Checklist

### Before Production

- [ ] HTTPS configured
- [ ] OAuth2 enabled
- [ ] Secrets in secret manager
- [ ] CORS restricted
- [ ] Security headers set
- [ ] Security headers set
- [ ] Audit logging enabled
- [ ] Dependency scanning
- [ ] Penetration testing
- [ ] Incident response plan

### Regular Maintenance

- [ ] Update dependencies monthly
- [ ] Review audit logs weekly
- [ ] Security patches within 48 hours
- [ ] Quarterly security assessment
- [ ] Annual penetration test

## Reporting Security Issues

**Do NOT** open public issues for security vulnerabilities.

Instead:
1. Email: security@inxm.ai
2. Include detailed description
3. Steps to reproduce
4. Potential impact
5. Suggested fix (if any)

We aim to respond within 48 hours.

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OAuth 2.0 Security Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)
- [JWT Security Best Practices](https://tools.ietf.org/html/rfc8725)

## Summary

The Enterprise MCP Bridge security model:

✅ Multiple layers of defense  
✅ Industry-standard authentication  
✅ Fine-grained access control  
✅ Secure token management  
✅ Process isolation  
✅ Comprehensive audit logging  

## Next Steps

- [Architecture Overview](architecture.md)
- [Token Exchange Flow](token-exchange.md)
- [Configuration Reference](../reference/configuration.md)
