# Security Implementation Guide

This document outlines the comprehensive security measures implemented in the Podcast Ad Remover application.

## 🔒 Security Features Implemented

### 1. HTTP Security Headers

All HTTP responses include the following security headers via `SecurityHeadersMiddleware`:

| Header | Purpose | Configuration |
|--------|---------|---------------|
| **Content-Security-Policy** | Prevents XSS and injection attacks | Self-hosted content with inline scripts/styles |
| **Strict-Transport-Security** | Enforces HTTPS connections | max-age=31536000, includeSubDomains, preload |
| **X-Frame-Options** | Prevents clickjacking | DENY |
| **X-Content-Type-Options** | Prevents MIME-sniffing | nosniff |
| **Referrer-Policy** | Controls referrer information | strict-origin-when-cross-origin |
| **Permissions-Policy** | Restricts browser features | Blocks camera, microphone, geolocation, etc. |
| **X-XSS-Protection** | Legacy XSS protection | 1; mode=block |

**Implementation:** `app/web/security_headers.py`

### 2. Login Rate Limiting

Brute force protection for authentication endpoints:

- **Max Attempts:** 5 failed logins per IP
- **Time Window:** 15 minutes
- **Lockout Duration:** 15 minutes
- **Memory Management:** Automatic cleanup every 5 minutes

**Features:**
- Tracks failed attempts by IP address
- Locks IP after threshold exceeded
- Clear user feedback on lockout
- Automatic unlock after timeout
- Server-side logging of all lockout events

**Implementation:** `app/web/rate_limiter.py`

### 3. Secure Session Management

Session cookies are configured with the following security attributes:

```python
SessionMiddleware(
    secret_key=settings.SESSION_SECRET_KEY,
    max_age=30 * 24 * 60 * 60,  # 30 days
    session_cookie="session",
    same_site="lax",  # Prevents CSRF
    https_only=settings.COOKIE_SECURE
)
```

**Security Attributes:**
- ✅ **HttpOnly:** Automatically set by Starlette (prevents JavaScript access)
- ✅ **Secure:** Enabled in production (HTTPS-only)
- ✅ **SameSite=Lax:** Prevents CSRF attacks
- ✅ **Max-Age:** 30-day expiration

### 4. Custom Error Handlers

Prevents information disclosure through error messages:

**Error Types Handled:**
- 400 Bad Request
- 401 Unauthorized
- 403 Forbidden
- 404 Not Found
- 500 Internal Server Error
- All unhandled exceptions

**Protection:**
- ❌ No stack traces exposed to users
- ❌ No file paths or system details revealed
- ❌ No framework/library versions disclosed
- ✅ Generic user-friendly error messages
- ✅ Detailed logging server-side only
- ✅ Custom error pages with helpful actions

**Implementation:** 
- `app/web/error_handlers.py`
- `app/web/templates/error.html`

### 5. Production Mode Configuration

Environment-based security settings:

```bash
# .env file
ENVIRONMENT=production  # or "development"
```

**Production Mode Features:**
- Debug mode disabled
- API documentation hidden
- HTTPS-only session cookies
- Detailed error logging (server-side only)

**Development Mode Features:**
- Debug mode enabled
- API docs available at `/api/docs`
- HTTP session cookies allowed
- More verbose error messages

## 🔐 Security Best Practices

### Authentication
- ✅ Password hashing using bcrypt
- ✅ Rate limiting on login attempts
- ✅ Session-based authentication
- ✅ Secure session cookie attributes
- ✅ Login attempt logging with IP tracking

### Data Protection
- ✅ Session secrets via environment variables
- ✅ API keys stored in environment variables
- ✅ No sensitive data in logs (user-facing)
- ✅ Generic error messages prevent reconnaissance

### Attack Prevention
- ✅ **XSS:** Content-Security-Policy header
- ✅ **Clickjacking:** X-Frame-Options header
- ✅ **CSRF:** SameSite cookie attribute
- ✅ **Brute Force:** Login rate limiting
- ✅ **MIME Sniffing:** X-Content-Type-Options header
- ✅ **Protocol Downgrade:** HSTS header
- ✅ **Information Disclosure:** Custom error pages

## 📋 Configuration Checklist

### Required Environment Variables

```bash
# Security (REQUIRED - change defaults!)
SESSION_SECRET_KEY=your-random-secret-key-here
ENVIRONMENT=production

# Optional
LOG_LEVEL=INFO
```

### Production Deployment

1. **Set Environment to Production**
   ```bash
   export ENVIRONMENT=production
   ```

2. **Generate Strong Session Secret**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Set this as `SESSION_SECRET_KEY` before enabling dashboard or feed authentication. The app warns in System Settings and refuses to enable those modes while the default session secret is still configured.

3. **Configure HTTPS**
   - Use reverse proxy (nginx/traefik)
   - Enable SSL/TLS certificates
   - Set `COOKIE_SECURE=true` when users access the app through HTTPS
   - Set `TRUST_PROXY_HEADERS=true` only if the reverse proxy strips incoming client-supplied `CF-Connecting-IP`, `X-Forwarded-For`, and `X-Real-IP` headers before forwarding requests
   - Set `BASE_URL` or the System Settings public application URL to the browser-facing origin used for authenticated management access

4. **Verify Security Headers**
   ```bash
   curl -I https://your-domain.com
   ```

5. **Test Rate Limiting**
   - Attempt 5+ failed logins
   - Verify IP lockout occurs
   - Check server logs for lockout events

## 🛡️ Security Monitoring

### Log Monitoring

Monitor logs for security events:

```bash
# Failed login attempts
grep "Failed login attempt" /data/app.log

# Rate limit lockouts
grep "locked out" /data/app.log

# Error tracking
grep "ERROR" /data/app.log
```

### Security Headers Verification

Use online tools or curl to verify headers:

```bash
# Check all security headers
curl -I https://your-domain.com

# Specific header check
curl -I https://your-domain.com | grep -i "content-security-policy"
```

## 🔄 Maintenance

### Regular Security Tasks

1. **Review Logs:** Weekly check for suspicious activity
2. **Update Dependencies:** Monthly security updates
3. **Session Secret Rotation:** Annually or after breach
4. **Rate Limit Tuning:** Adjust based on legitimate vs malicious traffic

### Incident Response

If a security incident occurs:

1. Check logs for affected IPs/users
2. Review failed login attempts
3. Rotate session secrets if compromised
4. Clear all active sessions
5. Force password resets if needed

## 📚 References

- [OWASP Security Headers](https://owasp.org/www-project-secure-headers/)
- [Content Security Policy Reference](https://content-security-policy.com/)
- [Session Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)

## 🆘 Support

For security issues:
- Review this documentation
- Check application logs
- Test with curl/browser dev tools
- Verify environment configuration
