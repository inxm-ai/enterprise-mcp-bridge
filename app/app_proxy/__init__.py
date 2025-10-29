"""
Generic Web Application Reverse Proxy

A production-ready reverse proxy implementation that handles:
- HTTP/HTTPS proxying with all methods
- Intelligent URL rewriting in HTML, CSS, JavaScript, and JSON
- Cookie and redirect handling
- Proper header forwarding (X-Forwarded-*)
- Streaming responses for efficiency
- Compression support
- OpenTelemetry tracing
"""

from .route import router, forward_to_target

__all__ = ["router", "forward_to_target"]
