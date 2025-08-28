import logging
from typing import Optional, Dict
from contextlib import contextmanager

from opentelemetry.trace import Tracer

from app.utils import mask_token

logger = logging.getLogger("uvicorn.error")


@contextmanager
def traced_request(
    tracer: Tracer,
    operation: str,
    session_value: Optional[str],
    group: Optional[str],
    start_message: str,
    extra_attrs: Optional[Dict] = None,
):
    """Context manager to create a span, set common attributes, and log a start message."""
    with tracer.start_as_current_span(operation) as span:
        if session_value:
            span.set_attribute("session.id", session_value)
        if group:
            span.set_attribute("session.group", group)
        if extra_attrs:
            for k, v in extra_attrs.items():
                span.set_attribute(k, v)
        logger.info(mask_token(start_message, session_value))
        yield span
