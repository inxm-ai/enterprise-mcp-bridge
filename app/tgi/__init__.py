"""TGI package for LLM integration."""

# Re-export submodules for backward compatibility
from app.tgi.clients import llm_client
from app.tgi.models import model_formats
from app.tgi.services import service

# Re-export key classes for direct access
from app.tgi.clients.llm_client import LLMClient
from app.tgi.services.service import TGIService

__all__ = [
    "llm_client",
    "model_formats",
    "service",
    "LLMClient",
    "TGIService",
]
