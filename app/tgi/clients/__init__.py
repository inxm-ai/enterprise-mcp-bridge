"""TGI clients package."""

# Re-export llm_client module
from app.tgi.clients import llm_client

__all__ = [
    "llm_client",
]
