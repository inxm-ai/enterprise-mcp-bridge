# Ensure tests import modules from this service directory first.
# This avoids import collisions with other `app/*` packages in the monorepo
# and makes `import app.server` and `from app import ...` behave consistently.
import os
import sys
import pytest
from app.utils_tests.token_retriever_mock import (
    DummyTokenRetrieverFactory,
)

SERVICE_ROOT = os.path.dirname(__file__)
APP_DIR = os.path.join(SERVICE_ROOT, "app")

# Prepend both the service root (so `import app.*` works) and the app dir itself
# (so local-style imports like `from oauth.decorator import ...` still work).
for p in (SERVICE_ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def mock_token_retriever_factory(monkeypatch):
    """Mock token retriever factory for testing"""
    factory = DummyTokenRetrieverFactory("test_access_token")

    # Mock the TokenRetrieverFactory class in all modules that use it
    from app.oauth import token_exchange
    from app.mcp_server import server_params
    from app.oauth import decorator

    monkeypatch.setattr(token_exchange, "TokenRetrieverFactory", lambda: factory)
    monkeypatch.setattr(server_params, "TokenRetrieverFactory", lambda: factory)
    monkeypatch.setattr(decorator, "TokenRetrieverFactory", lambda: factory)

    return factory
