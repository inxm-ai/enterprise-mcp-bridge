import pytest

from app.utils_tests.token_retriever_mock import (
    DummyTokenRetrieverFactory,
)


@pytest.fixture
def mock_token_retriever_factory(monkeypatch):
    """Provide a mocked TokenRetrieverFactory for app-layer tests."""
    factory = DummyTokenRetrieverFactory("test_access_token")

    from app.oauth import token_exchange
    from app.mcp_server import server_params
    from app.oauth import decorator

    monkeypatch.setattr(token_exchange, "TokenRetrieverFactory", lambda: factory)
    monkeypatch.setattr(server_params, "TokenRetrieverFactory", lambda: factory)
    monkeypatch.setattr(decorator, "TokenRetrieverFactory", lambda: factory)

    return factory
