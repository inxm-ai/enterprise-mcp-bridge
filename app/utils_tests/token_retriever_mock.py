import pytest


class MockTokenRetriever:
    def retrieve_token(self, access_token):
        return {"access_token": access_token}


@pytest.fixture
def mock_token_retriever_factory(monkeypatch):
    def mock_get(self):
        return MockTokenRetriever()

    monkeypatch.setattr("app.oauth.token_exchange.TokenRetrieverFactory.get", mock_get)
