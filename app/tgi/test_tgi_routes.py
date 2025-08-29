import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

import os
from app.tgi.routes import router as tgi_router, UserLoggedOutException
from fastapi import HTTPException


# Ensure TGI_URL is set for all tests except the one that checks missing value


def pytest_configure(config=None):
    os.environ["TGI_URL"] = "https://api.test-llm.com/v1"
    os.environ["TGI_TOKEN"] = "test-token-123"


@pytest.fixture(autouse=True)
def _reset_env():
    pytest_configure()


test_app = FastAPI()
test_app.include_router(tgi_router)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_session_manager():
    """Mock session manager."""
    with patch("app.tgi.routes.sessions") as mock_sessions:
        yield mock_sessions


@pytest.fixture
def mock_mcp_session_context():
    """Mock MCP session context."""
    with patch("app.tgi.routes.mcp_session_context") as mock_context:
        mock_session = Mock()
        mock_session.list_prompts = AsyncMock()
        mock_session.list_tools = AsyncMock()
        mock_session.call_prompt = AsyncMock()
        mock_session.call_tool = AsyncMock()

        async def mock_context_func(*args, **kwargs):
            return mock_session

        mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

        yield mock_session


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "model": "test-model",
        "stream": False,
    }


class TestTGIRoutesEdgyCases:
    """More edge casy tests."""

    def test_chat_completions_missing_tgi_url(
        self, client, sample_chat_request, monkeypatch
    ):
        """Test that 400 is returned if TGI_URL is not set."""
        # Unset TGI_URL
        if "TGI_URL" in os.environ:
            del os.environ["TGI_URL"]
        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={"X-Auth-Request-Access-Token": "test-token"},
        )
        assert response.status_code == 400
        assert "TGI_URL not configured" in response.text

    def test_chat_completions_user_logged_out(
        self, client, sample_chat_request, monkeypatch
    ):
        """Test UserLoggedOutException handling (401)."""

        def mock_mcp_session_context(*args, **kwargs):
            raise UserLoggedOutException("Logged out!")

        monkeypatch.setattr(
            "app.tgi.routes.mcp_session_context", mock_mcp_session_context
        )
        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={"X-Auth-Request-Access-Token": "test-token"},
        )
        assert response.status_code == 401
        assert "Logged out" in response.json()["detail"]

    def test_chat_completions_internal_error(
        self, client, sample_chat_request, monkeypatch
    ):
        """Test generic Exception handling (500)."""

        def mock_mcp_session_context(*args, **kwargs):
            raise Exception("Something went wrong")

        monkeypatch.setattr(
            "app.tgi.routes.mcp_session_context", mock_mcp_session_context
        )
        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={"X-Auth-Request-Access-Token": "test-token"},
        )
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_chat_completions_exception_group(
        self, client, sample_chat_request, monkeypatch
    ):
        """Test find_exception_in_exception_groups returns HTTPException."""

        class CustomException(Exception):
            pass

        def mock_find_exception_in_exception_groups(e, exc_type):
            return HTTPException(status_code=418, detail="I'm a teapot")

        monkeypatch.setattr(
            "app.tgi.routes.find_exception_in_exception_groups",
            mock_find_exception_in_exception_groups,
        )

        def mock_mcp_session_context(*args, **kwargs):
            raise CustomException("group error")

        monkeypatch.setattr(
            "app.tgi.routes.mcp_session_context", mock_mcp_session_context
        )
        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={"X-Auth-Request-Access-Token": "test-token"},
        )
        assert response.status_code == 418
        assert "teapot" in response.json()["detail"]

    def test_streaming_prompt_not_found(self, client, sample_chat_request, monkeypatch):
        """Test streaming with non-existent prompt yields error chunk and [DONE]."""
        sample_chat_request["stream"] = True

        # Mock the proxied TGI service instead of the old service
        mock_service = AsyncMock()

        async def mock_chat_completion(*args, **kwargs):
            yield 'data: {"id":"test","choices":[{"delta":{"content":"Error: Prompt \'doesnotexist\' not found"},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        mock_service.chat_completion = mock_chat_completion

        monkeypatch.setattr("app.tgi.routes.tgi_service", mock_service)

        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            params={"prompt": "doesnotexist"},
            headers={
                "X-Auth-Request-Access-Token": "test-token",
                "Accept": "text/event-stream",
            },
        )
        assert response.status_code == 200
        assert (
            "Error: Prompt 'doesnotexist' not found" in response.text
            or "Error:" in response.text
        )
        assert "[DONE]" in response.text

    def test_streaming_with_tool_call(self, client, sample_chat_request, monkeypatch):
        """Test streaming with available tools yields tool call and result chunks."""
        sample_chat_request["stream"] = True

        # Mock the proxied TGI service
        mock_service = AsyncMock()

        async def mock_chat_completion(*args, **kwargs):
            yield 'data: {"id":"test","choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"demo_tool","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","choices":[{"delta":{"content":"Tool executed successfully"},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        mock_service.chat_completion = mock_chat_completion

        monkeypatch.setattr("app.tgi.routes.tgi_service", mock_service)

        response = client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={
                "X-Auth-Request-Access-Token": "test-token",
                "Accept": "text/event-stream",
            },
        )
        assert response.status_code == 200
        assert "demo_tool" in response.text
        assert "[DONE]" in response.text

    def test_session_extraction_header_vs_cookie(
        self, client, sample_chat_request, monkeypatch
    ):
        """Test session extraction from header and cookie."""
        # Patch session_id to check which value is used
        called = {}

        def mock_session_id(val, token):
            called["val"] = val
            called["token"] = token
            return "session123"

        monkeypatch.setattr("app.tgi.routes.session_id", mock_session_id)
        client.post(
            "/tgi/v1/chat/completions",
            json=sample_chat_request,
            headers={
                "X-Auth-Request-Access-Token": "test-token",
                "x-inxm-mcp-session": "header-session",
            },
            cookies={"x-inxm-mcp-session": "cookie-session"},
        )
        assert called["val"] == "header-session"
        assert called["token"] == "test-token"

    def test_chat_completions_missing_model(self, client, sample_chat_request):
        """Test request missing model field."""
        req = dict(sample_chat_request)
        req.pop("model")

        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mcp-bridge",  # Default model when none provided
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post(
                "/tgi/v1/chat/completions",
                json=req,
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )
            # this is kinda weird atm, but we return a 200 for missing model
            #   any real backend would probably fail, so should we eventually
            #   but for now just return and expect a 200
            assert response.status_code in [
                200
            ]  # for the future prob something like [422, 400]
            if response.status_code == 200:
                data = response.json()
                assert data["model"] == "mcp-bridge"

    def test_chat_completions_missing_messages(self, client):
        """Test request missing messages field."""
        req = {"model": "test-model"}
        response = client.post(
            "/tgi/v1/chat/completions",
            json=req,
            headers={"X-Auth-Request-Access-Token": "test-token"},
        )
        assert response.status_code in [422, 400]

    def test_chat_completions_accept_header_variants(self, client, sample_chat_request):
        """Test with different Accept header values."""
        with patch("app.tgi.routes.tgi_service") as mock_service:
            # Mock response for non-streaming
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

            # Async generator for streaming
            async def streaming_generator():
                yield "data: " + json.dumps(
                    {
                        "id": "chatcmpl-test123",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hello!"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ) + "\n\n"
                yield "data: [DONE]\n\n"

            for accept in ["application/json", "text/event-stream", "*/*"]:
                # Adjust the request for streaming when Accept header indicates streaming
                req = dict(sample_chat_request)
                if accept == "text/event-stream":
                    req["stream"] = True

                    # For streaming, set the service to an async generator function (not awaited)
                    async def mock_stream(*args, **kwargs):
                        async for chunk in streaming_generator():
                            yield chunk

                    mock_service.chat_completion = mock_stream
                else:
                    # For non-streaming, return a JSON dict via an awaited async function
                    mock_service.chat_completion = AsyncMock(return_value=mock_response)

                response = client.post(
                    "/tgi/v1/chat/completions",
                    json=req,
                    headers={
                        "X-Auth-Request-Access-Token": "test-token",
                        "Accept": accept,
                    },
                )
                assert response.status_code in [200, 400, 422]


class TestTGIRoutes:
    """Test cases for TGI routes."""

    def test_chat_completions_non_streaming(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test non-streaming chat completions endpoint."""
        # Mock the proxied TGI service response
        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "id" in data
            assert data["object"] == "chat.completion"
            assert "created" in data
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "usage" in data

            choice = data["choices"][0]
            assert choice["index"] == 0
            assert "message" in choice
            assert "finish_reason" in choice

            message = choice["message"]
            assert message["role"] == "assistant"
            assert "content" in message

    def test_chat_completions_streaming(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test streaming chat completions endpoint."""
        # Enable streaming
        sample_chat_request["stream"] = True

        # Mock the proxied TGI service streaming response
        with patch("app.tgi.routes.tgi_service") as mock_service:

            async def mock_stream(*args, **kwargs):
                yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            mock_service.chat_completion = mock_stream

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                headers={
                    "X-Auth-Request-Access-Token": "test-token",
                    "Accept": "text/event-stream",
                },
            )

            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            content = response.text
            assert "data: " in content
            assert "[DONE]" in content

            # Extract first data chunk
            lines = content.split("\n")
            data_lines = [
                line
                for line in lines
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]

            if data_lines:
                first_chunk_json = data_lines[0][6:]  # Remove "data: " prefix
                first_chunk = json.loads(first_chunk_json)

                assert "id" in first_chunk
                assert first_chunk["object"] == "chat.completion.chunk"
                assert "choices" in first_chunk

    def test_chat_completions_with_specific_prompt(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test chat completions with specific prompt parameter."""
        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from test prompt!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "total_tokens": 23,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                params={"prompt": "test-prompt"},
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "Hello from test prompt!" in data["choices"][0]["message"]["content"]

    def test_chat_completions_prompt_not_found(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test chat completions with non-existent prompt."""
        with patch("app.tgi.routes.tgi_service") as mock_service:
            # Mock the service to raise an HTTPException for non-existent prompt
            from fastapi import HTTPException

            mock_service.chat_completion = AsyncMock(
                side_effect=HTTPException(status_code=404, detail="Prompt not found")
            )

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                params={"prompt": "nonexistent-prompt"},
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_chat_completions_with_tools(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test chat completions with tools provided."""
        sample_chat_request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        with patch("app.tgi.routes.tgi_service") as mock_service:
            # Mock response with tool call
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            assert response.status_code == 200

            # The response should potentially include tool calls
            data = response.json()
            choice = data["choices"][0]

            # Check if this is a tool call response or regular response
            if choice["finish_reason"] == "tool_calls":
                assert "tool_calls" in choice["message"]
                assert choice["message"]["content"] is None
            else:
                assert "content" in choice["message"]

    def test_chat_completions_missing_auth_token(self, client, sample_chat_request):
        """Test chat completions without authentication token."""
        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post("/tgi/v1/chat/completions", json=sample_chat_request)

            # Should handle gracefully
            assert response.status_code in [200, 401, 422]

    def test_chat_completions_invalid_request_body(self, client):
        """Test chat completions with invalid request body."""
        invalid_request = {
            "messages": [],  # Empty messages array
            "model": "test-model",
        }

        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_service.chat_completion = AsyncMock(return_value={})

            response = client.post(
                "/tgi/v1/chat/completions",
                json=invalid_request,
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            # Should handle gracefully, just no 500
            assert response.status_code in [200, 400, 422]

    def test_chat_completions_with_group_parameter(
        self,
        client,
        mock_session_manager,
        mock_mcp_session_context,
        sample_chat_request,
    ):
        """Test chat completions with group parameter."""
        with patch("app.tgi.routes.tgi_service") as mock_service:
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello with group!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            mock_service.chat_completion = AsyncMock(return_value=mock_response)

            response = client.post(
                "/tgi/v1/chat/completions",
                json=sample_chat_request,
                params={"group": "test-group"},
                headers={"X-Auth-Request-Access-Token": "test-token"},
            )

            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
