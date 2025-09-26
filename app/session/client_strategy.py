import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientMetadata, OAuthToken, OAuthClientInformationFull

from app.mcp_server.server_params import get_server_params
from app.oauth.token_exchange import TokenRetrieverFactory, UserLoggedOutException
from app.utils import mask_token
from app.utils.exception_logging import log_exception_with_details
from app.vars import (
    MCP_REMOTE_BEARER_TOKEN,
    MCP_REMOTE_CLIENT_ID,
    MCP_REMOTE_CLIENT_SECRET,
    MCP_REMOTE_REDIRECT_URI,
    MCP_REMOTE_SCOPE,
    MCP_REMOTE_SERVER,
)

logger = logging.getLogger("uvicorn.error")


class MCPClientStrategy(ABC):
    """Strategy interface for establishing MCP client sessions."""

    @asynccontextmanager
    @abstractmethod
    async def session(self) -> AsyncIterator[ClientSession]:
        """Yield an initialized MCP client session."""
        yield  # pragma: no cover


class LocalMCPClientStrategy(MCPClientStrategy):
    """Run MCP server as a local process via stdio."""

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class _EphemeralTokenStorage(TokenStorage):
    """In-memory token storage implementing the TokenStorage protocol."""

    def __init__(
        self,
        token: Optional[OAuthToken] = None,
        client_info: Optional[OAuthClientInformationFull] = None,
    ) -> None:
        self._token = token
        self._client_info = client_info

    async def get_tokens(self) -> Optional[OAuthToken]:
        return self._token

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._token = tokens

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class RemoteMCPClientStrategy(MCPClientStrategy):
    """Connect to a remote MCP server over HTTP."""

    def __init__(
        self,
        url: str,
        *,
        access_token: Optional[str],
        requested_group: Optional[str],
        anon: bool,
    ) -> None:
        self.url = url
        self.access_token = access_token
        self.requested_group = requested_group
        self.anon = anon
        self.headers: dict[str, str] = {}
        self._auth_provider: Optional[OAuthClientProvider] = None
        self._token_storage: Optional[_EphemeralTokenStorage] = None
        self._client_metadata = self._build_client_metadata()
        self._prepare_auth()

    def _build_client_metadata(self) -> OAuthClientMetadata:
        scope = MCP_REMOTE_SCOPE or None
        redirect_uri = MCP_REMOTE_REDIRECT_URI
        if not redirect_uri:
            parsed = urlparse(self.url)
            if parsed.scheme and parsed.netloc:
                redirect_uri = f"{parsed.scheme}://{parsed.netloc}/oauth/callback"
            else:
                redirect_uri = "https://localhost/unused-callback"
        return OAuthClientMetadata(
            redirect_uris=[redirect_uri],
            scope=scope,
        )

    def _initial_client_info(self) -> Optional[OAuthClientInformationFull]:
        if not MCP_REMOTE_CLIENT_ID:
            return None
        return OAuthClientInformationFull(
            client_id=MCP_REMOTE_CLIENT_ID,
            client_secret=MCP_REMOTE_CLIENT_SECRET or None,
            redirect_uris=self._client_metadata.redirect_uris,
            scope=self._client_metadata.scope,
        )

    def _prepare_auth(self) -> None:
        if self.anon:
            logger.info(
                "[RemoteMCP] Anonymous remote session requested; skipping OAuth setup"
            )
            self._prepare_fallback_headers()
            return

        token_value: Optional[str] = None
        token_result: Optional[dict[str, object]] = None

        if self.access_token:
            try:
                retriever = TokenRetrieverFactory().get()
                token_result = retriever.retrieve_token(self.access_token)
                token_value = (
                    token_result.get("access_token") if token_result else None  # type: ignore[attr-defined]
                )
                if token_value:
                    logger.info(
                        mask_token(
                            "[RemoteMCP] Retrieved provider token via token exchange",
                            token_value,
                        )
                    )
            except UserLoggedOutException:
                raise
            except Exception as exc:  # pragma: no cover - safety net
                log_exception_with_details(logger, "[RemoteMCP]", exc)

        if not token_value:
            if MCP_REMOTE_BEARER_TOKEN:
                token_value = MCP_REMOTE_BEARER_TOKEN
                logger.info("[RemoteMCP] Falling back to MCP_REMOTE_BEARER_TOKEN")
            elif self.access_token:
                token_value = self.access_token
                logger.info("[RemoteMCP] Falling back to incoming access token")

        if token_value:
            oauth_token = OAuthToken(
                access_token=token_value,
                token_type=(
                    token_result.get("token_type") if token_result else "Bearer"
                ),
                expires_in=(token_result.get("expires_in") if token_result else None),
                scope=(token_result.get("scope") if token_result else None),
                refresh_token=(
                    token_result.get("refresh_token") if token_result else None
                ),
            )
            client_info = self._initial_client_info()
            self._token_storage = _EphemeralTokenStorage(oauth_token, client_info)
            self._auth_provider = OAuthClientProvider(
                server_url=self.url,
                client_metadata=self._client_metadata,
                storage=self._token_storage,
                redirect_handler=self._redirect_handler,
                callback_handler=self._callback_handler,
            )
        else:
            self._prepare_fallback_headers()

        self._add_env_headers()

    def _add_env_headers(self) -> None:
        for key, value in os.environ.items():
            if key.startswith("MCP_REMOTE_HEADER_"):
                header_name = key[len("MCP_REMOTE_HEADER_"):].replace("_", "-")
                if header_name and value:
                    self.headers[header_name] = value
                    logger.info(
                        f"[RemoteMCP] Adding custom header from environment: {header_name}"
                    )

    def _prepare_fallback_headers(self) -> None:
        if MCP_REMOTE_BEARER_TOKEN and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {MCP_REMOTE_BEARER_TOKEN}"
            logger.info(
                "[RemoteMCP] Using MCP_REMOTE_BEARER_TOKEN for Authorization header"
            )
        elif self.access_token and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.access_token}"
            logger.info(
                "[RemoteMCP] Using incoming access token for Authorization header"
            )

    @staticmethod
    async def _redirect_handler(url: str) -> None:  # pragma: no cover - defensive
        logger.error(
            "[RemoteMCP] OAuth redirect requested (%s) but interactive flows are unsupported",
            url,
        )
        raise RuntimeError("Interactive OAuth redirect unsupported in server mode")

    @staticmethod
    async def _callback_handler() -> (
        tuple[str, Optional[str]]
    ):  # pragma: no cover - defensive
        raise RuntimeError("Interactive OAuth callback unsupported in server mode")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        headers = self.headers or None
        async with streamablehttp_client(
            self.url,
            headers=headers,
            auth=self._auth_provider,
        ) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if get_session_id:
                    setattr(session, "get_remote_session_id", get_session_id)
                yield session


def build_mcp_client_strategy(
    *,
    access_token: Optional[str],
    requested_group: Optional[str],
    anon: bool = False,
) -> MCPClientStrategy:
    remote_server = (MCP_REMOTE_SERVER or "").strip()
    if remote_server:
        mcp_command = os.environ.get("MCP_SERVER_COMMAND", "").strip()
        if mcp_command:
            logger.error(
                "[ClientStrategy] MCP_REMOTE_SERVER and MCP_SERVER_COMMAND are both set; cannot use both"
            )
            raise ValueError(
                "MCP_REMOTE_SERVER cannot be used together with MCP_SERVER_COMMAND"
            )
        logger.info(f"Using remote MCP server at {remote_server}")
        return RemoteMCPClientStrategy(
            remote_server,
            access_token=access_token,
            requested_group=requested_group,
            anon=anon,
        )

    server_params = get_server_params(
        access_token=access_token,
        requested_group=requested_group,
        anon=anon,
    )
    return LocalMCPClientStrategy(server_params)
