from app.utils.mcp_fields import input_schema
from app.multi_server import (
    current_auth_provider,
    current_forward_access_token,
    current_keycloak_provider_alias,
)
from app.vars import AUTH_PROVIDER, KEYCLOAK_PROVIDER_ALIAS
from fastapi import HTTPException
import logging
from typing import Dict, Optional

from app.oauth.token_exchange import TokenRetrieverFactory

logger = logging.getLogger("uvicorn.error")


async def decorate_args_with_oauth_token(
    tools, tool_name, args: Optional[Dict], access_token: Optional[str]
) -> Dict:
    tool_info = next((tool for tool in tools.tools if tool.name == tool_name), None)

    if args is None:
        args = {}
    if access_token and not current_forward_access_token(True):
        # The caller's token (exchanged or not) must never reach this server.
        logger.info(
            f"[Tool-Call] forward_access_token is disabled; no oauth_token "
            f"is injected for tool {tool_name}."
        )
        return args

    oauth_token = None
    if access_token:
        # Keycloak mode without a provider alias passes the token through;
        # every other provider (e.g. user-api-key) must go through its
        # retriever — short-circuiting here would silently forward the
        # Keycloak token instead of the per-user credential.
        auth_provider = current_auth_provider(AUTH_PROVIDER)
        provider_alias = current_keycloak_provider_alias(KEYCLOAK_PROVIDER_ALIAS)
        if auth_provider == "keycloak" and not provider_alias:
            oauth_token = access_token
        else:
            retriever = TokenRetrieverFactory().get()
            token_result = retriever.retrieve_token(access_token)
            if not token_result or "access_token" not in token_result:
                raise ValueError(
                    f"Token retrieval failed with access_token: {access_token}"
                )
            oauth_token = token_result["access_token"]

    # inputSchema {'properties': {'file_name': {}, 'content_type': {}, 'file_content': {}, 'oauth_token': {'title': 'Oauth Token', 'type': 'string'}}, 'required': ['file_name', 'content_type', 'file_content', 'oauth_token'], 'title': 'upload_file_to_onedriveArguments', 'type': 'object'}
    schema = input_schema(tool_info) if tool_info else None
    if schema:
        # inputSchema might be a dict or an object with 'properties'
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
        else:
            properties = getattr(schema, "properties", {})
        if "oauth_token" in properties:
            if oauth_token:
                args["oauth_token"] = oauth_token
                logger.info(
                    f"[Tool-Call] Tool {tool_name} will be called with oauth_token."
                )
            else:
                logger.warning(
                    f"[Tool-Call] Tool {tool_name} requires oauth_token but none provided."
                )
                raise HTTPException(
                    status_code=401,
                    detail="Tool requires oauth_token but none provided.",
                )
        else:
            logger.info(f"[Tool-Call] Tool {tool_name} does not require oauth_token.")
    else:
        logger.info(f"[Tool-Call] Tool {tool_name} has no inputSchema or tool_info.")
    return args
