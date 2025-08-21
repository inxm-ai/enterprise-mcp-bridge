from fastapi import HTTPException
import logging
from typing import Dict, Optional

from app.oauth.token_exchange import TokenRetrieverFactory


logger = logging.getLogger("uvicorn.error")


async def decorate_args_with_oauth_token(
    tools, tool_name, args: Optional[Dict], access_token: Optional[str]
) -> Dict:
    tool_info = next((tool for tool in tools.tools if tool.name == tool_name), None)

    oauth_token = None
    if access_token:
        retriever = TokenRetrieverFactory().get()
        token_result = retriever.retrieve_token(access_token)
        if not token_result or "access_token" not in token_result:
            raise ValueError(
                f"Token retrieval failed with access_token: {access_token}"
            )
        oauth_token = token_result["access_token"]

    if args is None:
        args = {}
    # inputSchema {'properties': {'file_name': {}, 'content_type': {}, 'file_content': {}, 'oauth_token': {'title': 'Oauth Token', 'type': 'string'}}, 'required': ['file_name', 'content_type', 'file_content', 'oauth_token'], 'title': 'upload_file_to_onedriveArguments', 'type': 'object'}
    if tool_info and hasattr(tool_info, "inputSchema") and tool_info.inputSchema:
        # inputSchema might be a dict or an object with 'properties'
        if isinstance(tool_info.inputSchema, dict):
            properties = tool_info.inputSchema.get("properties", {})
        else:
            properties = getattr(tool_info.inputSchema, "properties", {})
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
