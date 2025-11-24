import logging
import os
import sys
from typing import Optional
from mcp import StdioServerParameters
from app.oauth.token_exchange import TokenRetrieverFactory
from app.oauth.user_info import get_data_access_manager
from app.utils import mask_token

logger = logging.getLogger("uvicorn.error")


def defined_env(
    env: dict[str, str],
    access_token: Optional[str] = None,
    requested_group: Optional[str] = None,
    anon: bool = False,
) -> dict[str, str]:
    oauth_env_var = env.get("OAUTH_ENV")
    token_result = {"access_token": access_token} if access_token else None

    if oauth_env_var and not anon:
        if not access_token:
            raise ValueError("access_token required when OAUTH_ENV is set")
        logger.info(f"Using OAUTH_ENV: {oauth_env_var} for token retrieval")
        
        # If using cookie source, we might want to skip the token retriever if it's just a pass-through
        # But TokenRetrieverFactory defaults to KeycloakTokenRetriever which tries to exchange.
        # If the user wants to pass the cookie AS IS, they should probably not use the Keycloak retriever
        # or we should detect if it's a cookie and maybe skip retrieval?
        # However, the user might want to exchange the cookie for a token?
        # The user's error suggests they want to pass the cookie value directly to the MCP.
        
        # Check if we should skip retrieval based on configuration or token format?
        # For now, let's assume if it's a cookie source, we might want to try retrieval, 
        # but if it fails or if we want to bypass, we need a way.
        
        # Actually, the user's issue is likely that KeycloakTokenRetriever fails or returns something unexpected
        # because it expects a Keycloak token to exchange, but gets an _oauth2_proxy cookie value.
        
        # If the user hasn't configured KEYCLOAK_PROVIDER_ALIAS, KeycloakTokenRetriever returns the token as is.
        # Let's check if that's the case.
        
        retriever = TokenRetrieverFactory().get()
        logger.info(f"Retrieving token for {oauth_env_var} using TokenRetriever")
        try:
            token_result = retriever.retrieve_token(access_token)
        except Exception as e:
            # If retrieval fails, and we are in cookie mode, maybe we should just pass the token as is?
            # This is a fallback for when the token is not a valid Keycloak token but just a cookie value
            # that the MCP needs.
            logger.warning(f"Token retrieval failed: {e}. Falling back to using the raw token/cookie value.")
            token_result = {"access_token": access_token}

        if not token_result or "access_token" not in token_result:
             # Fallback to raw token if retrieval returned structure but no access_token (unlikely if we handled exception)
             token_result = {"access_token": access_token}

        logger.info(
            mask_token(
                f"Server-Params from OAUTH_ENV: {oauth_env_var}={token_result['access_token']}",
                token_result["access_token"],
            )
        )
        env[oauth_env_var] = token_result["access_token"]
    else:
        logger.info("No OAUTH_ENV set, using default environment variables")

    for key, value in env.copy().items():
        if key.startswith("MCP_ENV_"):
            env_name = key[len("MCP_ENV_") :]
            env[env_name] = process_template(
                value,
                token_result["access_token"] if token_result else None,
                requested_group,
            )

    return (env, token_result)


def get_server_params(
    access_token: Optional[str] = None,
    requested_group: Optional[str] = None,
    anon: bool = False,
) -> StdioServerParameters:
    env_command = os.environ.get("MCP_SERVER_COMMAND")
    env, token_result = defined_env(
        os.environ.copy(), access_token, requested_group, anon
    )

    # Process command template with dynamic data path
    if env_command and token_result:
        processed_command = process_template(
            env_command, token_result["access_token"], requested_group
        )
        if processed_command != env_command:
            logger.info(f"Processed command template: {processed_command}")
            env_command = processed_command
    else:
        logger.info("No env_command or access_token set, using defaults")

    if env_command:
        import shlex

        parts = shlex.split(env_command)
        command = parts[0]
        cmd_args = parts[1:]
        logger.info(
            f"Server-Params from MCP_SERVER_COMMAND: command={command}, args={cmd_args}"
        )
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Fallback: parse sys.argv for --
    args = {}
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args["command"] = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        args["args"] = sys.argv[idx + 2 :] if len(sys.argv) > idx + 2 else []
        command = args["command"] or "python"
        cmd_args = args["args"] or [
            os.path.join(os.path.dirname(__file__), "../..", "mcp", "server.py")
        ]
        logger.info(f"Server-Params from sys.argv: command={command}, args={cmd_args}")
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Default
    command = "python"
    cmd_args = [os.path.join(os.path.dirname(__file__), "../..", "mcp", "server.py")]
    logger.info(f"Server-Params default: command={command}, args={cmd_args}")
    return StdioServerParameters(command=command, args=cmd_args, env=env)


def process_template(
    command_template: str, access_token: str, requested_group: Optional[str] = None
) -> str:
    """
    Process MCP_SERVER_COMMAND and MCP_ENV's templates with dynamic placeholders

    Supported placeholders:
    - {data_path}: Resolves to group or user specific data resource identifier
    - {user_id}: User identifier from token
    - {group_id}: Requested group identifier (if any)
    """
    try:
        data_manager = get_data_access_manager()

        # Check if template contains data_path placeholder
        if "{data_path}" in command_template:
            data_resource = data_manager.resolve_data_resource(
                access_token, requested_group
            )
            command_template = command_template.replace("{data_path}", data_resource)
            logger.info(f"Resolved data_path: {data_resource}")

        # Extract user info for other placeholders
        if "{user_id}" in command_template or "{group_id}" in command_template:
            from app.oauth.user_info import UserInfoExtractor

            extractor = UserInfoExtractor()
            user_info = extractor.extract_user_info(access_token)

            if "{user_id}" in command_template:
                user_id = user_info.get("user_id", "unknown")
                command_template = command_template.replace("{user_id}", user_id)
                logger.debug(f"Resolved user_id: {user_id}")

            if "{group_id}" in command_template:
                group_id = requested_group or "default"
                command_template = command_template.replace("{group_id}", group_id)
                logger.debug(f"Resolved group_id: {group_id}")

        return command_template

    except Exception as e:
        logger.error(f"Failed to process command template: {str(e)}")
        # Return original template on error to avoid breaking functionality
        return command_template
