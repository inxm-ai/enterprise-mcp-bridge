import logging
import os
import sys
from typing import Optional
from mcp import StdioServerParameters
from app.oauth.token_exchange import TokenRetrieverFactory

logger = logging.getLogger("uvicorn.error")


def defined_env(
    env: dict[str, str], access_token: Optional[str] = None
) -> dict[str, str]:
    oauth_env_var = env.get("OAUTH_ENV")
    if oauth_env_var:
        if not access_token:
            raise ValueError("access_token required when OAUTH_ENV is set")
        logger.info(f"Using OAUTH_ENV: {oauth_env_var} for token retrieval")
        retriever = TokenRetrieverFactory().get()
        logger.info(f"Retrieving token for {oauth_env_var} using TokenRetriever")
        token_result = retriever.retrieve_token(access_token)
        if not token_result or "access_token" not in token_result:
            raise ValueError(
                f"Token retrieval failed for {oauth_env_var} with access_token: {access_token}"
            )
        logger.info(
            f"Server-Params from OAUTH_ENV: {oauth_env_var}={token_result['access_token']}"
        )
        env[oauth_env_var] = token_result["access_token"]
    else:
        logger.info("No OAUTH_ENV set, using default environment variables")
    return env


def get_server_params(access_token: Optional[str] = None) -> StdioServerParameters:
    env_command = os.environ.get("MCP_SERVER_COMMAND")
    env = defined_env(os.environ.copy(), access_token)
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
