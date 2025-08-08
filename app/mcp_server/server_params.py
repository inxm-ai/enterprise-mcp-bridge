import logging
import os
import sys
from mcp import StdioServerParameters

logger = logging.getLogger("uvicorn.error")
def get_server_params():
    env_command = os.environ.get("MCP_SERVER_COMMAND")
    env = os.environ.copy()
    if env_command:
        # Split the env variable into command and args (simple shell-like split)
        import shlex
        parts = shlex.split(env_command)
        command = parts[0]
        cmd_args = parts[1:]
        logger.info(f"Server-Params from MCP_SERVER_COMMAND: command={command}, args={cmd_args}")
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Fallback: parse sys.argv for --
    args = {}
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args["command"] = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        args["args"] = sys.argv[idx + 2:] if len(sys.argv) > idx + 2 else []
        command = args["command"] or "python"
        cmd_args = args["args"] or [os.path.join(os.path.dirname(__file__), "..", "mcp", "server.py")]
        logger.info(f"Server-Params from sys.argv: command={command}, args={cmd_args}")
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Default
    command = "python"
    cmd_args = [os.path.join(os.path.dirname(__file__), "..", "mcp", "server.py")]
    logger.info(f"Server-Params default: command={command}, args={cmd_args}")
    return StdioServerParameters(command=command, args=cmd_args, env=env)