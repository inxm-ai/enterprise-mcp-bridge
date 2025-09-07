import os

SERVICE_NAME = os.getenv("SERVICE_NAME", "enterprise-mcp-bridge")
TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", os.environ.get("TGI_MODEL_NAME", ""))
OAUTH_ENV = os.environ.get("OAUTH_ENV", "")
HOST = os.environ.get("HOSTNAME", "")
PORT = os.environ.get("PORT", "")

AGENT_CARD_CACHE_FILE = os.getenv("AGENT_CARD_CACHE_FILE", "/tmp/agent_card_cache.json")

OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT")
OTLP_HEADERS = os.getenv("OTLP_HEADERS", "")

AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "")
AUTH_PROVIDER = os.getenv("AUTH_PROVIDER", "keycloak").lower()
KEYCLOAK_PROVIDER_ALIAS = os.getenv("KEYCLOAK_PROVIDER_ALIAS", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "inxm")

AUTH_ALLOW_UNSAFE_CERT = os.getenv("AUTH_ALLOW_UNSAFE_CERT", "false").lower() == "true"

MCP_SESSION_MANAGER = os.getenv("MCP_SESSION_MANAGER", "InMemorySessionManager")
MCP_GROUP_DATA_ACCESS_TEMPLATE = os.getenv(
    "MCP_GROUP_DATA_ACCESS_TEMPLATE", "g/{group_id}"
)
MCP_USER_DATA_ACCESS_TEMPLATE = os.getenv(
    "MCP_USER_DATA_ACCESS_TEMPLATE", "u/{user_id}"
)
MCP_SHARED_DATA_ACCESS_TEMPLATE = os.getenv(
    "MCP_SHARED_DATA_ACCESS_TEMPLATE", "shared/{shared_id}"
)
