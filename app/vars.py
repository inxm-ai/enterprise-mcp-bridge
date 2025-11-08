import os

SERVICE_NAME = os.getenv("SERVICE_NAME", "enterprise-mcp-bridge")
TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]
# Tools that are modifying or notifying or similar
EFFECT_TOOLS = [t for t in os.environ.get("EFFECT_TOOLS", "").split(",") if t]

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", os.environ.get("TGI_MODEL_NAME", ""))
OAUTH_ENV = os.environ.get("OAUTH_ENV", "")
HOST = os.environ.get("HOSTNAME", "")
PORT = os.environ.get("PORT", "")

TGI_MODEL_NAME = os.getenv("TGI_MODEL_NAME", os.environ.get("DEFAULT_MODEL", ""))
TGI_MODEL_FORMAT = os.getenv("TGI_MODEL_FORMAT", "")

TOOL_INJECTION_MODE = os.getenv("TOOL_INJECTION_MODE", "openai")
TOOL_CHUNK_SIZE = int(os.getenv("TOOL_CHUNK_SIZE", "10000"))
AGENT_CARD_CACHE_FILE = os.getenv("AGENT_CARD_CACHE_FILE", "/tmp/agent_card_cache.json")

LLM_MAX_PAYLOAD_BYTES = int(os.getenv("LLM_MAX_PAYLOAD_BYTES", "120000"))

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

MCP_REMOTE_SERVER = os.getenv("MCP_REMOTE_SERVER", "")
MCP_REMOTE_SCOPE = os.getenv("MCP_REMOTE_SCOPE", "")
MCP_REMOTE_REDIRECT_URI = os.getenv(
    "MCP_REMOTE_REDIRECT_URI", "https://localhost/unused-callback"
)
MCP_REMOTE_CLIENT_ID = os.getenv("MCP_REMOTE_CLIENT_ID", "")
MCP_REMOTE_CLIENT_SECRET = os.getenv("MCP_REMOTE_CLIENT_SECRET", "")
MCP_REMOTE_BEARER_TOKEN = os.getenv("MCP_REMOTE_BEARER_TOKEN", "")
MCP_REMOTE_ANON_BEARER_TOKEN = os.getenv("MCP_REMOTE_ANON_BEARER_TOKEN", "")
MCP_REMOTE_SERVER_FORWARD_HEADERS = [
    h.strip()
    for h in os.getenv("MCP_REMOTE_SERVER_FORWARD_HEADERS", "").split(",")
    if h.strip()
]


def _parse_map_header_to_input(raw: str) -> dict:
    mapping: dict = {}
    if not raw:
        return mapping
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            key, val = entry.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                mapping[key] = val
    return mapping


MCP_MAP_HEADER_TO_INPUT = _parse_map_header_to_input(
    os.getenv("MCP_MAP_HEADER_TO_INPUT", "")
)

GENERATED_UI_PROMPT_DUMP = os.getenv("GENERATED_UI_PROMPT_DUMP", "")
