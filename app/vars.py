import os
import json
import re

SERVICE_NAME = os.getenv("SERVICE_NAME", "enterprise-mcp-bridge")
TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
TOKEN_COOKIE_NAME = os.environ.get("TOKEN_COOKIE_NAME", "_oauth2_proxy")
TOKEN_SOURCE = os.environ.get("TOKEN_SOURCE", "header").lower()
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]
# Tools that are modifying or notifying or similar
EFFECT_TOOLS = [t for t in os.environ.get("EFFECT_TOOLS", "").split(",") if t]
TGI_ENABLED = os.environ.get("TGI_URL", None) is not None


def _load_tool_output_schemas():
    def _load_env_mapping(env_name: str) -> dict:
        raw = os.environ.get(env_name)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception as e:
            print(f"Error parsing {env_name}: {e}")
            return {}
        if not isinstance(parsed, dict):
            print(f"Error parsing {env_name}: expected a JSON object mapping")
            return {}
        return parsed

    # Backwards compatibility: support legacy singular env name too.
    schemas = _load_env_mapping("TOOL_OUTPUT_SCHEMA")
    schemas.update(_load_env_mapping("TOOL_OUTPUT_SCHEMAS"))

    for tool_name, schema_or_path in schemas.items():
        if isinstance(schema_or_path, str):
            try:
                with open(schema_or_path, "r") as f:
                    schemas[tool_name] = json.load(f)
            except Exception as e:
                print(
                    f"Error loading schema for {tool_name} from {schema_or_path}: {e}"
                )
    return schemas


TOOL_OUTPUT_SCHEMAS = _load_tool_output_schemas()


def _canonicalize_tool_name(tool_name: str) -> str:
    if not isinstance(tool_name, str):
        return ""
    parts = re.split(r"[^a-z0-9]+", tool_name.lower())
    normalized = []
    for part in parts:
        if not part:
            continue
        if len(part) > 3 and part.endswith("ies"):
            part = f"{part[:-3]}y"
        elif len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
            part = part[:-1]
        normalized.append(part)
    return "-".join(normalized)


def get_tool_output_schema(tool_name: str):
    """Get output schema by exact match first, then canonicalized alias match."""
    if not isinstance(tool_name, str) or not tool_name:
        return None

    exact = TOOL_OUTPUT_SCHEMAS.get(tool_name)
    if exact is not None:
        return exact

    canonical_name = _canonicalize_tool_name(tool_name)
    if not canonical_name:
        return None

    matched_schema = None
    for name, schema in TOOL_OUTPUT_SCHEMAS.items():
        if _canonicalize_tool_name(name) != canonical_name:
            continue
        if matched_schema is not None and matched_schema != schema:
            return None
        matched_schema = schema
    return matched_schema


DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", os.environ.get("TGI_MODEL_NAME", ""))
OAUTH_ENV = os.environ.get("OAUTH_ENV", "")
HOST = os.environ.get("HOSTNAME", "")
PORT = os.environ.get("PORT", "")

TGI_MODEL_NAME = os.getenv("TGI_MODEL_NAME", os.environ.get("DEFAULT_MODEL", ""))

TOOL_CHUNK_SIZE = int(os.getenv("TOOL_CHUNK_SIZE", "10000"))
AGENT_CARD_CACHE_FILE = os.getenv("AGENT_CARD_CACHE_FILE", "/tmp/agent_card_cache.json")

LLM_MAX_PAYLOAD_BYTES = int(os.getenv("LLM_MAX_PAYLOAD_BYTES", "120000"))

OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT")
OTLP_HEADERS = os.getenv("OTLP_HEADERS", "")

AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "")
AUTH_PROVIDER = os.getenv("AUTH_PROVIDER", "keycloak").lower()
KEYCLOAK_PROVIDER_ALIAS = os.getenv("KEYCLOAK_PROVIDER_ALIAS", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "inxm")
KEYCLOAK_PROVIDER_REFRESH_MODE = os.getenv(
    "KEYCLOAK_PROVIDER_REFRESH_MODE", "oidc"
).lower()

AUTH_ALLOW_UNSAFE_CERT = os.getenv("AUTH_ALLOW_UNSAFE_CERT", "false").lower() == "true"
LOG_TOKEN_VALUES = os.getenv("LOG_TOKEN_VALUES", "false").lower() == "true"

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

APP_CONVERSATIONAL_UI_ENABLED = (
    os.getenv("APP_CONVERSATIONAL_UI_ENABLED", "false").lower() == "true"
)
APP_UI_SESSION_TTL_MINUTES = int(os.getenv("APP_UI_SESSION_TTL_MINUTES", "120"))
APP_UI_PATCH_ENABLED = os.getenv("APP_UI_PATCH_ENABLED", "true").lower() == "true"
GENERATED_UI_FIX_CODE_FIRST = (
    os.getenv("GENERATED_UI_FIX_CODE_FIRST", "true").lower() == "true"
)
GENERATED_UI_TOOL_TEXT_CAP = int(os.getenv("GENERATED_UI_TOOL_TEXT_CAP", "4000"))
GENERATED_UI_READ_ONLY_STREAK_LIMIT = int(
    os.getenv("GENERATED_UI_READ_ONLY_STREAK_LIMIT", "4")
)

WORKFLOW_MAX_PARALLEL_AGENTS = int(os.getenv("WORKFLOW_MAX_PARALLEL_AGENTS", "4"))
