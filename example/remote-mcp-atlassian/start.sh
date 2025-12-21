#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v curl >/dev/null || { echo -e "${RED}ERROR:${RESET} curl required"; exit 1; }
command -v docker >/dev/null || { echo -e "${RED}ERROR:${RESET} docker required"; exit 1; }
command -v openssl >/dev/null || { echo -e "${RED}ERROR:${RESET} openssl required"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FRONTEND_DOMAIN=inxm.local
KEYCLOAK_DOMAIN=auth.inxm.local
ENV_FILE=.env
PYTHON_BIN=""

set_env_var() {
  local key="$1"
  local value="$2"
  local tmp
  tmp=$(mktemp) || {
    echo -e "${RED}ERROR:${RESET} Failed to allocate temporary file while setting $key" >&2
    return 1
  }
  grep -v "^${key}=" "$ENV_FILE" > "$tmp" 2>/dev/null || true
  printf "%s=\"%s\"\n" "$key" "$value" >> "$tmp"
  if ! mv "$tmp" "$ENV_FILE"; then
    echo -e "${RED}ERROR:${RESET} Failed to persist $key to $ENV_FILE" >&2
    rm -f "$tmp"
    return 1
  fi
  return 0
}

ensure_python() {
  if [ -n "$PYTHON_BIN" ]; then
    return 0
  fi
  if command -v python3 >/dev/null; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null; then
    PYTHON_BIN=$(command -v python)
  elif [ -x "$HERE/.venv/bin/python" ]; then
    PYTHON_BIN="$HERE/.venv/bin/python"
  elif [ -x "$HERE/../.venv/bin/python" ]; then
    PYTHON_BIN="$HERE/../.venv/bin/python"
  else
    return 1
  fi
  return 0
}

register_atlassian_oauth_client() {
  local registration_url="https://cf.mcp.atlassian.com/v1/register"
  local redirect_uri="https://${KEYCLOAK_DOMAIN}/realms/inxm/broker/atlassian/endpoint"
  local auth_method=${ATLASSIAN_OIDC_CLIENT_AUTH_METHOD:-none}
  local scope=${ATLASSIAN_IDP_SCOPE:-"read:me offline_access"}
  case "$auth_method" in
    none|client_secret_post|client_secret_basic) ;;
    *)
      echo -e "${YELLOW}WARN:${RESET} Unknown auth method '$auth_method', defaulting to none."
      auth_method="none"
      ;;
  esac
  local payload
  payload=$(cat <<EOF
{"client_name":"inxm-atlassian-mcp-demo","redirect_uris":["${redirect_uri}"],"grant_types":["authorization_code","refresh_token"],"response_types":["code"],"token_endpoint_auth_method":"${auth_method}","scope":"${scope}"}
EOF
  )

  if ! ensure_python; then
    echo -e "${RED}ERROR:${RESET} python is required for Atlassian client registration. Install python or set ATLASSIAN_OIDC_CLIENT_ID manually." >&2
    return 1
  fi

  local response
  if ! response=$(curl -sS -X POST -H "Content-Type: application/json" -d "$payload" "$registration_url"); then
    echo -e "${RED}ERROR:${RESET} Failed to register Atlassian OAuth client." >&2
    return 1
  fi

  local client_id client_secret registered_auth_method
  _client_data=()
  while IFS= read -r line; do
    _client_data+=("$line")
  done < <(JSON_RESPONSE="$response" "$PYTHON_BIN" - <<'PY'
import json
import os
data = json.loads(os.environ["JSON_RESPONSE"])
print(data.get("client_id", ""))
print(data.get("client_secret", ""))
print(data.get("token_endpoint_auth_method", ""))
PY
  ) || true

  client_id=${_client_data[0]:-}
  client_secret=${_client_data[1]:-}
  registered_auth_method=${_client_data[2]:-}

  if [ -z "$client_id" ]; then
    echo -e "${RED}ERROR:${RESET} Atlassian registration did not return a client_id." >&2
    echo -e "${RED}Response:${RESET} $response" >&2
    return 1
  fi

  if [ -n "$registered_auth_method" ]; then
    auth_method=$registered_auth_method
  fi

  if [ -z "$client_secret" ]; then
    auth_method="none"
  fi

  ATLASSIAN_OIDC_CLIENT_ID=$client_id
  ATLASSIAN_OIDC_CLIENT_SECRET=$client_secret
  ATLASSIAN_OIDC_CLIENT_AUTH_METHOD=$auth_method
  export ATLASSIAN_OIDC_CLIENT_ID ATLASSIAN_OIDC_CLIENT_SECRET

  set_env_var "ATLASSIAN_OIDC_CLIENT_ID" "$ATLASSIAN_OIDC_CLIENT_ID"
  set_env_var "ATLASSIAN_OIDC_CLIENT_SECRET" "$ATLASSIAN_OIDC_CLIENT_SECRET"
  set_env_var "ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD"
  echo -e "${GREEN}OK${RESET} Registered Atlassian OAuth client."
  return 0
}

if [ ! -f "$ENV_FILE" ]; then
  cat <<"BANNER"
This setup connects the Enterprise MCP Bridge to an Atlassian MCP server and enables
workflow routing for Jira + Confluence tasks. You need a running Atlassian MCP server
(see https://github.com/atlassian/atlassian-mcp-server).

Login is routed through Keycloak and the Atlassian OAuth2 identity provider.
You can use a public client (PKCE, auth method none) or a confidential client.
BANNER

  read -rp "Do you want to continue? [Y/n]: " confirm
  if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo -e "${RED}Aborted by user.${RESET}"
    exit 0
  fi

  read -rp "Atlassian MCP server URL [https://mcp.atlassian.com/v1/sse]: " MCP_SERVER_INPUT
  ATLASSIAN_MCP_SERVER=${MCP_SERVER_INPUT:-"https://mcp.atlassian.com/v1/sse"}

  if [ -z "$ATLASSIAN_MCP_SERVER" ]; then
    echo -e "${RED}ERROR:${RESET} Atlassian MCP server URL is required"
    exit 1
  fi

  read -rp "Optional MCP bearer token (leave empty if not required): " ATLASSIAN_MCP_BEARER_TOKEN

  ATLASSIAN_IDP_PROVIDER="oauth2"

  read -rp "OAuth scopes (default: read:me offline_access): " SCOPE_INPUT
  ATLASSIAN_IDP_SCOPE=${SCOPE_INPUT:-"read:me offline_access"}

  read -rp "User info URL (default: https://api.atlassian.com/me): " USERINFO_INPUT
  ATLASSIAN_IDP_USERINFO_URL=${USERINFO_INPUT:-"https://api.atlassian.com/me"}

  read -rp "Atlassian OAuth client auth method [none/client_secret_post/client_secret_basic] (default: none): " AUTH_METHOD_INPUT
  ATLASSIAN_OIDC_CLIENT_AUTH_METHOD=${AUTH_METHOD_INPUT:-none}
  case "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" in
    none|client_secret_post|client_secret_basic) ;;
    *)
      echo -e "${YELLOW}WARN:${RESET} Unknown auth method '$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD', defaulting to none."
      ATLASSIAN_OIDC_CLIENT_AUTH_METHOD="none"
      ;;
  esac

  cat > "$ENV_FILE" <<EOF_ENV
ATLASSIAN_MCP_SERVER=$ATLASSIAN_MCP_SERVER
ATLASSIAN_IDP_PROVIDER=oauth2
ATLASSIAN_IDP_SCOPE="$ATLASSIAN_IDP_SCOPE"
ATLASSIAN_IDP_USERINFO_URL=$ATLASSIAN_IDP_USERINFO_URL
ATLASSIAN_OIDC_CLIENT_AUTH_METHOD=$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
EOF_ENV

  if [ -n "$ATLASSIAN_MCP_BEARER_TOKEN" ]; then
    echo "ATLASSIAN_MCP_BEARER_TOKEN=$ATLASSIAN_MCP_BEARER_TOKEN" >> "$ENV_FILE"
  fi

  echo "" >> "$ENV_FILE"
  echo "This demo optionally connects to an OpenAI-compatible LLM."
  read -rp "Enter the base url for your OpenAI-compatible service (ie. https://api.openai.com/v1/, leave empty for bundled dummy LLM): " BASE_URL
  if [ -z "$BASE_URL" ]; then
    BASE_URL="http://inxm.dummy-llm:8765/v1"
    cat >> "$ENV_FILE" <<EOF_ENV
OAI_BASE_URL=$BASE_URL
OAI_HOST=inxm.dummy-llm
OAI_API_TOKEN=none
OAI_MODEL_NAME=none
EOF_ENV
    echo -e "${GREEN}OK${RESET} Dummy LLM configured"
  else
    echo "OAI_BASE_URL=$BASE_URL" >> "$ENV_FILE"
    read -rp "Enter the API token for OpenAI-compatible services: " API_TOKEN
    echo "OAI_API_TOKEN=$API_TOKEN" >> "$ENV_FILE"
    OAI_HOST=$(echo "$BASE_URL" | awk -F[/:] '{print $4}')
    echo "OAI_HOST=$OAI_HOST" >> "$ENV_FILE"
    read -rp "Model Name for OpenAI-compatible service [gpt-5-mini-2025-08-07]: " MODEL_NAME
    echo "OAI_MODEL_NAME=${MODEL_NAME:-"gpt-5-mini-2025-08-07"}" >> "$ENV_FILE"
    echo -e "${GREEN}OK${RESET} LLM connection configured"
  fi
else
  echo -e "${GREEN}OK${RESET} Reusing existing .env"
fi

source "$ENV_FILE"

if [ -z "${ATLASSIAN_MCP_SERVER:-}" ]; then
  echo -e "${RED}ERROR:${RESET} Missing ATLASSIAN_MCP_SERVER in .env"
  exit 1
fi

ATLASSIAN_IDP_PROVIDER=${ATLASSIAN_IDP_PROVIDER:-oauth2}
ATLASSIAN_IDP_SCOPE="${ATLASSIAN_IDP_SCOPE:-"read:me offline_access"}"
ATLASSIAN_IDP_USERINFO_URL=${ATLASSIAN_IDP_USERINFO_URL:-"https://api.atlassian.com/me"}
ATLASSIAN_OIDC_CLIENT_ID=${ATLASSIAN_OIDC_CLIENT_ID:-}
ATLASSIAN_OIDC_CLIENT_SECRET=${ATLASSIAN_OIDC_CLIENT_SECRET:-}
ATLASSIAN_OIDC_CLIENT_AUTH_METHOD=${ATLASSIAN_OIDC_CLIENT_AUTH_METHOD:-none}
ATLASSIAN_OIDC_PKCE_ENABLED=${ATLASSIAN_OIDC_PKCE_ENABLED:-}
ATLASSIAN_OIDC_PKCE_METHOD=${ATLASSIAN_OIDC_PKCE_METHOD:-}

if [ -z "$ATLASSIAN_OIDC_CLIENT_ID" ]; then
  echo -e "${INFO}No Atlassian OAuth client id found. Attempting dynamic registration...${RESET}"
  if ! register_atlassian_oauth_client; then
    read -rp "Atlassian OAuth client id: " ATLASSIAN_OIDC_CLIENT_ID
    if [ -z "$ATLASSIAN_OIDC_CLIENT_ID" ]; then
      echo -e "${RED}ERROR:${RESET} Atlassian OAuth client id is required"
      exit 1
    fi
    read -rp "Atlassian OAuth client secret (leave empty for public client): " ATLASSIAN_OIDC_CLIENT_SECRET
    if [ -n "$ATLASSIAN_OIDC_CLIENT_SECRET" ]; then
      read -rp "Token auth method [client_secret_post/client_secret_basic] (default: client_secret_post): " CLIENT_AUTH_INPUT
      ATLASSIAN_OIDC_CLIENT_AUTH_METHOD=${CLIENT_AUTH_INPUT:-client_secret_post}
      case "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" in
        client_secret_post|client_secret_basic) ;;
        *)
          echo -e "${YELLOW}WARN:${RESET} Unknown auth method '$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD', defaulting to client_secret_post."
          ATLASSIAN_OIDC_CLIENT_AUTH_METHOD="client_secret_post"
          ;;
      esac
    else
      ATLASSIAN_OIDC_CLIENT_AUTH_METHOD="none"
    fi
    set_env_var "ATLASSIAN_OIDC_CLIENT_ID" "$ATLASSIAN_OIDC_CLIENT_ID"
    set_env_var "ATLASSIAN_OIDC_CLIENT_SECRET" "$ATLASSIAN_OIDC_CLIENT_SECRET"
    set_env_var "ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD"
  fi
fi

if [ -z "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" ]; then
  if [ -n "$ATLASSIAN_OIDC_CLIENT_SECRET" ]; then
    ATLASSIAN_OIDC_CLIENT_AUTH_METHOD="client_secret_post"
  else
    ATLASSIAN_OIDC_CLIENT_AUTH_METHOD="none"
  fi
fi
if [ -n "$ATLASSIAN_IDP_PROVIDER" ] && [ "$ATLASSIAN_IDP_PROVIDER" != "oauth2" ]; then
  echo -e "${YELLOW}WARN:${RESET} ATLASSIAN_IDP_PROVIDER=$ATLASSIAN_IDP_PROVIDER is not supported; forcing oauth2."
fi
ATLASSIAN_IDP_PROVIDER="oauth2"
set_env_var "ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD"
set_env_var "ATLASSIAN_IDP_PROVIDER" "$ATLASSIAN_IDP_PROVIDER"
set_env_var "ATLASSIAN_IDP_SCOPE" "$ATLASSIAN_IDP_SCOPE"
set_env_var "ATLASSIAN_IDP_USERINFO_URL" "$ATLASSIAN_IDP_USERINFO_URL"

if ! ls keycloak/providers/*.jar >/dev/null 2>&1; then
  echo -e "${RED}ERROR:${RESET} OAuth2 broker plugin not found."
  echo -e "${RED}ERROR:${RESET} Place a Keycloak OAuth2 IdP SPI jar in keycloak/providers and re-run."
  exit 1
fi

if [ -z "$ATLASSIAN_OIDC_PKCE_ENABLED" ]; then
  if [ "$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD" = "none" ]; then
    ATLASSIAN_OIDC_PKCE_ENABLED="true"
  else
    ATLASSIAN_OIDC_PKCE_ENABLED="false"
  fi
fi
ATLASSIAN_OIDC_PKCE_METHOD=${ATLASSIAN_OIDC_PKCE_METHOD:-S256}
set_env_var "ATLASSIAN_OIDC_PKCE_ENABLED" "$ATLASSIAN_OIDC_PKCE_ENABLED"
set_env_var "ATLASSIAN_OIDC_PKCE_METHOD" "$ATLASSIAN_OIDC_PKCE_METHOD"

TEMPLATE=keycloak/realm-export/realm-atlassian-template.json
TARGET=keycloak/realm-export/realm-atlassian.json
if [ ! -f "$TEMPLATE" ]; then
  echo -e "${RED}ERROR:${RESET} Keycloak realm template missing at $TEMPLATE"
  exit 1
fi

echo -e "${INFO}Rendering Keycloak realm export...${RESET}"
mkdir -p keycloak/realm-export
export ATLASSIAN_IDP_PROVIDER ATLASSIAN_IDP_SCOPE ATLASSIAN_IDP_USERINFO_URL
export ATLASSIAN_OIDC_CLIENT_ID ATLASSIAN_OIDC_CLIENT_SECRET ATLASSIAN_OIDC_CLIENT_AUTH_METHOD
export ATLASSIAN_OIDC_PKCE_ENABLED ATLASSIAN_OIDC_PKCE_METHOD
if command -v envsubst >/dev/null; then
  envsubst < "$TEMPLATE" > "$TARGET"
else
  sed \
    -e "s/\${ATLASSIAN_IDP_PROVIDER}/$ATLASSIAN_IDP_PROVIDER/g" \
    -e "s/\${ATLASSIAN_IDP_SCOPE}/$ATLASSIAN_IDP_SCOPE/g" \
    -e "s|\${ATLASSIAN_IDP_USERINFO_URL}|$ATLASSIAN_IDP_USERINFO_URL|g" \
    -e "s/\${ATLASSIAN_OIDC_CLIENT_ID}/$ATLASSIAN_OIDC_CLIENT_ID/g" \
    -e "s/\${ATLASSIAN_OIDC_CLIENT_SECRET}/$ATLASSIAN_OIDC_CLIENT_SECRET/g" \
    -e "s/\${ATLASSIAN_OIDC_CLIENT_AUTH_METHOD}/$ATLASSIAN_OIDC_CLIENT_AUTH_METHOD/g" \
    -e "s/\${ATLASSIAN_OIDC_PKCE_ENABLED}/$ATLASSIAN_OIDC_PKCE_ENABLED/g" \
    -e "s/\${ATLASSIAN_OIDC_PKCE_METHOD}/$ATLASSIAN_OIDC_PKCE_METHOD/g" \
    "$TEMPLATE" > "$TARGET"
fi

copy_hosts_entry() {
  local host=$1
  if ! grep -Eq "\\b$host\\b" /etc/hosts; then
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null; then
        echo -e "${YELLOW}WARN:${RESET} Adding hosts entry for $host (requires sudo)..."
        if echo "127.0.0.1 $host" | sudo tee -a /etc/hosts >/dev/null; then
          echo -e "${GREEN}OK${RESET} Added hosts entry: 127.0.0.1 $host"
        else
          echo -e "${RED}WARN:${RESET} Failed to add hosts entry via sudo. Add manually: 127.0.0.1 $host" >&2
        fi
      else
        echo -e "${RED}WARN:${RESET} sudo not available. Add this line to /etc/hosts manually: 127.0.0.1 $host" >&2
      fi
    else
      echo "127.0.0.1 $host" >> /etc/hosts
      echo -e "${GREEN}OK${RESET} Added hosts entry: 127.0.0.1 $host"
    fi
  else
    echo -e "${INFO}Hosts entry for $host already present${RESET}"
  fi
}

echo -e "${INFO}Ensuring /etc/hosts has required domain entries...${RESET}"
copy_hosts_entry "$FRONTEND_DOMAIN"
copy_hosts_entry "$KEYCLOAK_DOMAIN"

CERT_DIR=$HERE/../dev-local-certs
CA_KEY=$CERT_DIR/devLocalCA.key
CA_CERT=$CERT_DIR/devLocalCA.pem

generate_cert() {
  local domain=$1
  local key=$CERT_DIR/$domain.key
  local csr=$CERT_DIR/$domain.csr
  local crt=$CERT_DIR/$domain.crt
  local ext=$CERT_DIR/$domain.ext
  if [ -f "$crt" ]; then
    echo -e "Cert already exists for $domain"
    return 0
  fi
  echo -e "Generating certificate for $domain"
  openssl genrsa -out "$key" 2048 >/dev/null 2>&1
  openssl req -new -key "$key" -out "$csr" -subj "/CN=$domain" >/dev/null 2>&1
  cat > "$ext" <<EOF_EXT
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $domain
EOF_EXT
  openssl x509 -req -in "$csr" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial -out "$crt" -days 825 -sha256 -extfile "$ext" >/dev/null 2>&1
  echo -e "${GREEN}OK${RESET} Created cert: $crt"
}

echo -e "${INFO}Preparing local CA and certificates (sudo may be required to trust CA)...${RESET}"
mkdir -p "$CERT_DIR"
if [ ! -f "$CA_KEY" ]; then
  echo -e "Generating local development CA"
  openssl genrsa -out "$CA_KEY" 4096 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 -out "$CA_CERT" -subj "/CN=Local Dev CA" >/dev/null 2>&1
  if command -v sudo >/dev/null; then
    if grep -qi darwin <<<"$(uname)"; then
      sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CA_CERT" || \
        echo -e "${RED}WARN:${RESET} Failed to trust CA automatically on macOS. Install manually: $CA_CERT" >&2
    else
      sudo cp "$CA_CERT" /usr/local/share/ca-certificates/dev-local-ca.crt >/dev/null 2>&1 && \
        sudo update-ca-certificates >/dev/null 2>&1 || \
        echo -e "${RED}WARN:${RESET} Failed to trust CA automatically. Install manually: $CA_CERT" >&2
    fi
  fi
else
  echo -e "Local CA already present"
fi

generate_cert "$FRONTEND_DOMAIN"
generate_cert "$KEYCLOAK_DOMAIN"

echo -e "${INFO}Copying certificates to local dev-local-certs directory...${RESET}"
mkdir -p "$HERE/dev-local-certs"
cp "$CERT_DIR"/*.crt "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.key "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.pem "$HERE/dev-local-certs/" 2>/dev/null || true

echo -e "${INFO}Starting docker compose...${RESET}"
docker compose up -d --build

echo
echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}OK${RESET} Atlassian MCP workflow demo is ready!"
echo -e "${INFO}===========================================${RESET}"
echo
echo -e "Open the portal:   ${INFO}https://$FRONTEND_DOMAIN${RESET}"
echo -e "Keycloak admin:    ${INFO}https://$KEYCLOAK_DOMAIN${RESET} (user: $KEYCLOAK_ADMIN / pass: $KEYCLOAK_ADMIN_PASSWORD)"
echo -e "Remote MCP tools:  ${INFO}https://$FRONTEND_DOMAIN/api/mcp/atlassian/tools${RESET}"
echo -e "Login with Atlassian when redirected."
