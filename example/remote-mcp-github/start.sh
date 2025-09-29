#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v curl >/dev/null || { echo -e "${RED}❌ curl required${RESET}"; exit 1; }

PYTHON_BIN=""

mask_token() {
  local token="$1"
  local len=${#token}
  if [ "$len" -le 8 ]; then
    printf "%s" "$token"
  else
    printf "%s***%s" "${token:0:4}" "${token: -4}"
  fi
}

set_env_var() {
  local key="$1"
  local value="$2"
  local tmp
  tmp=$(mktemp) || {
    echo -e "${RED}❌ Failed to allocate temporary file while setting $key${RESET}" >&2
    return 1
  }
  grep -v "^${key}=" "$ENV_FILE" > "$tmp" 2>/dev/null || true
  printf "%s=%s\n" "$key" "$value" >> "$tmp"
  if ! mv "$tmp" "$ENV_FILE"; then
    echo -e "${RED}❌ Failed to persist $key to $ENV_FILE${RESET}" >&2
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
  else
    echo -e "${RED}❌ python3 (or python) is required to parse GitHub device flow responses${RESET}" >&2
    return 1
  fi
  return 0
}

github_device_flow() {
  local label="$1"
  local scopes="$2"

  if ! ensure_python; then
    return 1
  fi

  local device_response
  if ! device_response=$(curl -sS \
    -X POST \
    -H "Accept: application/json" \
    -d "client_id=$GITHUB_CLIENT_ID" \
    -d "scope=$scopes" \
    https://github.com/login/device/code); then
    echo -e "${RED}❌ Failed to start GitHub device flow${RESET}" >&2
    return 1
  fi

  local device_code user_code verification_uri interval
  _device_data=()
  while IFS= read -r line; do
    _device_data+=("$line")
  done < <(JSON_RESPONSE="$device_response" "$PYTHON_BIN" - <<'PY'
import json, os
data = json.loads(os.environ["JSON_RESPONSE"])
print(data.get("device_code", ""))
print(data.get("user_code", ""))
print(data.get("verification_uri", ""))
print(data.get("interval", 5))
PY
  ) || true

  device_code=${_device_data[0]:-}
  user_code=${_device_data[1]:-}
  verification_uri=${_device_data[2]:-}
  interval=${_device_data[3]:-5}

  if [ -z "$device_code" ] || [ -z "$user_code" ] || [ -z "$verification_uri" ]; then
    echo -e "${RED}❌ Unexpected response from GitHub while initiating device flow${RESET}" >&2
    echo -e "${RED}Response: $device_response${RESET}" >&2
    return 1
  fi

  echo -e "${INFO}▶ Authorize ${label} token${RESET}" >&2
  echo -e "  1. Visit ${YELLOW}$verification_uri${RESET}" >&2
  echo -e "  2. Enter the code ${YELLOW}$user_code${RESET}" >&2
  echo -e "  3. Approve the requested scopes: $scopes" >&2
  read -rp "Press Enter once authorization is complete..." _ < /dev/tty

  local poll_interval=$interval
  local attempts=0
  while [ $attempts -lt 60 ]; do
    local token_response
    if ! token_response=$(curl -sS \
      -X POST \
      -H "Accept: application/json" \
      -d "client_id=$GITHUB_CLIENT_ID" \
      -d "device_code=$device_code" \
      -d "grant_type=urn:ietf:params:oauth:grant-type:device_code" \
      https://github.com/login/oauth/access_token); then
      echo -e "${RED}❌ Failed to poll GitHub for ${label} token${RESET}" >&2
      return 1
    fi

    local access_token error_code error_description
    _token_data=()
    while IFS= read -r line; do
      _token_data+=("$line")
    done < <(JSON_RESPONSE="$token_response" "$PYTHON_BIN" - <<'PY'
import json, os
data = json.loads(os.environ["JSON_RESPONSE"])
print(data.get("access_token", ""))
print(data.get("error", ""))
print(data.get("error_description", ""))
PY
    ) || true

    access_token=${_token_data[0]:-}
    error_code=${_token_data[1]:-}
    error_description=${_token_data[2]:-}

    if [ -n "$access_token" ]; then
      printf "%s" "$access_token"
      return 0
    fi

    if [ "$error_code" = "authorization_pending" ]; then
      sleep "$poll_interval"
    elif [ "$error_code" = "slow_down" ]; then
      poll_interval=$((poll_interval + 5))
      sleep "$poll_interval"
    elif [ "$error_code" = "expired_token" ]; then
      echo -e "${RED}❌ Authorization expired. Restart the device flow for ${label}.${RESET}" >&2
      return 1
    elif [ -n "$error_code" ]; then
      echo -e "${RED}❌ GitHub device flow error: $error_code ${error_description}${RESET}" >&2
      return 1
    else
      sleep "$poll_interval"
    fi
    attempts=$((attempts + 1))
  done

  echo -e "${RED}❌ Timed out waiting for ${label} authorization${RESET}" >&2
  return 1
}

maybe_fetch_token() {
  local env_key="$1"
  local label="$2"
  local default_scopes="$3"
  local existing_value=${!env_key:-}

  local prompt_action
  if [ -n "$existing_value" ]; then
    read -rp "Regenerate $label token for $env_key? [y/N]: " prompt_action || true
    if [[ ! "$prompt_action" =~ ^[Yy]$ ]]; then
        return 0
    fi
  else
    read -rp "Generate $label token for $env_key via GitHub device flow (needed for startup)? [Y/n]: " prompt_action || true
    if [[ "$prompt_action" =~ ^[Nn]$ ]]; then
        return 0
    fi
  fi


  read -rp "Scopes for $label token [$default_scopes]: " scope_input
  local scopes=${scope_input:-$default_scopes}

  local token
  if token=$(github_device_flow "$label" "$scopes"); then
    if set_env_var "$env_key" "$token"; then
      export "$env_key=$token"
      local masked
      masked=$(mask_token "$token")
      echo -e "${GREEN}✔${RESET} Stored $env_key=$masked in $ENV_FILE"
    else
      echo -e "${RED}❌ Failed to persist $env_key${RESET}" >&2
      return 1
    fi
  else
    echo -e "${RED}❌ Unable to obtain $label token${RESET}" >&2
    return 1
  fi

  return 0
}

command -v docker >/dev/null || { echo -e "${RED}❌ docker required${RESET}"; exit 1; }
command -v openssl >/dev/null || { echo -e "${RED}❌ openssl required${RESET}"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FRONTEND_DOMAIN=inxm.local
KEYCLOAK_DOMAIN=auth.inxm.local
ENV_FILE=.env

if [ ! -f "$ENV_FILE" ]; then
  cat <<"BANNER"
This setup will guide you through configuring a local Keycloak realm that delegates login to GitHub and then connects
that identity to the GitHub Remote MCP service (https://api.githubcopilot.com/mcp/).

You'll need a GitHub OAuth App (classic) with the following settings:
  • Authorization callback URL: https://auth.inxm.local/realms/inxm/broker/github/endpoint
  • Suggested scopes: read:user, user:email, repo (adjust to your needs)
  • Device flow enabled

Have the Client ID and Client Secret ready before continuing.
BANNER

  read -p "Do you want to continue? [Y/n]: " confirm
  if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo -e "${RED}Aborted by user.${RESET}"
    exit 0
  fi

  read -rp "GitHub OAuth Client ID: " GITHUB_CLIENT_ID
  if [ -z "$GITHUB_CLIENT_ID" ]; then
    echo -e "${RED}Client ID is required${RESET}"
    exit 1
  fi

  read -rp "GitHub OAuth Client Secret: " GITHUB_CLIENT_SECRET
  if [ -z "$GITHUB_CLIENT_SECRET" ]; then
    echo -e "${RED}Client Secret is required${RESET}"
    exit 1
  fi

  read -rp "GitHub OAuth scopes [read:user user:email repo]: " SCOPE_INPUT
  GITHUB_OAUTH_SCOPES=${SCOPE_INPUT:-"read:user user:email repo"}

  read -rp "Optional MCP workspace id (sets X-MCP-WORKSPACE-ID header) [skip]: " WORKSPACE_ID

  cat > "$ENV_FILE" <<EOF
GITHUB_CLIENT_ID=$GITHUB_CLIENT_ID
GITHUB_CLIENT_SECRET=$GITHUB_CLIENT_SECRET
GITHUB_OAUTH_SCOPES="$GITHUB_OAUTH_SCOPES"
MCP_REMOTE_HEADER_X_MCP_READONLY=true
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
MCP_REMOTE_REDIRECT_URI=https://$FRONTEND_DOMAIN/oauth2/callback
EOF

  if [ -n "$WORKSPACE_ID" ]; then
    echo "MCP_REMOTE_HEADER_X_MCP_WORKSPACE_ID=$WORKSPACE_ID" >> "$ENV_FILE"
  fi

  echo "" >> "$ENV_FILE"
  echo "⚠️ This demo optionally connects to an OpenAI-compatible LLM."
  read -p "Enter the base url for your OpenAI-compatible service (leave empty for bundled dummy LLM): " BASE_URL
  if [ -z "$BASE_URL" ]; then
    BASE_URL="http://inxm.dummy-llm:8765/v1"
    cat >> "$ENV_FILE" <<EOF
OAI_BASE_URL=$BASE_URL
OAI_HOST=inxm.dummy-llm
OAI_API_TOKEN=none
OAI_MODEL_NAME=none
EOF
    echo -e "${GREEN}✔${RESET} Dummy LLM configured"
  else
    echo "OAI_BASE_URL=$BASE_URL" >> "$ENV_FILE"
    read -p "Enter the API token for OpenAI-compatible services: " API_TOKEN
    echo "OAI_API_TOKEN=$API_TOKEN" >> "$ENV_FILE"
    OAI_HOST=$(echo "$BASE_URL" | awk -F[/:] '{print $4}')
    echo "OAI_HOST=$OAI_HOST" >> "$ENV_FILE"
    read -p "Model Name for OpenAI-compatible service: " MODEL_NAME
    echo "OAI_MODEL_NAME=$MODEL_NAME" >> "$ENV_FILE"
    echo -e "${GREEN}✔${RESET} LLM connection configured"
  fi
else
  echo -e "${GREEN}✔${RESET} Reusing existing .env"
fi

source "$ENV_FILE"

if [ -z "${GITHUB_CLIENT_ID:-}" ] || [ -z "${GITHUB_CLIENT_SECRET:-}" ]; then
  echo -e "${RED}❌ Missing GitHub OAuth credentials in .env${RESET}"
  exit 1
fi

if ! grep -q MCP_REMOTE_HEADER_X_MCP_READONLY "$ENV_FILE"; then
  echo "MCP_REMOTE_HEADER_X_MCP_READONLY=true" >> "$ENV_FILE"
fi

if ! grep -q MCP_REMOTE_REDIRECT_URI "$ENV_FILE"; then
  echo "MCP_REMOTE_REDIRECT_URI=https://$FRONTEND_DOMAIN/oauth2/callback" >> "$ENV_FILE"
fi

maybe_fetch_token "MCP_REMOTE_ANON_BEARER_TOKEN" "anonymous fallback" "read:user user:email"
source "$ENV_FILE"

export GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET GITHUB_OAUTH_SCOPES
TEMPLATE=keycloak/realm-export/realm-ghmcp-template.json
TARGET=keycloak/realm-export/realm-inxm.json

if [ ! -f "$TEMPLATE" ]; then
  echo -e "${RED}❌ Keycloak realm template missing at $TEMPLATE${RESET}"
  exit 1
fi

echo -e "${INFO}🔄 Rendering Keycloak realm export...${RESET}"
mkdir -p keycloak/realm-export
if command -v envsubst >/dev/null; then
  envsubst < "$TEMPLATE" > "$TARGET"
else
  sed \
    -e "s/\${GITHUB_CLIENT_ID}/$GITHUB_CLIENT_ID/g" \
    -e "s/\${GITHUB_CLIENT_SECRET}/$GITHUB_CLIENT_SECRET/g" \
    -e "s/\${GITHUB_OAUTH_SCOPES}/$GITHUB_OAUTH_SCOPES/g" \
    "$TEMPLATE" > "$TARGET"
fi

copy_hosts_entry() {
  local host=$1
  if ! grep -Eq "\\b$host\\b" /etc/hosts; then
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null; then
        echo -e "${YELLOW}⚠️ Adding hosts entry for $host (requires sudo)...${RESET}"
        if echo "127.0.0.1 $host" | sudo tee -a /etc/hosts >/dev/null; then
          echo -e "${GREEN}✔${RESET} Added hosts entry: 127.0.0.1 $host"
        else
          echo -e "${RED}❌ WARNING: Failed to add hosts entry via sudo. Add manually: 127.0.0.1 $host${RESET}" >&2
        fi
      else
        echo -e "${RED}❌ WARNING: sudo not available. Add this line to /etc/hosts manually: 127.0.0.1 $host${RESET}" >&2
      fi
    else
      echo "127.0.0.1 $host" >> /etc/hosts
      echo -e "${GREEN}✔${RESET} Added hosts entry: 127.0.0.1 $host"
    fi
  else
    echo -e "${INFO}ℹ️ Hosts entry for $host already present${RESET}"
  fi
}

echo -e "${INFO}🔍 Ensuring /etc/hosts has required domain entries...${RESET}"
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
    echo -e "ℹ️ Cert already exists for $domain"
    return 0
  fi
  echo -e "🔐 Generating certificate for $domain"
  openssl genrsa -out "$key" 2048 >/dev/null 2>&1
  openssl req -new -key "$key" -out "$csr" -subj "/CN=$domain" >/dev/null 2>&1
  cat > "$ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $domain
EOF
  openssl x509 -req -in "$csr" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial -out "$crt" -days 825 -sha256 -extfile "$ext" >/dev/null 2>&1
  echo -e "${GREEN}✔${RESET} Created cert: $crt"
}

echo -e "${INFO}🔄 Preparing local CA and certificates (sudo may be required to trust CA)...${RESET}"
mkdir -p "$CERT_DIR"
if [ ! -f "$CA_KEY" ]; then
  echo -e "🔑 Generating local development CA"
  openssl genrsa -out "$CA_KEY" 4096 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 -out "$CA_CERT" -subj "/CN=Local Dev CA" >/dev/null 2>&1
  if command -v sudo >/dev/null; then
    if grep -qi darwin <<<"$(uname)"; then
      sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CA_CERT" || \
        echo -e "${RED}❌ WARNING: Failed to trust CA automatically on macOS. Install manually: $CA_CERT${RESET}" >&2
    else
      sudo cp "$CA_CERT" /usr/local/share/ca-certificates/dev-local-ca.crt >/dev/null 2>&1 && \
        sudo update-ca-certificates >/dev/null 2>&1 || \
        echo -e "${RED}❌ WARNING: Failed to trust CA automatically. Install manually: $CA_CERT${RESET}" >&2
    fi
  fi
else
  echo -e "ℹ️ Local CA already present"
fi

generate_cert "$FRONTEND_DOMAIN"
generate_cert "$KEYCLOAK_DOMAIN"

echo -e "${INFO}🔄 Copying certificates to local dev-local-certs directory...${RESET}"
mkdir -p "$HERE/dev-local-certs"
cp "$CERT_DIR"/*.crt "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.key "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.pem "$HERE/dev-local-certs/" 2>/dev/null || true

echo -e "${INFO}🔄 Starting docker compose...${RESET}"
docker compose up -d --build

echo
echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}🎉 GitHub Remote MCP demo is ready!${RESET}"
echo -e "${INFO}===========================================${RESET}"
echo
echo -e "Open the portal:   ${INFO}https://$FRONTEND_DOMAIN${RESET}"
echo -e "Keycloak admin:    ${INFO}https://$KEYCLOAK_DOMAIN${RESET} (user: $KEYCLOAK_ADMIN / pass: $KEYCLOAK_ADMIN_PASSWORD)"
echo -e "Remote MCP tools:  ${INFO}https://$FRONTEND_DOMAIN/api/mcp/github/tools${RESET}"
echo -e "Login with GitHub at the prompt when redirected."
