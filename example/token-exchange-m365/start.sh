#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v docker >/dev/null || { echo -e "${RED}❌ docker required${RESET}"; exit 1; }
command -v az >/dev/null || { echo -e "${RED}❌ Azure CLI (az) required${RESET}"; exit 1; }

echo "This script will need access to your Entra account to set up a new application."
echo "Admin permissions will be required."
read -p "Do you want to continue? [Y/n]: " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
  echo -e "${RED}Aborted by user.${RESET}"
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FRONTEND_DOMAIN=inxm.local
KEYCLOAK_DOMAIN=auth.inxm.local
ENV_FILE=.env

retry_command() {
  local retries=10
  local wait_time=2
  local attempt=1

  while [ $attempt -le $retries ]; do
    echo -e "🔄 Attempt $attempt of $retries..."
    "$@" && { echo -e "${GREEN}✔${RESET} Command succeeded"; return 0; }
    echo -e "${YELLOW}⚠️ Command failed. Retrying in $wait_time seconds...${RESET}"
    sleep $wait_time
    wait_time=$((wait_time * 2))
    attempt=$((attempt + 1))
  done

  echo -e "${RED}❌ Command failed after $retries attempts.${RESET}"
  return 1
}

if grep -q AZURE_CLIENT_ID "$ENV_FILE" 2>/dev/null; then
  echo -e "${GREEN}✔${RESET} Reusing existing .env"
  source "$ENV_FILE"
else
  echo -e "⚠️ Setting up new env, shutting down docker compose if open"
  docker compose down || true
  echo -e "${INFO}🔑 Azure login (device code if needed)...${RESET}"
  if ! az account show >/dev/null 2>&1; then az login --allow-no-subscriptions --use-device-code >/dev/null; fi
  TENANT_ID=$(az account show --query tenantId -o tsv)
  DISPLAY_NAME="mcp-rest-demo-$(date +%s)"
  echo -e "${GREEN}✔ Creating Entra app $DISPLAY_NAME in tenant $TENANT_ID${RESET}"
  APP_ID=$(az ad app create --display-name "$DISPLAY_NAME" --query appId -o tsv \
    --web-redirect-uris \
      "https://$FRONTEND_DOMAIN/" \
      "https://$FRONTEND_DOMAIN/oauth2/callback" \
      "https://$KEYCLOAK_DOMAIN/realms/inxm/account" \
      "https://$KEYCLOAK_DOMAIN/realms/inxm/broker/ms365/endpoint")
  SECRET_JSON=$(az ad app credential reset --id "$APP_ID" --display-name demo-secret)
  CLIENT_SECRET=$(echo "$SECRET_JSON" | grep -o '"password": "[^"]*"' | cut -d '"' -f4)
  cat > "$ENV_FILE" <<EOF
AZURE_CLIENT_ID=$APP_ID
AZURE_CLIENT_SECRET=$CLIENT_SECRET
AZURE_TENANT_ID=$TENANT_ID
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
EOF
  echo -e "${GREEN}✔${RESET} .env created"
  source "$ENV_FILE"

  echo -e "${INFO}🔍 Checking for existing Service Principal...${RESET}"
  GRAPH_API_ID="00000003-0000-0000-c000-000000000000"
  SERVICE_PRINCIPAL_ID=$(az ad sp show --id "$APP_ID" --query "id" -o tsv 2>/dev/null || echo "")

  if [ -z "$SERVICE_PRINCIPAL_ID" ]; then
      echo -e "⚠️ Service Principal not found. Creating one now..."
      SERVICE_PRINCIPAL_ID=$(az ad sp create --id "$APP_ID" --query "id" -o tsv)
      echo -e "${GREEN}✔${RESET} Created Service Principal with ID: $SERVICE_PRINCIPAL_ID"
  else
      echo -e "${GREEN}✔${RESET} Service Principal already exists"
  fi
  permission_scopes=(
      "User.Read"
      "MailboxSettings.Read"
      "Mail.Read"
  )
  get_permission_id() {
    local permission_name="$1"
    az ad sp show \
      --id "$GRAPH_API_ID" \
      --query "oauth2PermissionScopes[?value=='$permission_name'].id" \
      -o tsv
  }

  echo -e "For this demo, the following permissions will be granted: ${permission_scopes[*]}"
  echo -e "${INFO}🔧${RESET} Adding API permissions to the app..."
  for scope in "${permission_scopes[@]}"; do
    PERMISSION_ID=$(get_permission_id "$scope")
    if [ -n "$PERMISSION_ID" ]; then
        echo "Adding delegated permission: $scope (ID: $PERMISSION_ID)"
        az ad app permission add \
          --id "$APP_ID" \
          --api "$GRAPH_API_ID" \
          --api-permissions "$PERMISSION_ID"=Scope
    else
        echo "Warning: Could not find a GUID for permission: $scope"
    fi
  done

  echo -e "${GREEN}✔${RESET} Granting API permissions"
  az ad app permission grant --id "$SERVICE_PRINCIPAL_ID" --api "$GRAPH_API_ID" --scope "openid profile email offline_access ${permission_scopes[*]}" 
  echo -e "${INFO}🔑${RESET} Add admin consent - Note, this may fail on the first tries as the other things are still applied. It will retry automatically"
  retry_command az ad app permission admin-consent --id "$APP_ID"
  if [ $? -ne 0 ]; then
    exit 1
  fi

  echo -e "${INFO}🔧 Enabling access and ID tokens for the app...${RESET}"
  az rest --method PATCH --uri "https://graph.microsoft.com/v1.0/applications/$(az ad app show --id $APP_ID --query id -o tsv)" \
  --headers "Content-Type=application/json" \
  --body '{
    "web": {
      "implicitGrantSettings": {
        "enableAccessTokenIssuance": true,
        "enableIdTokenIssuance": true
      }
    }
  }'
  echo -e "${GREEN}✔${RESET} Access and ID tokens enabled"
fi

if ! grep -q API_TOKEN "$ENV_FILE" 2>/dev/null; then
  echo "⚠️ This demo needs an openai api compatible llm provider with api token"
  echo "    It is shipped with a dummy-llm that will just select one tool as fallback"

  read -p "Enter the base url for your OpenAI-compatible service (leave empty for fallback): " BASE_URL
  echo >> "$ENV_FILE"
  if [ -z "$BASE_URL" ]; then
    BASE_URL="http://dummy-llm:8765/v1"
    
    echo "OAI_BASE_URL=$BASE_URL" >> "$ENV_FILE"
    echo "OAI_HOST=dummy-llm" >> "$ENV_FILE"
    echo "OAI_API_TOKEN=none" >> "$ENV_FILE"
    echo "OAI_MODEL_NAME=none" >> "$ENV_FILE"
    echo -e "${GREEN}✔${RESET} Dummy-LLM added to .env"
  else
    echo "OAI_BASE_URL=$BASE_URL" >> "$ENV_FILE"

    read -p "Enter the API token for OpenAI-compatible services: " API_TOKEN
    echo "OAI_API_TOKEN=$API_TOKEN" >> "$ENV_FILE"

    OAI_HOST=$(echo "$BASE_URL" | awk -F[/:] '{print $4}')
    echo "OAI_HOST=$OAI_HOST" >> "$ENV_FILE"

    read -p "Model Name for OpenAI-compatible service: " MODEL_NAME
    echo "OAI_MODEL_NAME=$MODEL_NAME" >> "$ENV_FILE"

    echo -e "${GREEN}✔${RESET} BASEURL & API token added to .env"
  fi
fi
source "$ENV_FILE"

echo -e "${INFO}🔄 Rendering Keycloak realm export...${RESET}"
mkdir -p keycloak/realm-export
export AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID
TEMPLATE=keycloak/realm-export/realm-inxm-template.json
TARGET=keycloak/realm-export/realm-inxm.json
if command -v envsubst >/dev/null; then envsubst < "$TEMPLATE" > "$TARGET"; else sed "s/\${AZURE_CLIENT_ID}/$AZURE_CLIENT_ID/g; s/\${AZURE_CLIENT_SECRET}/$AZURE_CLIENT_SECRET/g; s/\${AZURE_TENANT_ID}/$AZURE_TENANT_ID/g" "$TEMPLATE" > "$TARGET"; fi

echo -e "${INFO}🔍 Ensuring /etc/hosts has required domain entries...${RESET}"
NEEDED_HOSTS=("$FRONTEND_DOMAIN" "$KEYCLOAK_DOMAIN")
for host in "${NEEDED_HOSTS[@]}"; do
  if ! grep -Eq "\\b$host\\b" /etc/hosts; then
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null; then
        echo -e "${YELLOW}⚠️ Adding hosts entry for $host (requires sudo)...${RESET}"
        if echo "127.0.0.1 $host" | sudo tee -a /etc/hosts >/dev/null; then
          echo -e "${GREEN}✔ Added hosts entry: 127.0.0.1 $host${RESET}"
        else
          echo -e "${RED}❌ WARNING: Failed to add hosts entry via sudo. Add manually: 127.0.0.1 $host${RESET}" >&2
        fi
      else
        echo -e "${RED}❌ WARNING: sudo not available. Add this line to /etc/hosts manually: 127.0.0.1 $host${RESET}" >&2
      fi
    else
      echo "127.0.0.1 $host" >> /etc/hosts
      echo -e "${GREEN}✔ Added hosts entry: 127.0.0.1 $host${RESET}"
    fi
  else
    echo -e "${INFO}ℹ️ Hosts entry for $host already present${RESET}"
  fi

done

CERT_DIR=$HERE/dev-local-certs
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
  echo -e "🔐 Generating certificate for $domain$"
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
      if sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CA_CERT"; then
        echo -e "${GREEN}✔${RESET} Trusted local CA in macOS system keychain"
      else
        echo -e "${RED}❌ WARNING: Failed to trust CA automatically on macOS. Install manually: $CA_CERT${RESET}" >&2
      fi
    else
      # Linux: add CA to system store
      if sudo cp "$CA_CERT" /usr/local/share/ca-certificates/dev-local-ca.crt && sudo update-ca-certificates >/dev/null 2>&1; then
        echo -e "${GREEN}✔${RESET} Trusted local CA in system store"
      else
        echo -e "${RED}❌ WARNING: Failed to trust CA automatically. Install manually: $CA_CERT${RESET}" >&2
      fi
    fi
  fi
else
  echo -e "ℹ️ Local CA already present"
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  echo -e "${YELLOW}⚠️ Detected WSL environment.${RESET}"
  echo -e "${INFO}To trust the local CA in Windows for browser access, run:${RESET}"
  echo -e "    powershell.exe -Command \"Import-Certificate -FilePath '$CA_CERT' -CertStoreLocation 'Cert:\\LocalMachine\\Root'\""
  echo -e "${INFO}Or manually import the CA certificate ($CA_CERT) into Windows Trusted Root Certification Authorities.${RESET}"
fi

generate_cert "$FRONTEND_DOMAIN"
generate_cert "$KEYCLOAK_DOMAIN"

echo -e "${INFO}ℹ️ Certificates directory: $CERT_DIR${RESET}"
echo -e "    - CA: $CA_CERT"
echo -e "    - Frontend: $CERT_DIR/$FRONTEND_DOMAIN.crt"
echo -e "    - Keycloak: $CERT_DIR/$KEYCLOAK_DOMAIN.crt"

echo -e "${INFO}🔄 Starting docker compose...${RESET}"
docker compose up -d --build

echo
echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}🎉 All setup steps completed successfully!${RESET}"
echo -e "${INFO}===========================================${RESET}"
echo
echo -e "Open the page via:  ${INFO}https://$FRONTEND_DOMAIN${RESET}"
echo -e "See all MCP Tools:  ${INFO}https://$FRONTEND_DOMAIN/api/mcp/m365/tools${RESET}"
echo -e "See what you did:   ${INFO}https://$FRONTEND_DOMAIN/ops/jaeger${RESET}"
echo -e "Login via Keycloak -> Entra provider (ms365), then invoke tools."
