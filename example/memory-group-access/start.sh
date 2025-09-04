#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v docker >/dev/null || { echo -e "${RED}❌ docker required${RESET}"; exit 1; }

echo "This script will set up a memory-based MCP server with group-based access control."
echo "It demonstrates how different users can access different memory stores based on their group memberships."
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

# Create basic .env file for the memory server example
if [ ! -f "$ENV_FILE" ]; then
  echo -e "${INFO}🔄 Creating .env file...${RESET}"
  cat > "$ENV_FILE" <<EOF
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
# No external LLM needed for this example - it's just demonstrating group-based memory access
EOF
  echo -e "${GREEN}✔${RESET} .env created"
fi

source "$ENV_FILE"

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

echo -e "${INFO}🔄 Copying certificates to local dev-local-certs directory...${RESET}"
mkdir -p "$HERE/dev-local-certs"
cp "$CERT_DIR"/*.crt "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.key "$HERE/dev-local-certs/" 2>/dev/null || true
cp "$CERT_DIR"/*.pem "$HERE/dev-local-certs/" 2>/dev/null || true

echo -e "${INFO}ℹ️ Certificates directory: $CERT_DIR${RESET}"
echo -e "    - CA: $CA_CERT"
echo -e "    - Frontend: $CERT_DIR/$FRONTEND_DOMAIN.crt"
echo -e "    - Keycloak: $CERT_DIR/$KEYCLOAK_DOMAIN.crt"

echo -e "${INFO}🔄 Creating memory data directories...${RESET}"
mkdir -p data/g data/u
# Create sample memory files for different groups
echo '{"type": "entity", "name": "administrators_memory", "entityType": "memory", "observations": ["The admin'\''s name is Matthias", "System maintenance scheduled for next weekend."]}' > data/g/administrators.json
echo '{"type": "entity", "name": "sales_memory", "entityType": "memory", "observations": ["Quarterly sales target review", "Client acquisition strategy discussion."]}' > data/g/sales.json
echo '{"type": "entity", "name": "engineering_memory", "entityType": "memory", "observations": ["Codebase refactoring plan", "New feature development roadmap."]}' > data/g/engineering.json
echo '{"type": "entity", "name": "marketing_memory", "entityType": "memory", "observations": ["Upcoming product launch campaign", "Social media engagement analysis."]}' > data/g/marketing.json
echo '{"type": "entity", "name": "admin_memory", "entityType": "memory", "observations": ["Admin'\''s personal notes", "System-wide announcements."]}' > data/u/admin.json
echo '{"type": "entity", "name": "bob_sales_memory", "entityType": "memory", "observations": ["Bob'\''s sales meeting notes", "Client follow-up reminders."]}' > data/u/bob.sales.json
echo '{"type": "entity", "name": "jane_marketing_memory", "entityType": "memory", "observations": ["Jane'\''s marketing ideas", "Campaign performance metrics."]}' > data/u/jane.marketing.json
echo '{"type": "entity", "name": "john_engineer_memory", "entityType": "memory", "observations": ["John'\''s engineering tasks", "Bug fixes and improvements."]}' > data/u/john.engineer.json
echo -e "${GREEN}✔${RESET} Sample memory files created"

echo -e "${INFO}🔄 Starting docker compose...${RESET}"
docker compose up -d --build

echo -e "${INFO}⏳ Waiting for services to be ready...${RESET}"
sleep 10

echo
echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}🎉 Memory MCP Server with Group Access is ready!${RESET}"
echo -e "${INFO}===========================================${RESET}"
echo
echo -e "Open the demo page:     ${INFO}https://$FRONTEND_DOMAIN${RESET}"
echo -e "Keycloak admin:         ${INFO}https://$KEYCLOAK_DOMAIN${RESET}"
echo -e "See memory API:         ${INFO}https://$FRONTEND_DOMAIN/api/mcp/memory/tools${RESET}"
echo
echo -e "${INFO}Test users configured in Keycloak:${RESET}"
echo -e "  • admin (password: admin123) - 	All group memories + personal"
echo -e "  • john.engineer (password: engineer123) - Engineering memories + personal"
echo -e "  • jane.marketing (password: marketing123) - Marketing memories + personal"  
echo -e "  • bob.sales (password: sales123) - Sales memories + personal"
echo
echo -e "${INFO}Group-based memory access:${RESET}"
echo -e "  • User-specific: /data/u/{user_id}.json (always accessible to the user)"
echo -e "  • Group-specific: /data/g/{group_id}.json (accessible only to group members)"
echo
echo -e "${INFO}Try these API calls:${RESET}"
echo -e "  • Start session with group access: POST /api/mcp/memory/session/start?group=finance"
echo -e "  • List tools: GET /api/mcp/memory/tools"
echo -e "  • Access user memory: POST /api/mcp/memory/tools/add_memory (without group)"
echo -e "  • Access group memory: POST /api/mcp/memory/tools/add_memory (with group session)"
echo
echo -e "${YELLOW}Note: This example demonstrates group-based data access patterns.${RESET}"
echo -e "${YELLOW}Each user can only access memory stores for groups they belong to.${RESET}"
