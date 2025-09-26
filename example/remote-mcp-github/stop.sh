#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v docker >/dev/null || { echo -e "${RED}❌ docker required${RESET}"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FRONTEND_DOMAIN=ghmcp.local
KEYCLOAK_DOMAIN=auth.ghmcp.local
ENV_FILE=.env

if [ ! -f "$ENV_FILE" ]; then
  echo -e "${YELLOW}⚠️ .env file not found. Continuing with Docker cleanup only.${RESET}"
else
  source "$ENV_FILE"
fi

echo -e "${INFO}🔄 Stopping Docker containers...${RESET}"
docker compose down -t 0 || { echo -e "${RED}❌ Failed to stop Docker containers.${RESET}"; exit 1; }

echo -e "${INFO}🔄 Cleaning up /etc/hosts...${RESET}"
for host in "$FRONTEND_DOMAIN" "$KEYCLOAK_DOMAIN"; do
  if grep -Eq "\b$host\b" /etc/hosts; then
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null; then
        echo -e "${YELLOW}⚠️ Removing hosts entry for $host (requires sudo)...${RESET}"
        if sudo sed -i "/\b$host\b/d" /etc/hosts; then
          echo -e "${GREEN}✔ Removed hosts entry for $host${RESET}"
        else
          echo -e "${RED}❌ WARNING: Failed to remove hosts entry via sudo. Remove manually.${RESET}" >&2
        fi
      else
        echo -e "${RED}❌ WARNING: sudo not available. Remove this line manually: $host${RESET}" >&2
      fi
    else
      sed -i "/\b$host\b/d" /etc/hosts
      echo -e "${GREEN}✔ Removed hosts entry for $host${RESET}"
    fi
  else
    echo -e "${INFO}ℹ️ Hosts entry for $host not found${RESET}"
  fi
done

if [ -f "$ENV_FILE" ]; then
  rm -f "$ENV_FILE"
  echo -e "${GREEN}✔ Removed local .env${RESET}"
fi

rm -f keycloak/realm-export/realm-ghmcp.json || true

echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}🎉 Cleanup complete${RESET}"
echo -e "${INFO}===========================================${RESET}"
echo -e "Remember to delete or rotate the GitHub OAuth app credentials if they were only created for this demo."
