#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
RESET="\033[0m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
INFO="\033[0;36m"

command -v docker >/dev/null || { echo -e "${RED}ERROR:${RESET} docker required"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FRONTEND_DOMAIN=inxm.local
KEYCLOAK_DOMAIN=auth.inxm.local
ENV_FILE=.env

if [ ! -f "$ENV_FILE" ]; then
  echo -e "${YELLOW}WARN:${RESET} .env file not found. Continuing with Docker cleanup only."
else
  source "$ENV_FILE"
fi

echo -e "${INFO}Stopping Docker containers...${RESET}"
docker compose down -t 0 || { echo -e "${RED}ERROR:${RESET} Failed to stop Docker containers."; exit 1; }

echo -e "${INFO}Cleaning up /etc/hosts...${RESET}"
for host in "$FRONTEND_DOMAIN" "$KEYCLOAK_DOMAIN"; do
  if grep -Eq "\b$host\b" /etc/hosts; then
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null; then
        echo -e "${YELLOW}WARN:${RESET} Removing hosts entry for $host (requires sudo)..."
        if sudo sed -i "/\b$host\b/d" /etc/hosts; then
          echo -e "${GREEN}OK${RESET} Removed hosts entry for $host"
        else
          echo -e "${RED}WARN:${RESET} Failed to remove hosts entry via sudo. Remove manually." >&2
        fi
      else
        echo -e "${RED}WARN:${RESET} sudo not available. Remove this line manually: $host" >&2
      fi
    else
      sed -i "/\b$host\b/d" /etc/hosts
      echo -e "${GREEN}OK${RESET} Removed hosts entry for $host"
    fi
  else
    echo -e "${INFO}Hosts entry for $host not found${RESET}"
  fi
done

if [ -f "$ENV_FILE" ]; then
  rm -f "$ENV_FILE"
  echo -e "${GREEN}OK${RESET} Removed local .env"
fi

echo -e "${INFO}===========================================${RESET}"
echo -e "${GREEN}OK${RESET} Cleanup complete"
echo -e "${INFO}===========================================${RESET}"
