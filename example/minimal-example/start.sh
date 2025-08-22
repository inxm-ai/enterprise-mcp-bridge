#!/bin/bash
set -e
echo "Default mcp example"

echo "Starting docker container"
server=$(docker run -d -p 8000:8000 ghcr.io/inxm-ai/mcp-rest-server)
sleep 5
echo "Calling the add tool with {\"a\": 2, \"b\": 1}"
curl -X 'POST' \
  'http://127.0.0.1:8000/tools/add' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "a": 2,
  "b": 1
}'
echo
server=$(docker rm -f $server)