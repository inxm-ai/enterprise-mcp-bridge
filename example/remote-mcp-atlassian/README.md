# Atlassian MCP Workflow Demo

This example connects the Enterprise MCP Bridge to the Atlassian MCP server (Jira + Confluence) and ships with a workflow that checks sprint status and posts a summary to a "Status Update" Confluence page.

It provisions a Keycloak realm with an Atlassian OAuth2 identity provider (via an OAuth2 IdP plugin), funnels identities through `oauth2-proxy`, and forwards OpenAI-compatible chat traffic to the bridge using Docker Compose. The bridge talks to your Atlassian MCP server over HTTP and uses the Atlassian token for MCP requests.

## Topology

```
Browser --TLS--> oauth2-proxy --> Keycloak (local realm)
   |
   v
Enterprise MCP Bridge --> Atlassian MCP Server --> Jira + Confluence APIs
```

Optional: The stack ships with the same dummy LLM from the M365 example. Provide OpenAI-compatible credentials during setup to stream through your own model provider.

## Prerequisites

- Docker and Docker Compose
- `openssl` (certificate generation)
- A running Atlassian MCP server (see https://github.com/atlassian/atlassian-mcp-server)
- An Atlassian account with Jira + Confluence access (for the MCP server)
- A Keycloak OAuth2 IdP SPI jar compatible with 25.x (drop into `keycloak/providers/`)
- (Optional) An OpenAI-compatible service endpoint & API key

Note: The scripts can run on macOS, Linux, and WSL. If you are on WSL, remember to trust the generated CA certificate inside Windows for browser access.

## Quick start

1. Start the Atlassian MCP server and note its base URL (for example `http://localhost:8000`).
2. From this directory run:

   ```bash
   ./start.sh
   ```

3. Provide the Atlassian MCP server URL and optional bearer token when prompted. Choose an Atlassian OAuth client auth method (public client with `none` + PKCE, or confidential with `client_secret_post`/`client_secret_basic`). The script will attempt to dynamically register an Atlassian OAuth client; if that fails, enter a client id (and optional secret). Choose the dummy LLM or bring your own OpenAI-compatible endpoint.
4. Browse to `https://inxm.local` and log in via the Atlassian consent screen.
5. Call MCP tools from the UI or by hitting `https://inxm.local/api/mcp/atlassian/tools`.

When finished, stop everything with:

```bash
./stop.sh
```

This script tears down the Compose stack, cleans `/etc/hosts`, and removes the `.env` file (leaving certificate material untouched for future runs).

## Workflow example

The workflow definition lives at `workflows/sprint-status-update.json` with `flow_id: "sprint_status_update"`.

Example call (bypassing oauth2-proxy by hitting the bridge container directly):

```bash
curl -X POST http://localhost:8001/api/mcp/atlassian/tgi/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": "Check the sprint status for project ENG and update the Status Update page in the ENG space."
      }
    ],
    "use_workflow": "sprint_status_update"
  }'
```

If you prefer to route through the UI, log in via the browser and reuse the session cookie in your request to `https://inxm.local/api/mcp/atlassian/tgi/v1/chat/completions`.

## Environment variables

The `start.sh` wizard writes a `.env` file that Docker Compose consumes. You can edit and re-run the script at any time. Notable fields:

| Variable | Description |
| --- | --- |
| `ATLASSIAN_MCP_SERVER` | Base URL of the running Atlassian MCP server. Use `http://host.docker.internal:PORT` if it runs on your host. |
| `ATLASSIAN_MCP_BEARER_TOKEN` | Optional bearer token passed to the MCP server (leave empty if not required). |
| `ATLASSIAN_MCP_SCOPE` / `ATLASSIAN_MCP_CLIENT_ID` / `ATLASSIAN_MCP_CLIENT_SECRET` / `ATLASSIAN_MCP_REDIRECT_URI` | Optional OAuth settings if your MCP server expects them. |
| `ATLASSIAN_OIDC_CLIENT_ID` | Atlassian OAuth client id used by Keycloak for the brokered login. |
| `ATLASSIAN_OIDC_CLIENT_SECRET` | Optional secret for the Atlassian OAuth client (leave empty for public clients). |
| `ATLASSIAN_OIDC_CLIENT_AUTH_METHOD` | Token auth method for the Atlassian OAuth client (`none`, `client_secret_post`, or `client_secret_basic`). |
| `ATLASSIAN_OIDC_PKCE_ENABLED` | Enable PKCE for the brokered login (`true` for public clients, `false` for confidential). |
| `ATLASSIAN_OIDC_PKCE_METHOD` | PKCE method to use when enabled (default: `S256`). |
| `ATLASSIAN_IDP_SCOPE` | Scopes to request during registration and login (default: `read:me offline_access`). |
| `ATLASSIAN_IDP_USERINFO_URL` | User info URL for OAuth2 provider mode (default: `https://api.atlassian.com/me`). |
| `OAI_*` | Connection details for your OpenAI-compatible endpoint. Defaults to the bundled dummy LLM. |
| `TOKEN_NAME` | Header name that carries the Keycloak access token from oauth2-proxy (defaulted to `X-Forwarded-Access-Token` in this compose). |

Note: The Atlassian user info endpoint typically requires the `read:me` scope.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Browser cannot reach `https://inxm.local` | Ensure the dev CA is trusted and the `/etc/hosts` entries exist. Re-run `./start.sh` to recreate them. |
| Tools list is empty | Confirm the Atlassian MCP server is reachable from the container (`ATLASSIAN_MCP_SERVER`). If it runs on your host, use `http://host.docker.internal:PORT`. |
| Workflow cannot update Confluence | Check the Atlassian MCP server credentials, Confluence space permissions, and page title. |
| Keycloak fails with "Invalid identity provider id [oauth2]" | Drop a compatible OAuth2 IdP SPI jar into `keycloak/providers/` and restart. |
| Bridge logs show `Invalid token.` when calling `/broker/atlassian/token` | Ensure the user has the `broker:read-token` client role and re-login so the access token includes it. |
| Bridge logs still show `Invalid token.` after `broker:read-token` | Ensure the access token `aud` includes `broker` (the template adds an `audience-broker` mapper). Re-login after reimporting the realm. |
| Token retrieval still fails | Ensure the access token has `resource_access.broker.roles` (the template adds a `broker-roles` mapper). Re-login after reimporting the realm. |

Note: The Atlassian identity provider is defined in `keycloak/realm-export/realm-atlassian-template.json`. The OAuth2 IdP plugin requires an `identityScript` to map the profile; the template includes a placeholder script you should replace with real mapping logic. For opaque access tokens, the template disables user info lookups (`disableUserInfo: true`) to avoid JWT parsing errors; flip it to `false` if you want to use `userInfoUrl`. If you register the client manually, use redirect URI `https://auth.inxm.local/realms/inxm/broker/atlassian/endpoint`. Add a compatible OAuth2 IdP SPI jar into `keycloak/providers/` so Keycloak can load the broker.

## Cleanup reminders

- `./stop.sh` removes the `.env` file but does not delete your Atlassian credentials.
- Certificates under `../dev-local-certs` are reused across runs. Delete them if you no longer need the local CA.

Happy hacking with Atlassian MCP workflows!
