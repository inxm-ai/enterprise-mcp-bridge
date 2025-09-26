# GitHub Remote MCP Demo

This example mirrors the Microsoft 365 token exchange demo, but instead of running a local MCP server it connects the Enterprise MCP Bridge to the **GitHub Remote MCP service** (`https://api.githubcopilot.com/mcp/`).

It provisions a local Keycloak realm that brokers authentication to GitHub, funnels identities through `oauth2-proxy`, and forwards OpenAI-compatible chat traffic to the bridge using Docker Compose.

## Topology

```
Browser ──TLS──▶ oauth2-proxy ──▶ Keycloak (GitHub IdP)
   │                                     │
   │                        GitHub OAuth  │
   ▼                                     ▼
Enterprise MCP Bridge ──▶ GitHub Remote MCP API
```

Optional: The stack ships with the same dummy LLM from the M365 example. Provide OpenAI-compatible credentials during setup to stream through your own model provider.

## Prerequisites

- Docker and Docker Compose
- `openssl` (certificate generation)
- `envsubst` (usually provided by `gettext`)
- A GitHub OAuth App **or** GitHub App configured with OAuth credentials
  - Authorization callback URL **must** be `https://ghmcp.local/oauth2/callback`
  - Recommended scopes: `read:user`, `user:email`, `copilot:chat`, plus any tool-specific scopes you plan to expose
- (Optional) An OpenAI-compatible service endpoint & API key

> ℹ️ The scripts can run on macOS, Linux, and WSL. If you are on WSL, remember to trust the generated CA certificate inside Windows for browser access.

## Quick start

1. Create or identify a GitHub OAuth application.
   - Homepage URL: `https://ghmcp.local`
   - Authorization callback URL: `https://ghmcp.local/oauth2/callback`
   - Note the **Client ID** and **Client Secret**.
2. From this directory run:

   ```bash
   ./start.sh
   ```

3. Provide the GitHub client ID/secret and scope list when prompted. Choose the dummy LLM or bring your own OpenAI-compatible endpoint.
4. Browse to `https://ghmcp.local` and initiate a login. Keycloak will redirect you to GitHub and mint an access token for the bridge.
5. Call MCP tools from the UI or by hitting `https://ghmcp.local/api/mcp/github/tools`.

When finished, stop everything with:

```bash
./stop.sh
```

This script tears down the Compose stack, cleans `/etc/hosts`, deletes the rendered Keycloak realm, and removes the `.env` file (leaving certificate material untouched for future runs).

## Environment variables

The `start.sh` wizard writes a `.env` file that Docker Compose consumes. You can edit and re-run the script at any time. Notable fields:

| Variable | Description |
| --- | --- |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | OAuth credentials used by Keycloak to authenticate against GitHub. |
| `GITHUB_OAUTH_SCOPES` | Space-separated scopes requested from GitHub. These are also forwarded to the remote MCP server as `MCP_REMOTE_SCOPE`. |
| `MCP_REMOTE_HEADER_X_MCP_READONLY` | Sends `X-MCP-Readonly: true` by default. Set to `false` to request write permissions. |
| `MCP_REMOTE_HEADER_X_MCP_WORKSPACE_ID` | Optional header to scope the remote MCP workspace. Leave blank to omit. |
| `MCP_REMOTE_REDIRECT_URI` | Callback URI shared with Keycloak and GitHub. Defaults to `https://ghmcp.local/oauth2/callback`. |
| `OAI_*` | Connection details for your OpenAI-compatible endpoint. Defaults to the bundled dummy LLM. |

Any variable prefixed with `MCP_REMOTE_HEADER_` is automatically translated into an HTTP header when the bridge talks to the remote server. This makes it easy to forward custom GitHub headers (for example `MCP_REMOTE_HEADER_X_MCP_ORGANIZATION_ID`).

## Updating the Keycloak realm

- The realm template lives at `keycloak/realm-export/realm-ghmcp-template.json`.
- `start.sh` renders the template to `realm-ghmcp.json` via `envsubst`.
- If you add new placeholders, re-run `./start.sh` to refresh the realm file before restarting the stack.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Browser cannot reach `https://ghmcp.local` | Ensure the dev CA is trusted and the `/etc/hosts` entries exist. Re-run `./start.sh` to recreate them. |
| GitHub login loops back to Keycloak | Verify the OAuth callback matches `https://ghmcp.local/oauth2/callback` exactly and that the GitHub scopes are valid. |
| Remote MCP returns 403 | Set `MCP_REMOTE_HEADER_X_MCP_READONLY=false` or provide additional headers like `MCP_REMOTE_HEADER_X_MCP_WORKSPACE_ID`. |
| Tools list is empty | Confirm your GitHub account has access to the requested MCP workspace and that scopes cover the tools you expect. |

## Cleanup reminders

- `./stop.sh` removes the `.env` file but **does not** delete your GitHub OAuth App. Rotate or delete those credentials manually.
- Certificates under `../dev-local-certs` are reused across runs. Delete them if you no longer need the local CA.

Happy hacking with remote MCP! ✨
