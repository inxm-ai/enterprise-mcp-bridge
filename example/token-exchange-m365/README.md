# Standalone Getting Started Example

This is a complete demo including authentication with m365

## Architecture

```mermaid
graph TD
    subgraph Auth
        Keycloak["Keycloak (auth.inxm.local)"]
        OAuth2Proxy["OAuth2-Proxy (inxm.local)"]
        Redis[Redis]
    end

    subgraph Ops
        Jaeger[Jaeger]
        Grafana[Grafana]
        Prometheus[Prometheus]
    end

    subgraph Apps
        AppNginx[App-Nginx]
        AppFrontend[App-Frontend]
        AppMCPRest[enterprise-mcp-bridge]
    end

    Entra[Microsoft Entra]
    GraphAPI[Microsoft Graph API]

    User --> Ingress
    Ingress --[inxm.local]--> OAuth2Proxy
    Ingress --[auth.inxm.local]--> Keycloak
    AppNginx --> AppFrontend
    AppNginx --> AppMCPRest
    AppMCPRest --> OAuth2Proxy
    AppMCPRest --request provider token--> Keycloak
    OAuth2Proxy --> Keycloak
    OAuth2Proxy --> Redis
    OAuth2Proxy --> AppNginx
    Keycloak --> Entra
    Entra --> Keycloak
    AppMCPRest --request with provider token--> GraphAPI

    Prometheus --> AppMCPRest
    Jaeger --> AppMCPRest
    Grafana --> Prometheus
```

## Login Flow

```mermaid
sequenceDiagram
    participant User
    participant ApplicationIngress as Application-Ingress
    participant Keycloak as Keycloak (auth.inxm.local)
    participant OAuth2Proxy as OAuth2-Proxy (inxm.local)
    participant AppNginx as App-Nginx
    participant AppFrontend as App-Frontend
    participant AppMCPRest as App-MCP-Rest

    User ->> ApplicationIngress: Access Application
    ApplicationIngress ->> Keycloak: Authenticate (auth.inxm.local)
    ApplicationIngress ->> OAuth2Proxy: Authenticate (inxm.local)
    OAuth2Proxy ->> AppNginx: Forward Request
    AppNginx ->> AppFrontend: Serve Frontend
    AppNginx ->> AppMCPRest: Forward API Request
```

## Request Graph API flow

```mermaid 
sequenceDiagram
    participant OAuth2Proxy as OAuth2-Proxy
    participant AppMCPRest as enterprise-mcp-bridge
    participant Keycloak as Keycloak
    participant Entra as Microsoft Entra
    participant GraphAPI as Microsoft Graph API

    OAuth2Proxy ->> AppMCPRest: Provide Auth Token
    AppMCPRest ->> Keycloak: Request Microsoft Token with Auth Token
    Keycloak ->> Entra: Exchange for Entra Token
    Keycloak -->> AppMCPRest: Return Entra Token
    AppMCPRest ->> GraphAPI: Access Microsoft Graph API with Entra Token
```


## What it Provides
* Keycloak with token-exchange feature and ingress
* Automated Entra (Azure AD) app registration
* Enterprise MCP Bridge launched with `npx -y @softeria/ms-365-mcp-server --org-mode`
* Minimal frontend
* Tracing via Jaeger
* Monitoring via Prometheus

## Prerequisites
* Docker & Docker Compose [link](https://docs.docker.com/engine/install/)
* Azure CLI (`az`) [link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
* A free Entra test account (create it [here](https://learn.microsoft.com/sk-sk/entra/verified-id/how-to-create-a-free-developer-account))
* A valid api token for the openai api (or any ai provider using the openai api standard)

## Run

```bash
./start.sh
```

Then open https://inxm.local

## Cleanup

```bash
./stop.sh
```
