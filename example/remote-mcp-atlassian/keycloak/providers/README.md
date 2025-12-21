# OAuth2 Broker Plugin

Keycloak does not ship a generic OAuth2 identity provider by default. To use
`ATLASSIAN_IDP_PROVIDER=oauth2`, you must supply a Keycloak OAuth2 IdP SPI plugin
JAR that provides the `oauth2` identity provider id.

Place the compatible JAR in this directory (mounted into `/opt/keycloak/providers`).
Restart the stack after adding the plugin.

Notes:
- The plugin must be compatible with Keycloak 25.x.
- Look for an OAuth2 identity provider SPI that advertises provider id `oauth2`.
