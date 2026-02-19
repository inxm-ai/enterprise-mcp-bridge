# Changelog

All notable changes to Enterprise MCP Bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation site using Docsify
- Diataxis-based documentation structure (Tutorials, How-To, Reference, Explanation)
- GitHub Pages deployment workflow

## [0.4.2] - 2024-01-15

### Added
- Session management with automatic cleanup
- Group-based data access control
- Token exchange for downstream services
- Remote MCP server support
- OpenTelemetry integration for observability

### Changed
- Improved error handling and HTTP status codes
- Enhanced OAuth token validation
- Better session lifecycle management

### Fixed
- Session timeout edge cases
- Token refresh timing issues
- Memory leaks in long-running sessions

## [0.4.0] - 2024-01-01

### Added
- Multi-user session support
- OAuth2 authentication integration
- Redis-based session manager
- Automatic token injection
- Prometheus metrics endpoint

### Changed
- Migrated to FastAPI from Flask
- Improved MCP protocol handling
- Enhanced security model

## [0.3.0] - 2023-12-01

### Added
- Basic session management
- Tool discovery and invocation
- Docker support
- Auto-generated API documentation

### Changed
- Simplified configuration
- Improved process management

## [0.2.0] - 2023-11-01

### Added
- Initial FastAPI implementation
- MCP server wrapper
- REST endpoint mapping

## [0.1.0] - 2023-10-01

### Added
- Initial release
- Basic MCP protocol support
- Simple REST interface

---

## Release Notes

### Version 0.4.2

This release focuses on production readiness and enterprise features:

**Highlights:**
- 🔒 Enhanced security with OAuth2 token exchange
- 📊 Comprehensive observability with OpenTelemetry
- 🔄 Remote MCP server support for distributed architectures
- 👥 Group-based access control for multi-tenant deployments
- 📝 Complete documentation overhaul

**Migration Guide:**

From 0.3.x to 0.4.x:
1. Update environment variables for new OAuth configuration
2. Migrate to Redis session manager for production deployments
3. Review and update security configurations
4. Test group-based access if using multi-tenancy

**Breaking Changes:**
- Session manager configuration changed from `REDIS_ENABLED` to `SESSION_MANAGER_TYPE`
- OAuth configuration restructured (see documentation)

**Deprecations:**
- None in this release

For detailed migration instructions, see the [Migration Guide](https://github.com/inxm-ai/enterprise-mcp-bridge/blob/main/MIGRATION.md).

---

## Contributing

See [CONTRIBUTING.md](contributing.md) for how to contribute changes and additions to this changelog.

## Links

- [Repository](https://github.com/inxm-ai/enterprise-mcp-bridge)
- [Issues](https://github.com/inxm-ai/enterprise-mcp-bridge/issues)
- [Releases](https://github.com/inxm-ai/enterprise-mcp-bridge/releases)
