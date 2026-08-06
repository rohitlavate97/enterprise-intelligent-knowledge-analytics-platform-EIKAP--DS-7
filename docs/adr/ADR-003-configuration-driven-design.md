# Architecture Decision Record

## Title: Configuration-Driven Design
## Status: Accepted

## Context
EIKAP operates in multiple environments and has many tunable parameters across its infrastructure and modules.

## Decision
We will use YAML configuration files supplemented with environment variable interpolation. All configurations will be validated by Pydantic models at runtime startup.

## Consequences
- No hardcoded configuration values scattered throughout the codebase.
- Simple, predictable overrides for environment-specific settings.
- Type-safe configuration with early failure on missing or malformed variables.
