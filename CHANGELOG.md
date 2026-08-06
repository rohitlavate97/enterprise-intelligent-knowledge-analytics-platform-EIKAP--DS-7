# Changelog

All notable changes to EIKAP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-06

### Added
- Initial project structure and repository setup
- EIKAPModule plugin interface with maturity labels (standard/restricted)
- Universal Module Contract with 10 automated compliance checks
- Module Compliance Test Suite skeleton
- Shared infrastructure: configuration, logging, exceptions, DI container
- Docker Compose setup with PostgreSQL, Redis, MLflow server
- GitHub Actions CI pipeline with lint, test, and compliance jobs
- Pre-commit hooks (ruff, mypy, trailing whitespace)
- Developer tooling (Makefile, .env.example)
- Initial ADRs for key architecture decisions
