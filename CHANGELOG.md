# Changelog

All notable changes to treval.

## [0.2.3] - 2026-06-10

### Added
- GitHub repository public
- LICENSE (MIT)
- CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
- Badges in README (PyPI, Python, License)
- Issue and PR templates
- `.gitignore`

### Changed
- Template agent translated to English
- Installation docs use `pip install treval`
- pyproject.toml author and URLs updated

## [0.2.2] - 2026-06-09

### Added
- OpenTelemetry exporter
- `treval replay` command
- Agent test suites (`treval test`)
- Dashboard HTML improvements (favicon, responsive)

## [0.2.1] - 2026-06-08

### Added
- Multi-model comparison with statistics
- LLM-as-judge evaluation
- API cost tracking
- HTML comparison reports

## [0.2.0] - 2026-06-07

### Added
- `@agent`, `@tool`, `@operation` decorators
- Auto-instrumentation (`treval.instrument()`)
- SQLite span store
- CLI with 15 commands
- Gateway proxy for HTTP interception
- Web dashboard
- First PyPI release
