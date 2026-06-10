# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in treval, please report it privately:

- **Email:** [tu-email si quieres, o quitar esta línea]
- **GitHub:** Use the [private vulnerability reporting](https://github.com/AmOrFeU86/treval/security/advisories/new) feature

Do not open public issues for security vulnerabilities.

## What to Include

A useful report includes:

- Description of the vulnerability and severity
- Steps to reproduce
- Affected component (file path + line range)
- Environment details (treval version, Python version, OS)

## Scope

Treval is an observability and evaluation framework. Security concerns include:

- **API key exposure** — treval never logs or stores API keys
- **Code injection** — eval of user-provided expressions
- **Gateway security** — the proxy gateway binds to localhost by default
- **HTML injection** — dashboard-generated HTML with user data

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅ |
| < 0.2   | ❌ |
