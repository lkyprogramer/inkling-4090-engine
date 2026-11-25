# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues. Use GitHub's private vulnerability reporting ("Report a vulnerability" under the repository's **Security** tab) to reach the maintainers confidentially.

Please include a description of the issue, the affected version (`astral version`), steps to reproduce and the impact you expect. We aim to acknowledge reports within five working days.

## Scope

The HTTP server (`astral serve`) is designed to run behind a trusted gateway. It has no authentication or TLS; do not expose it directly to untrusted networks.
