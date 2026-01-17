# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.9.x   | :white_check_mark: |
| 1.8.x   | :x:                |
| < 1.8   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LLMTeam, please report it responsibly.

### How to Report

1. **Email**: Send details to LLMTeamai@gmail.com
2. **Subject**: Use "[SECURITY] LLMTeam Vulnerability Report"
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release

### Security Features

LLMTeam includes several security features by design:

1. **Multi-Tenant Isolation** (v1.7.0+)
   - Complete data separation between tenants
   - TenantContext enforcement
   - TenantIsolatedStore pattern

2. **Audit Trail** (v1.7.0+)
   - SHA-256 checksum chain for integrity
   - Immutable AuditRecord
   - Compliance-ready logging

3. **Agent Context Security** (v1.7.0+)
   - Horizontal isolation: Agents cannot see each other's contexts
   - Vertical visibility: Parent-to-child only
   - SealedData: Owner-only access

4. **Rate Limiting** (v1.7.0+)
   - Circuit breaker pattern
   - Request throttling
   - Resource protection

5. **Instance Namespacing** (v2.0.0+)
   - Workflow instances isolated within tenant
   - RuntimeContext scoping

## Security Best Practices

When using LLMTeam:

1. **Never hardcode secrets** - Use RuntimeContext.get_secret()
2. **Enable audit logging** - For compliance and forensics
3. **Use tenant isolation** - Even for single-tenant deployments
4. **Validate all inputs** - Especially in custom step handlers
5. **Review edge conditions** - In segment definitions
6. **Limit LLM permissions** - Follow least-privilege principle

## Disclosure Policy

- We follow coordinated disclosure
- Security patches are released as soon as possible
- CVE identifiers are requested for significant vulnerabilities
- Security advisories are published on GitHub
