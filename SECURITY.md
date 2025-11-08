# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Eigen Geometric Control seriously. If you discover a security vulnerability, please follow these steps:

### Please Do:

1. **Email us privately** at mcreynolds.jon@gmail.com with:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (if you have them)

2. **Give us time to fix it** - We aim to respond within 48 hours and will work with you to:
   - Confirm the vulnerability
   - Develop and test a fix
   - Release a security patch
   - Publicly credit you (if desired)

3. **Keep it confidential** until we've released a fix

### Please Don't:

- ❌ Open a public GitHub issue about the vulnerability
- ❌ Disclose the vulnerability publicly before we've had a chance to fix it
- ❌ Exploit the vulnerability in production systems

## Security Best Practices for Users

When using this library:

1. **Keep dependencies updated** - Run `pip install --upgrade eigen-geometric-control` regularly
2. **Review configuration files** - Ensure your `config.yaml` doesn't contain sensitive data
3. **Validate inputs** - Always validate robot control inputs before passing to the control system
4. **Monitor system behavior** - Watch for unexpected robot movements or behaviors

## Security Features

Our codebase includes:

- ✅ **Automated security scanning** with Bandit in CI/CD
- ✅ **Dependency vulnerability scanning** via Dependabot
- ✅ **Input validation** on control parameters
- ✅ **Safe numerical operations** to prevent overflow/underflow
- ✅ **Configuration validation** with schema checking

## Disclosure Policy

When we release a security patch:

1. We'll publish a GitHub Security Advisory
2. Update the CHANGELOG with security fix details
3. Credit the reporter (if they wish to be credited)
4. Notify users via GitHub releases

## Contact

For security issues: mcreynolds.jon@gmail.com

For general questions: Open a [GitHub Issue](https://github.com/InauguralPhysicist/Eigen-Geometric-Control/issues)

---

**Thank you for helping keep Eigen Geometric Control and our users safe!**
