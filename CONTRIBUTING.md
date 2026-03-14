# Contributing to RiichiEnv

Thank you for your interest in contributing to RiichiEnv! This document provides guidelines and information to help you get started.

We welcome valuable contributions, even if they are AI-generated. You, as a human, are responsible for judging whether your contribution is valuable.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Reporting Bugs](#reporting-bugs)
- [License](#license)

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

## How to Contribute

There are many ways to contribute:

- **Report bugs** — If you find a bug, please open an issue with a clear description and steps to reproduce
- **Suggest features** — Have an idea? Open an issue to discuss it
- **Submit pull requests** — Bug fixes, new features, documentation improvements are all welcome
- **Review pull requests** — Help review other contributors' pull requests
- **Improve documentation** — Fix typos, clarify explanations, or add missing information

## Development Setup

Prerequisites, build instructions, testing, linting, benchmarks, and other development details are documented in **[docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)**.

Before submitting a pull request, make sure all pre-commit checks pass:

```bash
uv run pre-commit run --config .pre-commit-config.yaml
```

> But honestly, I forget to run this all the time and end up failing CI myself — so don't worry if it happens to you too.

## Reporting Bugs

When reporting a bug, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Rust version, Python version)
- Any relevant logs or error messages

But honestly, you don't have to include all of the above. Skip anything you think isn't relevant. If we need more info, we'll ask. Feel free to report casually!

## License

By contributing to RiichiEnv, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).