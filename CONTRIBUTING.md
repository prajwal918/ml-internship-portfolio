# Contributing to ML Internship Portfolio

We welcome contributions! Please review the guidelines below.

## Code Standards
- **Conventional Commits**: All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification (e.g., `feat(core): implement resilient architecture`).
- **Resilient Engineering**: All entry points must include robust `try-catch` exception handling. We do not tolerate "happy paths" in production environments.
- **Structural Integrity**: Code lives in `src/` and tests live in `tests/`.

## Workflow
1. Fork and clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Make your changes in a feature branch.
4. Run tests and linting.
5. Create a Pull Request outlining your changes.
