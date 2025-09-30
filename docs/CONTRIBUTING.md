# Contributing to ASOPA

Thank you for your interest in contributing to ASOPA! This document provides guidelines for the contribution process.

## Table of Contents
- [Development Workflow](#development-workflow)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)

## Development Workflow

### Getting Started

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** following the coding standards (see [Development Guide](DEVELOPMENT_GUIDE.md#coding-standards))
4. **Add tests** for new functionality (see [Testing](DEVELOPMENT_GUIDE.md#testing))
5. **Update documentation** as needed
6. **Run test suite**: `pytest`
7. **Submit pull request**

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements
- `build/description` - Build system changes

### Development Environment

For detailed setup instructions, including DevPod configuration, environment variables, and development commands, see the [Development Guide](DEVELOPMENT_GUIDE.md#getting-started).

## Code Review Process

### Before Submitting

- [ ] Code follows PEP 8 style guide (see [Coding Standards](DEVELOPMENT_GUIDE.md#coding-standards))
- [ ] All tests pass (see [Testing](DEVELOPMENT_GUIDE.md#testing))
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No hardcoded values
- [ ] Error handling is implemented
- [ ] Performance is acceptable
- [ ] Type hints are added where appropriate
- [ ] Code is properly commented

### Review Checklist

- [ ] Code is readable and well-commented
- [ ] Architecture is sound
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Performance impact is considered
- [ ] Backward compatibility is maintained
- [ ] Security implications are considered
- [ ] Memory usage is reasonable

### Review Process

1. **Automated checks** run on all pull requests
2. **Code review** by at least one maintainer
3. **Testing** in the development environment
4. **Documentation review** for completeness
5. **Approval** and merge

## Release Process

### Version Management

1. **Update version numbers** in relevant files
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** to ensure stability
4. **Update documentation** for new features
5. **Create release tag** following semantic versioning
6. **Publish release notes** with highlights

### Semantic Versioning

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are updated
- [ ] Release notes are prepared
- [ ] Tag is created
- [ ] Release is published

