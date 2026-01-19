# Contributing to Human Activity Detection

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/human_activity_detection.git`
3. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

### Project Structure

- `src/` - Main source code
- `config/` - Configuration files
- `scripts/` - Utility scripts
- `docs/` - Documentation

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes thoroughly
4. Update documentation if needed
5. Commit with clear messages: `git commit -m "Add feature: description"`

### Testing

Before submitting:

- Run `python scripts/quick_start.py` to verify setup
- Test training with a small dataset
- Test inference on sample images/videos
- Check for linting errors

### Submitting Changes

1. Push to your fork: `git push origin feature/your-feature-name`
2. Create a Pull Request on GitHub
3. Provide a clear description of changes
4. Reference any related issues

## Areas for Contribution

- Model improvements
- Performance optimizations
- Documentation enhancements
- Bug fixes
- New features
- Test coverage

## Questions?

Open an issue on GitHub for questions or discussions.

Thank you for contributing! 🎉
