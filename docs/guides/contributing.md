# Contributing Guide

Welcome to the MLX Distributed Training Framework! We're excited that you want to contribute.

## Getting Started

1. **Fork the Repository**
   - Visit the GitHub repository
   - Click the "Fork" button to create your own copy

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/your-username/mlx-train.git
   cd mlx-train
   ```

3. **Set Up Development Environment**

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a Feature Branch**

   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make Your Changes**
   - Write clean, documented code
   - Follow our coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests**

   ```bash
   pytest tests/
   ```

4. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

5. **Push to GitHub**

   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Visit your fork on GitHub
   - Click "New Pull Request"
   - Describe your changes in detail
   - Link any related issues

## Code Standards

- Use type hints for better IDE support
- Follow Google Python Style Guide
- Document new functions and classes
- Include docstring examples
- Add unit tests for new features

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Include integration tests where appropriate
- Test on multiple Apple Silicon devices if possible

## Documentation

- Update relevant documentation
- Add docstrings to new code
- Include examples for new features
- Update API reference if needed

## Pull Request Guidelines

1. **Description**
   - Clearly describe the changes
   - Link related issues
   - List any breaking changes

2. **Code Quality**
   - Follow style guidelines
   - Include appropriate tests
   - Update documentation

3. **Review Process**
   - Address reviewer feedback
   - Keep PR scope focused
   - Maintain clean commit history

Thank you for contributing to our MLX Distributed Training Framework! 🚀
