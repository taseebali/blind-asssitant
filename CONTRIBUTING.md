# Contributing to Blind Assistant

Thank you for your interest in contributing to Blind Assistant! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - OS and Python version
   - GPU model and CUDA version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Features

1. Check existing feature requests
2. Explain the use case
3. Describe the proposed solution
4. Consider alternatives

### Pull Requests

1. **Fork** the repository
2. **Create** a branch (`git checkout -b feature/AmazingFeature`)
3. **Make** your changes
4. **Test** thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Open** a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/blind-assistant.git
cd blind-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/
```

## Coding Standards

### Python Style Guide

Follow PEP 8:
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Code Quality

- Write clear, self-documenting code
- Add docstrings to functions and classes
- Include type hints
- Keep functions small and focused
- Avoid deep nesting

### Example

```python
from typing import List, Tuple
import numpy as np

def detect_objects(frame: np.ndarray, threshold: float = 0.30) -> Tuple[List[dict], np.ndarray]:
    """
    Detect objects in the given frame.
    
    Args:
        frame: Input image as numpy array
        threshold: Confidence threshold for detections
    
    Returns:
        Tuple of (detections list, annotated frame)
    
    Raises:
        ValueError: If frame is invalid
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")
    
    # Implementation
    ...
    
    return detections, annotated_frame
```

## Testing

### Writing Tests

```python
import pytest
from src.services.detection.object_detector import ObjectDetector

def test_object_detector_initialization():
    """Test detector can be initialized"""
    detector = ObjectDetector()
    assert detector is not None

def test_object_detection():
    """Test object detection with sample image"""
    detector = ObjectDetector()
    frame = cv2.imread('tests/test_data/test_image.jpg')
    objects, _ = detector.detect(frame)
    assert len(objects) > 0
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_object_detector.py

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When this happens
        TypeError: When that happens
    
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

### README Updates

- Update main README for new features
- Add module READMEs for new packages
- Include code examples
- Update troubleshooting section

## Commit Messages

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(detection): add distance estimation to object detector

- Calculate relative depth based on object size
- Add position tracking (left/center/right)
- Update narration service to use depth info

Closes #123
```

## Project Structure

When adding new features:

```
src/
├── services/
│   ├── your_service/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── your_module.py
│   └── ...
tests/
├── test_your_service.py
└── ...
```

## Pull Request Process

1. **Update documentation**
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG** (if applicable)
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Descriptive commit messages

## Review Process

Maintainers will:
1. Review code quality
2. Check test coverage
3. Verify documentation
4. Test functionality
5. Provide feedback

## Questions?

- Open an issue for discussion
- Join our community chat
- Email: taseebali@example.com

## Recognition

Contributors will be added to:
- README acknowledgments
- CONTRIBUTORS.md
- Release notes

Thank you for contributing! 🎉
