# Contributing to FlipItNews NLP Project

First off, thank you for considering contributing to FlipItNews! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project and everyone participating in it is governed by respect and professionalism. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any similar features in other projects**

### Your First Code Contribution

Unsure where to begin? You can start by looking through these issues:

- **Beginner issues** - issues which should only require a few lines of code
- **Help wanted issues** - issues which should be a bit more involved

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

1. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NLP-Word-Embedding-FlipItNews.git
   cd NLP-Word-Embedding-FlipItNews
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Run tests**
   ```bash
   streamlit run app.py
   ```

## Pull Request Process

1. **Update the README.md** with details of changes if applicable
2. **Update the documentation** for any new features
3. **Add tests** for new functionality
4. **Ensure all tests pass** before submitting
5. **Update the requirements.txt** if you add dependencies
6. **Follow the code style guidelines** below
7. **Write clear commit messages** (see below)

### PR Checklist

- [ ] Code follows the style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Dependencies updated if needed

## Style Guidelines

### Python Code Style

We follow PEP 8 style guidelines with some modifications:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group imports (standard library, third-party, local)
- **Naming conventions**:
  - Functions: `lowercase_with_underscores`
  - Classes: `CapitalizedWords`
  - Constants: `UPPERCASE_WITH_UNDERSCORES`
  - Variables: `lowercase_with_underscores`

### Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

class TextClassifier:
    """A class for text classification."""
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
    
    def train(self, X_train, y_train):
        """Train the classification model."""
        # Implementation here
        pass
```

### Documentation Style

- Use docstrings for all public modules, functions, classes, and methods
- Follow Google style docstrings
- Include type hints where applicable

```python
def classify_text(text: str, model: object) -> str:
    """
    Classify a text article into a category.
    
    Args:
        text: The article text to classify
        model: The trained classification model
        
    Returns:
        The predicted category as a string
        
    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")
    return model.predict([text])[0]
```

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(models): add BERT-based classification model

Implemented a new BERT-based classifier that achieves 93% accuracy
on the test set. Includes preprocessing pipeline and evaluation metrics.

Closes #123
```

```
fix(ui): resolve prediction error for SGD classifier

Fixed the predict_proba AttributeError by implementing decision
function visualization for SGD classifier.

Fixes #456
```

## Testing Guidelines

### Running Tests

```bash
# Run Streamlit app
streamlit run app.py

# Run Python script
python FLIPLTNews_Word_Embedding_Word2Vec_2.py

# Run Jupyter notebook
jupyter notebook FLIPLTNews_Word_Embedding_Word2Vec_2.ipynb
```

### Writing Tests

- Test all new features
- Test edge cases
- Test error handling
- Maintain test coverage above 80%

## Questions?

Feel free to contact the maintainer:

- **Email**: ratneshkumar181998@gmail.com
- **LinkedIn**: [linkedin.com/in/ratnesh-kumar-181998](https://www.linkedin.com/in/ratnesh-kumar-181998)
- **GitHub**: [@Ratnesh-181998](https://github.com/Ratnesh-181998)

## Recognition

Contributors will be recognized in the README.md file and in release notes.

---

Thank you for contributing to FlipItNews! 🚀
