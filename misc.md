# Stata

Stata is a Python package that provides a robust, scalable framework for building and deploying machine learning models for various prediction tasks. It supports classification, regression, and includes comprehensive feature engineering capabilities.

## Features

- Support for multiple ML algorithms (Random Forest, Gradient Boosting, etc.)
- Automated feature engineering and preprocessing
- Cross-validation and model evaluation
- Comprehensive logging and model tracking
- Docker support for development and deployment
- Extensive test coverage
- Type hints and documentation

## Installation

### Using pip

```bash
pip install stata
```

### Development Installation

```bash
git clone https://github.com/tsj7ww/stata.git
cd stata
pip install -e ".[dev]"
```

## Quick Start

```python
from predml.models import ClassificationModel
from predml.preprocessing import FeatureEngineer
from predml.training import ModelTrainer

# Initialize components
model = ClassificationModel(model_type="random_forest")
feature_engineer = FeatureEngineer(
    numeric_features=['feature1', 'feature2'],
    categorical_features=['category1']
)
trainer = ModelTrainer(model, feature_engineer)

# Train model
metrics = trainer.train(data, target_column='target', stratify=True)

# Make predictions
predictions = model.predict(new_data)
```

## Project Structure

```
stata/
├── src/
│   └── predml/
│       ├── models/          # Model implementations
│       ├── preprocessing/   # Feature engineering
│       ├── training/        # Training utilities
│       └── utils/           # Helper functions
├── tests/                   # Test suite
├── docker/                  # Docker configuration
├── docs/                    # Documentation
└── examples/                # Usage examples
```

## Development

### Running Tests

```bash
# Run tests with pytest
pytest

# Run tests with coverage
pytest --cov=stata

# Run tests in Docker
docker-compose run stata-dev
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Building Documentation

```bash
# Build documentation
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## Configuration

Example configuration file (`config.yaml`):

```yaml
model_type: "random_forest"
model_params:
  n_estimators: 100
  max_depth: 10
  random_state: 42

feature_engineering:
  scaling_method: "standard"
  handle_missing: true
  numeric_features:
    - feature1
    - feature2
  categorical_features:
    - category1

training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.