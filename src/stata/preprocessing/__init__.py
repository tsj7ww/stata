"""
Preprocessing Package
===================

This package provides preprocessing components for different types of features:
- Numerical features
- Categorical features
- Text features
- Feature selection
"""

from auto_ml.preprocessing.base import BasePreprocessor
from auto_ml.preprocessing.numerical import NumericalPreprocessor
from auto_ml.preprocessing.categorical import CategoricalPreprocessor
from auto_ml.preprocessing.text import TextPreprocessor
from auto_ml.preprocessing.feature_selection import FeatureSelector

__all__ = [
    'BasePreprocessor',
    'NumericalPreprocessor',
    'CategoricalPreprocessor',
    'TextPreprocessor',
    'FeatureSelector'
]