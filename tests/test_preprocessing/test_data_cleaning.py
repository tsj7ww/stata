"""Tests for data cleaning functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from predml.preprocessing.data_cleaning import DataCleaner


@pytest.fixture
def sample_dirty_data():
    """Create sample data with various issues to clean."""
    return pd.DataFrame({
        'id': [1, 1, 2, 3, 4, 5],  # Contains duplicate
        'value': [10.0, 10.0, np.nan, 30.0, 1000.0, 50.0],  # Contains missing and outlier
        'category': ['A', 'A', 'B', None, 'C', 'C'],  # Contains missing
        'date': ['2024-01-01', '2024-01-01', 'invalid', '2024-01-03', '2024-01-04', '2024-01-05']
    })

def test_data_cleaner_initialization():
    """Test DataCleaner initialization."""
    cleaner = DataCleaner(
        remove_duplicates=True,
        handle_missing=True,
        drop_threshold=0.9,
        date_columns=['date']
    )
    assert cleaner.remove_duplicates is True
    assert cleaner.handle_missing is True
    assert cleaner.drop_threshold == 0.9
    assert cleaner.date_columns == ['date']

def test_remove_duplicates(sample_dirty_data):
    """Test duplicate removal."""
    cleaner = DataCleaner(remove_duplicates=True)
    cleaned_data = cleaner.clean(sample_dirty_data)
    
    assert len(cleaned_data) < len(sample_dirty_data)
    assert cleaner.duplicates_removed == 1
    assert cleaned_data['id'].nunique() == cleaned_data['id'].count()

def test_handle_missing_values(sample_dirty_data):
    """Test missing value handling."""
    cleaner = DataCleaner(handle_missing=True)
    cleaned_data = cleaner.clean(
        sample_dirty_data,
        numeric_columns=['value'],
        categorical_columns=['category']
    )
    
    assert not cleaned_data['value'].isnull().any()
    assert not cleaned_data['category'].isnull().any()

def test_date_parsing(sample_dirty_data):
    """Test date parsing functionality."""
    cleaner = DataCleaner(date_columns=['date'])
    cleaned_data = cleaner.clean(sample_dirty_data)
    
    valid_dates = cleaned_data['date'].dropna()
    assert all(isinstance(d, pd.Timestamp) for d in valid_dates)

def test_outlier_removal():
    """Test outlier removal functionality."""
    data = pd.DataFrame({
        'value': [1, 2, 3, 100, 4, 5]  # 100 is an outlier
    })
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.remove_outliers(
        data,
        columns=['value'],
        method='zscore',
        threshold=2.0
    )
    
    assert len(cleaned_data) < len(data)
    assert 100 not in cleaned_data['value'].values

def test_cleaning_report(sample_dirty_data):
    """Test cleaning report generation."""
    cleaner = DataCleaner(remove_duplicates=True)
    cleaned_data = cleaner.clean(sample_dirty_data)
    report = cleaner.get_cleaning_report()
    
    assert isinstance(report, dict)
    assert 'duplicates_removed' in report
    assert 'columns_dropped' in report
    assert 'timestamp' in report
    assert report['duplicates_removed'] == 1

def test_high_missing_ratio_column_drop():
    """Test dropping columns with high missing ratio."""
    data = pd.DataFrame({
        'good': [1, 2, 3, 4, 5],
        'bad': [np.nan, np.nan, np.nan, np.nan, 5]  # 80% missing
    })
    
    cleaner = DataCleaner(drop_threshold=0.7)  # Drop if >70% missing
    cleaned_data = cleaner.clean(data)
    
    assert 'bad' not in cleaned_data.columns
    assert 'bad' in cleaner.columns_dropped

def test_iqr_based_outlier_removal():
    """Test IQR-based outlier removal."""
    data = pd.DataFrame({
        'value': [1, 2, 2, 3, 3, 4, 100]  # 100 is an outlier
    })
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.remove_outliers(
        data,
        columns=['value'],
        method='iqr',
        threshold=1.5
    )
    
    assert len(cleaned_data) < len(data)
    assert 100 not in cleaned_data['value'].values

def test_multiple_column_cleaning():
    """Test cleaning multiple columns simultaneously."""
    data = pd.DataFrame({
        'num1': [1, np.nan, 3, 100],
        'num2': [10, 20, np.nan, 40],
        'cat1': ['A', None, 'B', 'B'],
        'cat2': ['X', 'Y', None, 'Z']
    })
    
    cleaner = DataCleaner(handle_missing=True)
    cleaned_data = cleaner.clean(
        data,
        numeric_columns=['num1', 'num2'],
        categorical_columns=['cat1', 'cat2']
    )
    
    assert not cleaned_data.isnull().any().any()