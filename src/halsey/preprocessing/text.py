"""
Text Preprocessor Module
======================

This module provides preprocessing functionality for text features,
including tokenization, cleaning, vectorization, and embedding generation.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging

from auto_pred_ml.preprocessing.base import BasePreprocessor

logger = logging.getLogger(__name__)

class TextPreprocessor(BasePreprocessor):
    """Preprocessor for text features.
    
    Supports:
    - Text cleaning and normalization
    - Multiple vectorization methods
    - Dimension reduction
    - Basic NLP preprocessing
    """
    
    def __init__(self,
                 features: List[str],
                 vectorizer: str = 'tfidf',
                 max_features: Optional[int] = 1000,
                 ngram_range: tuple = (1, 1),
                 dim_reduction: Optional[int] = None,
                 clean_text: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 min_df: float = 0.01,
                 max_df: float = 0.95,
                 handle_missing: bool = True,
                 n_jobs: int = -1):
        """Initialize the text preprocessor.
        
        Args:
            features: List of text feature names
            vectorizer: Vectorization method ('tfidf', 'count', 'hashing')
            max_features: Maximum number of features to create
            ngram_range: Range of ngrams to include
            dim_reduction: Number of dimensions for SVD reduction
            clean_text: Whether to clean and normalize text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            handle_missing: Whether to handle missing values
            n_jobs: Number of parallel jobs
        """
        super().__init__(features, handle_missing, 'ignore', n_jobs)
        
        self.vectorizer = vectorizer
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.dim_reduction = dim_reduction
        self.clean_text = clean_text
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize components
        self.vectorizers: Dict[str, Any] = {}
        self.svd_models: Dict[str, Any] = {}
        self.vocabulary_: Dict[str, Dict[str, int]] = {}
        
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Error downloading NLTK resources: {str(e)}")
            self.stop_words = set()
            self.lemmatizer = None
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        super()._validate_parameters()
        
        valid_vectorizers = {'tfidf', 'count', 'hashing'}
        if self.vectorizer not in valid_vectorizers:
            raise ValueError(f"vectorizer must be one of: {valid_vectorizers}")
            
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be positive")
            
        if not isinstance(self.ngram_range, tuple) or len(self.ngram_range) != 2:
            raise ValueError("ngram_range must be a tuple of length 2")
            
        if self.dim_reduction is not None and self.dim_reduction < 1:
            raise ValueError("dim_reduction must be positive")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def _get_vectorizer(self, feature: str):
        """Get vectorizer instance based on method.
        
        Args:
            feature: Feature name
            
        Returns:
            Vectorizer instance
        """
        if self.vectorizer == 'tfidf':
            return TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=None,  # Already handled in cleaning
                strip_accents='unicode',
                n_jobs=self.n_jobs
            )
        elif self.vectorizer == 'count':
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=None,
                strip_accents='unicode'
            )
        else:  # hashing
            from sklearn.feature_extraction.text import HashingVectorizer
            return HashingVectorizer(
                n_features=self.max_features,
                ngram_range=self.ngram_range,
                strip_accents='unicode',
                n_jobs=self.n_jobs
            )
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TextPreprocessor':
        """Fit text preprocessor to the data.
        
        Args:
            X: Input features
            y: Target variable (optional, not used)
            
        Returns:
            self: Fitted preprocessor
        """
        self._validate_data(X)
        
        for feature in self.features:
            # Handle missing values
            if self.handle_missing:
                X[feature] = X[feature].fillna('')
            
            # Clean text
            if self.clean_text:
                texts = X[feature].apply(self._clean_text)
            else:
                texts = X[feature]
            
            # Fit vectorizer
            vectorizer = self._get_vectorizer(feature)
            vectorized = vectorizer.fit_transform(texts)
            self.vectorizers[feature] = vectorizer
            
            if hasattr(vectorizer, 'vocabulary_'):
                self.vocabulary_[feature] = vectorizer.vocabulary_
            
            # Fit dimension reduction if needed
            if self.dim_reduction:
                svd = TruncatedSVD(
                    n_components=min(self.dim_reduction, vectorized.shape[1] - 1),
                    random_state=42
                )
                svd.fit(vectorized)
                self.svd_models[feature] = svd
            
            # Store feature statistics
            self.feature_statistics_[feature] = {
                'vocab_size': len(self.vocabulary_.get(feature, {})),
                'sparsity': 1.0 - (vectorized.nnz / 
                                 (vectorized.shape[0] * vectorized.shape[1])),
                'missing_rate': X[feature].isnull().mean(),
                'mean_length': X[feature].str.len().mean()
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text features using fitted preprocessors.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features DataFrame
        """
        self._check_is_fitted()
        self._validate_data(X)
        
        result = pd.DataFrame()
        
        for feature in self.features:
            # Handle missing values
            if self.handle_missing:
                X[feature] = X[feature].fillna('')
            
            # Clean text
            if self.clean_text:
                texts = X[feature].apply(self._clean_text)
            else:
                texts = X[feature]
            
            # Apply vectorization
            vectorized = self.vectorizers[feature].transform(texts)
            
            # Apply dimension reduction if needed
            if feature in self.svd_models:
                transformed = self.svd_models[feature].transform(vectorized)
                feature_names = [f"{feature}_dim_{i}" for i in range(transformed.shape[1])]
            else:
                transformed = vectorized.toarray()
                if feature in self.vocabulary_:
                    vocab = self.vocabulary_[feature]
                    feature_names = [f"{feature}_{word}" for word in 
                                   sorted(vocab, key=vocab.get)]
                else:
                    feature_names = [f"{feature}_{i}" for i in range(transformed.shape[1])]
            
            # Add to result DataFrame
            for i, name in enumerate(feature_names):
                result[name] = transformed[:, i]
        
        return result