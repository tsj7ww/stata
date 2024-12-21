from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from ..config.model_config import ModelConfig

class BaseNetwork(nn.Module):
    """Base neural network architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output dimensions
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

class MLPModel(BaseModel):
    """
    Multilayer Perceptron implementation with automatic architecture selection
    and hyperparameter optimization support.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, 
                 task: str = 'classification'):
        """
        Initialize MLP model.
        
        Args:
            model_config: Model configuration
            task: Either 'classification' or 'regression'
        """
        super().__init__(model_config)
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
    
    def _build_model(self) -> Any:
        """Build neural network with specified architecture."""
        params = self.model_config.get_model_params('mlp')
        
        # Default architecture if not specified
        hidden_dims = params.get('hidden_dims', [64, 32])
        dropout = params.get('dropout', 0.1)
        
        if not hasattr(self, 'input_dim'):
            raise ValueError("Input dimension not set. Call fit() first.")
        
        output_dim = 1 if self.task == 'regression' else 2  # Binary classification default
        
        return BaseNetwork(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        ).to(self.device)
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        params = self.model_config.get_model_params('mlp')
        optimizer_name = params.get('optimizer', 'adam').lower()
        lr = params.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, 
                           momentum=params.get('momentum', 0.9))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_criterion(self) -> nn.Module:
        """Get loss function based on task."""
        if self.task == 'regression':
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series]) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data for training.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Tuple of training DataLoader and optional validation DataLoader
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values) if self.task == 'regression' else \
                  torch.LongTensor(y.values)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Get batch size from config
        params = self.model_config.get_model_params('mlp')
        batch_size = params.get('batch_size', 32)
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None,
            **kwargs) -> 'MLPModel':
        """
        Fit the neural network.
        
        Args:
            X: Training features
            y: Target values
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional arguments including early_stopping_patience
        """
        self.input_dim = X.shape[1]
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
        
        # Prepare data
        train_loader = self._prepare_data(X, y)
        if validation_data:
            val_loader = self._prepare_data(*validation_data)
        else:
            val_loader = None
        
        # Get training parameters
        params = self.model_config.get_model_params('mlp')
        epochs = params.get('epochs', 100)
        early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Initialize optimizer and criterion
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.task == 'classification':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    break
        
        self.is_fitted = True
        return self
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Perform validation step."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                
                if self.task == 'classification':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained model."""
        self._check_is_fitted()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task == 'classification':
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = outputs.squeeze()
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        self._check_is_fitted()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            str: Model summary
        """
        return str(self.model)