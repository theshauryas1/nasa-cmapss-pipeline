"""
Neural Network Models for RUL Prediction

This module provides neural network models for Remaining Useful Life prediction,
including feedforward networks and LSTM for time-series modeling.

Models:
- SimpleNeuralNetwork: Basic feedforward network
- LSTMModel: LSTM for sequential/temporal patterns

These models can capture non-linear relationships that linear models miss,
while LSTM specifically leverages the temporal structure of sensor data.

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network models will not work.")
    print("Install with: pip install torch")


class SimpleNeuralNetwork:
    """
    Simple feedforward neural network for RUL prediction.
    
    Architecture:
    - Input layer
    - Hidden layers with ReLU activation
    - Dropout for regularization
    - Output layer (single value for RUL)
    
    Example:
        >>> model = SimpleNeuralNetwork(input_dim=24, hidden_dims=[64, 32])
        >>> history = model.fit(X_train, y_train, epochs=100)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer sizes.
            dropout_rate: Dropout probability.
            learning_rate: Adam optimizer learning rate.
            device: 'cpu', 'cuda', or 'auto'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network models")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build network
        self.model = self._build_model()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
        self.history_ = {'train_loss': [], 'val_loss': []}
        self.is_fitted_ = False
    
    def _build_model(self) -> nn.Module:
        """Build the PyTorch model."""
        layers = []
        
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X: Training features.
            y: Training targets (RUL).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction for validation.
            verbose: Print training progress.
        
        Returns:
            Training history dictionary.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
            
            train_loss = np.mean(train_losses)
            self.history_['train_loss'].append(train_loss)
            self.history_['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        self.is_fitted_ = True
        return self.history_
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict RUL values.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Predicted RUL values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self.scaler.transform(X)
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy().flatten()
        
        return np.clip(predictions, 0, None)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history_


class LSTMModel:
    """
    LSTM model for time-series RUL prediction.
    
    LSTM is particularly suited for this problem because:
    - Captures temporal dependencies in sensor data
    - Learns degradation patterns over time
    - Handles variable-length sequences
    
    Architecture:
    - LSTM layers with configurable hidden size
    - Fully connected output layer
    - Dropout for regularization
    
    Example:
        >>> model = LSTMModel(input_dim=24, hidden_dim=64, num_layers=2)
        >>> X_seq, y_seq = model.prepare_sequences(df, sequence_length=30)
        >>> model.fit(X_seq, y_seq, epochs=50)
        >>> predictions = model.predict(X_test_seq)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features per time step.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers.
            learning_rate: Optimizer learning rate.
            device: 'cpu', 'cuda', or 'auto'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.history_ = {'train_loss': [], 'val_loss': []}
        self.is_fitted_ = False
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM model."""
        
        class LSTMNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # x shape: (batch, sequence_length, features)
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use last time step
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
                out = self.fc(out)
                return out
        
        return LSTMNetwork(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout
        )
    
    @staticmethod
    def prepare_sequences(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'RUL',
        sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences from DataFrame for LSTM training.
        
        Creates fixed-length sequences from each engine's time series,
        padding shorter sequences with zeros.
        
        Args:
            df: DataFrame with engine_id, cycle, features, and target.
            feature_cols: List of feature column names.
            target_col: Target column name.
            sequence_length: Length of each sequence.
        
        Returns:
            Tuple of (X, y) arrays.
            X shape: (n_samples, sequence_length, n_features)
            y shape: (n_samples,)
        """
        X_list = []
        y_list = []
        
        for engine_id in df['engine_id'].unique():
            engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
            
            features = engine_df[feature_cols].values
            targets = engine_df[target_col].values
            
            n_cycles = len(engine_df)
            
            for i in range(n_cycles):
                # Get sequence ending at current cycle
                start_idx = max(0, i - sequence_length + 1)
                seq = features[start_idx:i + 1]
                
                # Pad if necessary
                if len(seq) < sequence_length:
                    padding = np.zeros((sequence_length - len(seq), len(feature_cols)))
                    seq = np.vstack([padding, seq])
                
                X_list.append(seq)
                y_list.append(targets[i])
        
        return np.array(X_list), np.array(y_list)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
            y: Target RUL values.
            epochs: Number of training epochs.
            batch_size: Batch size.
            validation_split: Fraction for validation.
            verbose: Print progress.
        
        Returns:
            Training history.
        """
        # Split validation
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
            
            train_loss = np.mean(train_losses)
            self.history_['train_loss'].append(train_loss)
            self.history_['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        self.is_fitted_ = True
        return self.history_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL values.
        
        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
        
        Returns:
            Predicted RUL values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy().flatten()
        
        return np.clip(predictions, 0, None)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history_


def plot_training_history(
    history: Dict,
    title: str = 'Training History',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark best validation epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7)
    ax.annotate(f'Best: {best_val:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + len(epochs) * 0.1, best_val * 1.2),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
