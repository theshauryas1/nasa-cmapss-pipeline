"""
Bidirectional LSTM with Attention Mechanism for RUL Prediction

This module implements a BiLSTM + Attention architecture optimized for the
NASA C-MAPSS turbofan engine degradation dataset. The attention mechanism
allows the model to focus on the most relevant time steps for RUL prediction.

Architecture:
- Bidirectional LSTM layers (sees both past and future context in the window)
- Bahdanau (additive) attention over all time steps → weighted context vector
- Fully connected output head with dropout

GTX 1650 Safe Defaults:
- Layers: 2
- Hidden size: 128
- Batch size: 16
- Window: 30
- Gradient clipping: 1.0
- ~1.5–2× slower than TCN per epoch

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. BiLSTM+Attention model will not work.")
    print("Install with: pip install torch")


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.

    Computes attention weights over all time steps of the LSTM output,
    producing a single weighted context vector that summarizes the
    most relevant temporal information for RUL prediction.

    Attention weights are interpretable — they show which time steps
    the model considers most important for its prediction.
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Size of the LSTM hidden state (per direction).
                        For BiLSTM, the input will be 2 * hidden_dim.
        """
        super().__init__()
        self.attention_dim = hidden_dim
        input_dim = hidden_dim * 2  # BiLSTM outputs 2x hidden dim

        self.W = nn.Linear(input_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor):
        """
        Compute attention-weighted context vector.

        Args:
            lstm_output: BiLSTM output, shape (batch, seq_len, 2*hidden_dim)

        Returns:
            context: Weighted context vector, shape (batch, 2*hidden_dim)
            attention_weights: Attention distribution, shape (batch, seq_len)
        """
        # Score each time step
        energy = torch.tanh(self.W(lstm_output))   # (batch, seq_len, attention_dim)
        scores = self.v(energy).squeeze(-1)         # (batch, seq_len)

        # Softmax to get attention distribution
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                       # (batch, seq_len, 2*hidden_dim)
        ).squeeze(1)                          # (batch, 2*hidden_dim)

        return context, attention_weights


class BiLSTMAttentionNetwork(nn.Module):
    """
    Full BiLSTM + Attention network for RUL regression.

    Architecture:
        Input → BiLSTM layers → Bahdanau Attention → Dropout → FC → RUL
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = BahdanauAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 2x for bidirectional

    def forward(self, x, return_attention: bool = False):
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim)
            return_attention: If True, also return attention weights.

        Returns:
            output: RUL prediction, shape (batch, 1)
            attention_weights: (optional) shape (batch, seq_len)
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2*hidden_dim)

        # Attention
        context, attention_weights = self.attention(lstm_out)

        # Output
        out = self.dropout(context)
        out = self.fc(out)

        if return_attention:
            return out, attention_weights
        return out


class BiLSTMAttentionModel:
    """
    BiLSTM + Attention model for time-series RUL prediction.

    Advantages over standard LSTM:
    - Bidirectional: sees both past and future context within the window
    - Attention: learns to focus on the most degradation-relevant time steps
    - Interpretable: attention weights show which cycles matter most

    Trade-off: ~1.5–2× slower than TCN. FD004 may take multiple nights.

    Example:
        >>> model = BiLSTMAttentionModel(input_dim=14, hidden_dim=128, num_layers=2)
        >>> X_seq, y_seq = BiLSTMAttentionModel.prepare_sequences(df, feature_cols, sequence_length=30)
        >>> model.fit(X_seq, y_seq, epochs=50, batch_size=16)
        >>> predictions = model.predict(X_test_seq)
        >>> predictions, attn_weights = model.predict(X_test_seq, return_attention=True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize the BiLSTM + Attention model.

        Args:
            input_dim: Number of input features per time step.
            hidden_dim: LSTM hidden state dimension (default 128).
            num_layers: Number of stacked BiLSTM layers (default 2).
            dropout: Dropout probability (default 0.2).
            learning_rate: Adam optimizer learning rate.
            device: 'cpu', 'cuda', or 'auto'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BiLSTM+Attention model")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = BiLSTMAttentionNetwork(
            input_dim, hidden_dim, num_layers, dropout
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.history_ = {'train_loss': [], 'val_loss': [], 'epoch_time': []}
        self.is_fitted_ = False
        self.best_val_loss_ = float('inf')
        self.best_state_dict_ = None

    @staticmethod
    def prepare_sequences(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'RUL',
        sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences from DataFrame for BiLSTM training.

        Creates fixed-length sequences from each engine's time series,
        padding shorter sequences with zeros.

        Args:
            df: DataFrame with engine_id, cycle, features, and target.
            feature_cols: List of feature column names.
            target_col: Target column name.
            sequence_length: Length of each sequence (default 30).

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
                start_idx = max(0, i - sequence_length + 1)
                seq = features[start_idx:i + 1]

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
        batch_size: int = 16,
        validation_split: float = 0.1,
        verbose: bool = True,
        use_amp: bool = True,
        patience: int = 10,
        gradient_clip: float = 1.0
    ) -> Dict:
        """
        Train the BiLSTM + Attention model.

        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
            y: Target RUL values.
            epochs: Number of training epochs.
            batch_size: Batch size (default 16, safe for GTX 1650).
            validation_split: Fraction for validation.
            verbose: Print training progress.
            use_amp: Enable mixed precision (FP16) training.
            patience: Early stopping patience.
            gradient_clip: Max gradient norm for clipping (crucial for LSTMs).

        Returns:
            Training history dictionary.
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

        # AMP setup
        use_amp = use_amp and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Early stopping
        patience_counter = 0

        if verbose:
            print(f"Training BiLSTM+Attention on {self.device} | AMP: {use_amp}")
            print(f"  Layers: {self.num_layers} | Hidden: {self.hidden_dim} | Bidirectional")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Batch: {batch_size}")
            print(f"  Gradient clipping: {gradient_clip}")
            if self.device.type == 'cuda':
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)")
            print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                scaler.scale(loss).backward()

                # Gradient clipping — essential for LSTM stability
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )

                scaler.step(self.optimizer)
                scaler.update()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()

            train_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start

            self.history_['train_loss'].append(train_loss)
            self.history_['val_loss'].append(val_loss)
            self.history_['epoch_time'].append(epoch_time)

            # Checkpointing
            if val_loss < self.best_val_loss_:
                self.best_val_loss_ = val_loss
                self.best_state_dict_ = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                gpu_mem = ""
                if self.device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated() / 1e6
                    gpu_mem = f" | GPU Mem: {mem_used:.0f}MB"
                print(
                    f"Epoch {epoch+1:3d}/{epochs} - "
                    f"Train: {train_loss:.4f} - Val: {val_loss:.4f} - "
                    f"Time: {epoch_time:.1f}s{gpu_mem}"
                    f"{' *' if patience_counter == 0 else ''}"
                )

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break

        # Restore best weights
        if self.best_state_dict_ is not None:
            self.model.load_state_dict(self.best_state_dict_)
            self.model.to(self.device)
            if verbose:
                print(f"Restored best model (val_loss={self.best_val_loss_:.4f})")

        self.is_fitted_ = True
        return self.history_

    def predict(
        self,
        X: np.ndarray,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict RUL values.

        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
            return_attention: If True, also return attention weights for
                             interpretability analysis.

        Returns:
            predictions: Predicted RUL values.
            attention_weights: (optional) Attention weights per time step.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X_t = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if return_attention:
                outputs, attn_weights = self.model(X_t, return_attention=True)
                predictions = outputs.cpu().numpy().flatten()
                attention = attn_weights.cpu().numpy()
                return np.clip(predictions, 0, None), attention
            else:
                outputs = self.model(X_t)
                predictions = outputs.cpu().numpy().flatten()
                return np.clip(predictions, 0, None)

    def get_history(self) -> Dict:
        """Get training history."""
        return self.history_

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
            },
            'history': self.history_,
            'best_val_loss': self.best_val_loss_,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'BiLSTMAttentionModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            device=device,
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.history_ = checkpoint['history']
        model.best_val_loss_ = checkpoint['best_val_loss']
        model.is_fitted_ = True
        return model
