"""
Compact Transformer Encoder for RUL Prediction

This module implements a small-footprint Transformer encoder architecture
optimized for the NASA C-MAPSS turbofan engine degradation dataset.

⚠️ WARNING: Full-size Transformers WILL OOM on GTX 1650.
This is a compact configuration. Do NOT exceed the hard limits below.

Architecture:
- Feature projection (input_dim → d_model)
- Sinusoidal positional encoding
- TransformerEncoder blocks (multi-head self-attention → FFN)
- Mean-pooling across time → Linear output head

GTX 1650 Hard Limits:
- d_model: 128 (DO NOT use ≥ 256)
- num_layers: 2-3 (DO NOT use ≥ 6)
- nhead: 4
- dim_feedforward: 256 (max 512)
- batch_size: 8 (max 8)
- sequence_length: 30 (DO NOT use ≥ 100)
- AMP: MANDATORY

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import time
import warnings

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
    print("PyTorch not available. Transformer model will not work.")
    print("Install with: pip install torch")


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Adds position-dependent signals to the input embeddings so the model
    can distinguish different time steps.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Position-encoded tensor, same shape.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CompactTransformerNetwork(nn.Module):
    """
    Compact Transformer Encoder for RUL regression.

    Architecture:
        Input → Linear Projection → Positional Encoding →
        TransformerEncoder → Mean Pooling → Dropout → FC → RUL
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        super().__init__()

        self.d_model = d_model

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim)

        Returns:
            output: RUL prediction, shape (batch, 1)
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Mean pooling over time dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Output
        x = self.dropout(x)
        x = self.fc_out(x)  # (batch, 1)

        return x


class CompactTransformerModel:
    """
    Compact Transformer model for time-series RUL prediction.

    ⚠️ This model is possible but painful on GTX 1650. Use as a last resort.
    TCN is recommended as the primary model.

    The Transformer's self-attention mechanism captures long-range dependencies
    without the sequential bottleneck of RNNs, but at the cost of quadratic
    memory usage (O(seq_len²)).

    Example:
        >>> model = CompactTransformerModel(input_dim=14, d_model=128, nhead=4, num_layers=2)
        >>> X_seq, y_seq = CompactTransformerModel.prepare_sequences(df, feature_cols, sequence_length=30)
        >>> model.fit(X_seq, y_seq, epochs=50, batch_size=8, use_amp=True)
        >>> predictions = model.predict(X_test_seq)
    """

    # Hard limits for GTX 1650 safety
    _GTX1650_LIMITS = {
        'd_model_max': 256,
        'num_layers_max': 5,
        'seq_len_max': 100,
        'batch_size_max': 16,
    }

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        device: str = 'auto'
    ):
        """
        Initialize the Compact Transformer model.

        Args:
            input_dim: Number of input features per time step.
            d_model: Transformer embedding dimension (default 128, HARD MAX 256 on 1650).
            nhead: Number of attention heads (must divide d_model).
            num_layers: Number of TransformerEncoder layers (default 2, max 3 on 1650).
            dim_feedforward: FFN intermediate dimension (default 256).
            dropout: Dropout probability.
            learning_rate: AdamW optimizer learning rate (lower than TCN/BiLSTM).
            device: 'cpu', 'cuda', or 'auto'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer model")

        # Safety warnings
        if d_model >= self._GTX1650_LIMITS['d_model_max']:
            warnings.warn(
                f"d_model={d_model} may OOM on GTX 1650! Recommended: 128",
                RuntimeWarning
            )
        if num_layers >= self._GTX1650_LIMITS['num_layers_max']:
            warnings.warn(
                f"num_layers={num_layers} may OOM on GTX 1650! Recommended: 2-3",
                RuntimeWarning
            )

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = CompactTransformerNetwork(
            input_dim, d_model, nhead, num_layers,
            dim_feedforward, dropout
        )
        self.model.to(self.device)

        # AdamW with weight decay (better for Transformers than vanilla Adam)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Cosine annealing LR scheduler
        self.scheduler = None  # Will be set in fit()

        self.criterion = nn.MSELoss()

        self.history_ = {'train_loss': [], 'val_loss': [], 'epoch_time': [], 'lr': []}
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
        Prepare sequences from DataFrame for Transformer training.

        Creates fixed-length sequences from each engine's time series,
        padding shorter sequences with zeros.

        Args:
            df: DataFrame with engine_id, cycle, features, and target.
            feature_cols: List of feature column names.
            target_col: Target column name.
            sequence_length: Length of each sequence (default 30, HARD MAX 100 on 1650).

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
        batch_size: int = 8,
        validation_split: float = 0.1,
        verbose: bool = True,
        use_amp: bool = True,
        patience: int = 15,
        gradient_clip: float = 0.5,
        warmup_epochs: int = 5
    ) -> Dict:
        """
        Train the Compact Transformer model.

        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
            y: Target RUL values.
            epochs: Number of training epochs.
            batch_size: Batch size (default 8, max 8 on GTX 1650).
            validation_split: Fraction for validation.
            verbose: Print training progress.
            use_amp: Enable mixed precision (MANDATORY on GTX 1650).
            patience: Early stopping patience (higher than TCN/BiLSTM because
                      Transformers converge slower).
            gradient_clip: Max gradient norm (lower than LSTM, Transformers
                          are more sensitive).
            warmup_epochs: Number of warmup epochs for LR scheduler.

        Returns:
            Training history dictionary.
        """
        # Safety check
        if X.shape[1] >= self._GTX1650_LIMITS['seq_len_max']:
            warnings.warn(
                f"Sequence length {X.shape[1]} may OOM on GTX 1650!",
                RuntimeWarning
            )

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

        # AMP setup — mandatory for GTX 1650
        use_amp = use_amp and self.device.type == 'cuda'
        if self.device.type == 'cuda' and not use_amp:
            warnings.warn(
                "AMP disabled on GPU! This will likely OOM on GTX 1650.",
                RuntimeWarning
            )
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Cosine annealing LR scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )

        # Early stopping
        patience_counter = 0

        if verbose:
            print(f"Training Compact Transformer on {self.device} | AMP: {use_amp}")
            print(f"  d_model: {self.d_model} | heads: {self.nhead} | "
                  f"layers: {self.num_layers} | ffn: {self.dim_feedforward}")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | "
                  f"Batch: {batch_size} | Seq len: {X.shape[1]}")
            if self.device.type == 'cuda':
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)")
            print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            train_losses = []

            # Linear warmup
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * warmup_factor

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                scaler.scale(loss).backward()

                # Gradient clipping (lower value than LSTM — Transformers
                # are more sensitive to gradient spikes)
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )

                scaler.step(self.optimizer)
                scaler.update()

                train_losses.append(loss.item())

            # Step LR scheduler (after warmup)
            if epoch >= warmup_epochs:
                self.scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()

            train_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history_['train_loss'].append(train_loss)
            self.history_['val_loss'].append(val_loss)
            self.history_['epoch_time'].append(epoch_time)
            self.history_['lr'].append(current_lr)

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
                    f"LR: {current_lr:.2e} - Time: {epoch_time:.1f}s{gpu_mem}"
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

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
            },
            'history': self.history_,
            'best_val_loss': self.best_val_loss_,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'CompactTransformerModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
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
