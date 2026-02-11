"""
Temporal Convolutional Network (TCN) for RUL Prediction

This module implements a TCN architecture optimized for the NASA C-MAPSS
turbofan engine degradation dataset. TCN is the recommended primary model
for GTX 1650 GPUs due to its efficient parallelizable convolutions.

Architecture:
- Stack of TemporalBlock layers with dilated causal 1D convolutions
- Dilation pattern: [1, 2, 4, 8, ...] (exponentially increasing receptive field)
- Each block: Conv1d → WeightNorm → ReLU → Dropout → Conv1d → WeightNorm → ReLU → Dropout + Residual
- Final linear projection to scalar RUL output

GTX 1650 Safe Defaults:
- Window: 30 (max 40)
- Channels: 64 (max 96)
- TCN blocks: 4 (max 6)
- Batch size: 16 (32 with AMP)
- AMP (FP16) enabled by default

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
    from torch.nn.utils import weight_norm
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. TCN model will not work.")
    print("Install with: pip install torch")


class Chomp1d(nn.Module):
    """Remove extra padding to enforce causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN block with two layers of dilated causal convolutions.

    Structure:
        Conv1d → WeightNorm → ReLU → Dropout →
        Conv1d → WeightNorm → ReLU → Dropout + Residual Connection
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 1x1 conv for residual if channel dimensions differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize conv weights with normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Full TCN backbone: stack of TemporalBlocks with exponential dilation.

    Args:
        num_inputs: Number of input features (channels).
        num_channels: List of channel sizes for each block.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        return self.network(x)


class TCNModel:
    """
    TCN model for time-series RUL prediction.

    This is the recommended primary model for GTX 1650 because:
    - Parallelizable convolutions are faster than recurrent models
    - Stable gradients (no vanishing/exploding gradient like RNNs)
    - Flexible receptive field via dilation
    - Lower VRAM usage than Transformers

    Example:
        >>> model = TCNModel(input_dim=14, num_channels=[64]*4)
        >>> X_seq, y_seq = TCNModel.prepare_sequences(df, feature_cols, sequence_length=30)
        >>> model.fit(X_seq, y_seq, epochs=50, batch_size=16, use_amp=True)
        >>> predictions = model.predict(X_test_seq)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize the TCN model.

        Args:
            input_dim: Number of input features per time step.
            num_channels: List of channel sizes for each TCN block.
                          Default: [64, 64, 64, 64] (4 blocks × 64 channels).
            kernel_size: Convolution kernel size (default 3).
            dropout: Dropout probability (default 0.2).
            learning_rate: Adam optimizer learning rate.
            device: 'cpu', 'cuda', or 'auto'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TCN model")

        self.input_dim = input_dim
        self.num_channels = num_channels or [64, 64, 64, 64]
        self.kernel_size = kernel_size
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

        self.history_ = {'train_loss': [], 'val_loss': [], 'epoch_time': []}
        self.is_fitted_ = False
        self.best_val_loss_ = float('inf')
        self.best_state_dict_ = None

    def _build_model(self) -> nn.Module:
        """Build the TCN model with linear output head."""

        class TCNRegressor(nn.Module):
            def __init__(self, input_dim, num_channels, kernel_size, dropout):
                super().__init__()
                self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
                self.fc = nn.Linear(num_channels[-1], 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x shape: (batch, sequence_length, features)
                # TCN expects: (batch, features/channels, sequence_length)
                x = x.transpose(1, 2)
                tcn_out = self.tcn(x)
                # Use last time step output
                last_out = tcn_out[:, :, -1]
                out = self.dropout(last_out)
                out = self.fc(out)
                return out

        return TCNRegressor(
            self.input_dim, self.num_channels,
            self.kernel_size, self.dropout
        )

    @staticmethod
    def prepare_sequences(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'RUL',
        sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences from DataFrame for TCN training.

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
        Train the TCN model.

        Args:
            X: Sequence data of shape (n_samples, seq_length, n_features).
            y: Target RUL values.
            epochs: Number of training epochs (default 50).
            batch_size: Batch size (default 16, safe for GTX 1650).
            validation_split: Fraction for validation.
            verbose: Print training progress.
            use_amp: Enable mixed precision (FP16) training. Default True.
            patience: Early stopping patience (epochs without improvement).
            gradient_clip: Max gradient norm for clipping (default 1.0).

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
            print(f"Training TCN on {self.device} | AMP: {use_amp}")
            print(f"  Channels: {self.num_channels} | Kernel: {self.kernel_size}")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Batch: {batch_size}")
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

                # Gradient clipping
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

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

    def get_receptive_field(self) -> int:
        """
        Compute the effective receptive field of the TCN.

        Returns:
            Number of time steps the model can look back.
        """
        num_levels = len(self.num_channels)
        return 1 + 2 * (self.kernel_size - 1) * (2 ** num_levels - 1)

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
            },
            'history': self.history_,
            'best_val_loss': self.best_val_loss_,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'TCNModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
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
