"""
NASA C-MAPSS Data Preprocessing Module

This module provides preprocessing and feature engineering functionality for
turbofan engine degradation data. It includes normalization, rolling statistics
computation, and domain-informed feature extraction.

The preprocessing pipeline is designed to:
1. Handle multivariate time-series sensor data
2. Normalize sensor readings across different scales
3. Generate degradation-related features
4. Preserve temporal structure within engine lifecycles

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .ingestion import get_sensor_columns, get_operational_columns, get_feature_columns


class DataPreprocessor:
    """
    Preprocessing pipeline for C-MAPSS turbofan sensor data.
    
    This class provides methods for:
    - Data normalization (min-max and z-score)
    - Rolling window statistics
    - Degradation indicator extraction
    - Feature engineering
    
    The preprocessor maintains state (e.g., normalization parameters) so that
    the same transformations can be applied consistently to train and test data.
    
    Example:
        >>> preprocessor = DataPreprocessor(normalization='minmax')
        >>> train_processed = preprocessor.fit_transform(train_df)
        >>> test_processed = preprocessor.transform(test_df)
    """
    
    def __init__(
        self,
        normalization: str = 'minmax',
        rolling_windows: List[int] = [5, 10, 20],
        drop_constant_sensors: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            normalization: Normalization method ('minmax', 'zscore', or None).
            rolling_windows: List of window sizes for rolling statistics.
            drop_constant_sensors: Whether to drop sensors with zero variance.
        """
        self.normalization = normalization
        self.rolling_windows = rolling_windows
        self.drop_constant_sensors = drop_constant_sensors
        
        # Parameters learned from training data
        self.norm_params_: Optional[Dict] = None
        self.constant_sensors_: Optional[List[str]] = None
        self.is_fitted_: bool = False
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor to training data.
        
        Learns normalization parameters and identifies constant sensors.
        
        Args:
            df: Training DataFrame.
        
        Returns:
            self (for method chaining).
        """
        sensor_cols = get_sensor_columns()
        op_cols = get_operational_columns()
        feature_cols = sensor_cols + op_cols
        
        # Identify constant sensors (zero variance)
        if self.drop_constant_sensors:
            variances = df[sensor_cols].var()
            self.constant_sensors_ = variances[variances < 1e-10].index.tolist()
        else:
            self.constant_sensors_ = []
        
        # Compute normalization parameters
        cols_to_normalize = [c for c in feature_cols if c not in self.constant_sensors_]
        
        if self.normalization == 'minmax':
            self.norm_params_ = {
                'type': 'minmax',
                'min': df[cols_to_normalize].min().to_dict(),
                'max': df[cols_to_normalize].max().to_dict(),
            }
        elif self.normalization == 'zscore':
            self.norm_params_ = {
                'type': 'zscore',
                'mean': df[cols_to_normalize].mean().to_dict(),
                'std': df[cols_to_normalize].std().to_dict(),
            }
        else:
            self.norm_params_ = {'type': None}
        
        self.is_fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
        """
        Transform data using learned parameters.
        
        Args:
            df: DataFrame to transform.
            add_features: Whether to add rolling window features.
        
        Returns:
            Transformed DataFrame.
        
        Raises:
            ValueError: If fit() has not been called.
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform()")
        
        df = df.copy()
        
        # Drop constant sensors
        if self.constant_sensors_:
            df = df.drop(columns=self.constant_sensors_, errors='ignore')
        
        # Apply normalization
        df = self._apply_normalization(df)
        
        # Add rolling features
        if add_features and self.rolling_windows:
            df = self._add_rolling_features(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, add_features=add_features)
    
    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to sensor and operational columns."""
        if self.norm_params_['type'] is None:
            return df
        
        for col in self.norm_params_.get('min', self.norm_params_.get('mean', {})).keys():
            if col not in df.columns:
                continue
            
            if self.norm_params_['type'] == 'minmax':
                min_val = self.norm_params_['min'][col]
                max_val = self.norm_params_['max'][col]
                range_val = max_val - min_val
                if range_val > 0:
                    df[col] = (df[col] - min_val) / range_val
                else:
                    df[col] = 0.0
            
            elif self.norm_params_['type'] == 'zscore':
                mean_val = self.norm_params_['mean'][col]
                std_val = self.norm_params_['std'][col]
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                else:
                    df[col] = 0.0
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics for each sensor.
        
        Features added for each window size:
        - Rolling mean
        - Rolling standard deviation
        - Rolling min/max range
        
        Rolling statistics are computed within each engine's lifecycle.
        """
        sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
        
        for window in self.rolling_windows:
            for col in sensor_cols:
                # Group by engine to prevent leakage across engines
                grouped = df.groupby('engine_id')[col]
                
                # Rolling mean
                df[f'{col}_roll_mean_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std (with min_periods=2 to avoid NaN)
                df[f'{col}_roll_std_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                ).fillna(0)
                
                # Rolling range (max - min)
                roll_max = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                roll_min = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'{col}_roll_range_{window}'] = roll_max - roll_min
        
        return df
    
    def get_active_sensors(self) -> List[str]:
        """Return list of sensors after dropping constant ones."""
        all_sensors = get_sensor_columns()
        if self.constant_sensors_:
            return [s for s in all_sensors if s not in self.constant_sensors_]
        return all_sensors


def add_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add degradation-related features to the dataset.
    
    Features based on domain knowledge about turbofan degradation:
    - Cycle-normalized position (% of observed lifecycle)
    - Sensor trend indicators (slope of recent observations)
    - Rate of change features
    
    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
    
    Returns:
        DataFrame with additional degradation features.
    """
    df = df.copy()
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    # Normalized cycle position within each engine
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_norm'] = df['cycle'] / max_cycles
    
    # Rate of change (difference from previous cycle)
    for col in sensor_cols:
        df[f'{col}_diff'] = df.groupby('engine_id')[col].diff().fillna(0)
    
    # Exponential weighted mean (emphasizes recent observations)
    for col in sensor_cols:
        df[f'{col}_ewm'] = df.groupby('engine_id')[col].transform(
            lambda x: x.ewm(span=10, min_periods=1).mean()
        )
    
    return df


def identify_important_sensors(
    df: pd.DataFrame,
    rul_column: str = 'RUL',
    method: str = 'correlation',
    top_k: int = 10
) -> List[str]:
    """
    Identify sensors most correlated with RUL (degradation).
    
    This is a domain-informed feature selection approach that helps focus
    modeling efforts on the most predictive sensors.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        rul_column: Name of the RUL column.
        method: 'correlation' (Pearson) or 'spearman'.
        top_k: Number of top sensors to return.
    
    Returns:
        List of sensor column names sorted by importance.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    if rul_column not in df.columns:
        raise ValueError(f"RUL column '{rul_column}' not found in DataFrame")
    
    correlations = {}
    for col in sensor_cols:
        if method == 'correlation':
            corr, _ = stats.pearsonr(df[col], df[rul_column])
        else:
            corr, _ = stats.spearmanr(df[col], df[rul_column])
        correlations[col] = abs(corr)  # Use absolute value
    
    # Sort by correlation magnitude
    sorted_sensors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    return [sensor for sensor, _ in sorted_sensors[:top_k]]


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'RUL'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences for time-series modeling (e.g., LSTM).
    
    Creates fixed-length sequences from each engine's time series.
    For engines with fewer cycles than sequence_length, pads with zeros.
    
    Args:
        df: DataFrame with engine_id, features, and target.
        sequence_length: Length of each sequence.
        feature_cols: List of feature columns to include.
        target_col: Target column name.
    
    Returns:
        Tuple of (X, y, engine_ids) arrays.
        X shape: (n_samples, sequence_length, n_features)
        y shape: (n_samples,)
        engine_ids shape: (n_samples,)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    X_list = []
    y_list = []
    engine_id_list = []
    
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
        
        features = engine_df[feature_cols].values
        targets = engine_df[target_col].values
        
        n_cycles = len(engine_df)
        
        # Create sequences
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
            engine_id_list.append(engine_id)
    
    return np.array(X_list), np.array(y_list), np.array(engine_id_list)


def split_by_engine(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by engine ID (not by individual samples).
    
    This prevents data leakage where samples from the same engine
    appear in both train and validation sets.
    
    Args:
        df: DataFrame with engine_id column.
        test_size: Fraction of engines for test set.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_df, val_df).
    """
    np.random.seed(random_state)
    
    engine_ids = df['engine_id'].unique()
    n_test = int(len(engine_ids) * test_size)
    
    test_engines = np.random.choice(engine_ids, size=n_test, replace=False)
    train_engines = np.array([e for e in engine_ids if e not in test_engines])
    
    train_df = df[df['engine_id'].isin(train_engines)].copy()
    val_df = df[df['engine_id'].isin(test_engines)].copy()
    
    return train_df, val_df
