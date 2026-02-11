"""
NASA C-MAPSS Data Ingestion Module

This module provides functionality for loading and validating the NASA C-MAPSS
(Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation
dataset. The dataset consists of multivariate time-series sensor data from simulated
aircraft engine run-to-failure experiments.

Dataset Structure:
- FD001: Single operating condition, single fault mode (HPC degradation)
- FD002: Six operating conditions, single fault mode (HPC degradation)
- FD003: Single operating condition, two fault modes (HPC + Fan degradation)
- FD004: Six operating conditions, two fault modes (HPC + Fan degradation)

Each record contains:
- Engine unit number (1-indexed)
- Time cycle
- 3 operational settings (altitude, Mach, throttle resolver angle)
- 21 sensor measurements

Author: Scientific Data Pipeline Project
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Column definitions for C-MAPSS dataset
COLUMN_NAMES = (
    ['engine_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Sensor descriptions based on C-MAPSS documentation
SENSOR_DESCRIPTIONS = {
    'sensor_1': 'Total temperature at fan inlet (°R)',
    'sensor_2': 'Total temperature at LPC outlet (°R)',
    'sensor_3': 'Total temperature at HPC outlet (°R)',
    'sensor_4': 'Total temperature at LPT outlet (°R)',
    'sensor_5': 'Pressure at fan inlet (psia)',
    'sensor_6': 'Total pressure in bypass-duct (psia)',
    'sensor_7': 'Total pressure at HPC outlet (psia)',
    'sensor_8': 'Physical fan speed (rpm)',
    'sensor_9': 'Physical core speed (rpm)',
    'sensor_10': 'Engine pressure ratio (P50/P2)',
    'sensor_11': 'Static pressure at HPC outlet (psia)',
    'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psi)',
    'sensor_13': 'Corrected fan speed (rpm)',
    'sensor_14': 'Corrected core speed (rpm)',
    'sensor_15': 'Bypass ratio',
    'sensor_16': 'Burner fuel-air ratio',
    'sensor_17': 'Bleed enthalpy',
    'sensor_18': 'Demanded fan speed (rpm)',
    'sensor_19': 'Demanded corrected fan speed (rpm)',
    'sensor_20': 'HPT coolant bleed (lbm/s)',
    'sensor_21': 'LPT coolant bleed (lbm/s)',
}

# Expected data types
EXPECTED_DTYPES = {
    'engine_id': np.int64,
    'cycle': np.int64,
}


class CMAPSSDataLoader:
    """
    Data loader for NASA C-MAPSS turbofan engine degradation dataset.
    
    This class handles loading, validation, and basic preprocessing of the
    C-MAPSS dataset files. It supports all four sub-datasets (FD001-FD004)
    and provides summary statistics for data exploration.
    
    Attributes:
        data_dir (Path): Directory containing the raw data files.
        available_datasets (List[str]): List of available sub-dataset IDs.
    
    Example:
        >>> loader = CMAPSSDataLoader('data/raw')
        >>> train_df, test_df, rul_df = loader.load_dataset('FD001')
        >>> print(f"Training samples: {len(train_df)}")
    """
    
    VALID_DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
    
    def __init__(self, data_dir: Union[str, Path] = 'data/raw'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing raw C-MAPSS files.
        """
        self.data_dir = Path(data_dir)
        self.available_datasets = self._check_available_datasets()
    
    def _check_available_datasets(self) -> List[str]:
        """Check which datasets are available in the data directory."""
        available = []
        for dataset_id in self.VALID_DATASETS:
            train_file = self.data_dir / f'train_{dataset_id}.txt'
            if train_file.exists():
                available.append(dataset_id)
        return available
    
    def load_dataset(
        self,
        dataset_id: str,
        include_rul: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load a specific C-MAPSS sub-dataset.
        
        Args:
            dataset_id: One of 'FD001', 'FD002', 'FD003', 'FD004'.
            include_rul: Whether to load the RUL ground truth file.
        
        Returns:
            Tuple of (train_df, test_df, rul_df) DataFrames.
            rul_df is None if include_rul is False.
        
        Raises:
            ValueError: If dataset_id is not valid.
            FileNotFoundError: If required files are not found.
        """
        if dataset_id not in self.VALID_DATASETS:
            raise ValueError(
                f"Invalid dataset_id '{dataset_id}'. "
                f"Must be one of {self.VALID_DATASETS}"
            )
        
        # Define file paths
        train_path = self.data_dir / f'train_{dataset_id}.txt'
        test_path = self.data_dir / f'test_{dataset_id}.txt'
        rul_path = self.data_dir / f'RUL_{dataset_id}.txt'
        
        # Check file existence
        for path, name in [(train_path, 'Training'), (test_path, 'Test')]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} file not found: {path}\n"
                    f"Please download the C-MAPSS dataset and place files in {self.data_dir}"
                )
        
        # Load training data
        train_df = self._load_data_file(train_path)
        
        # Load test data
        test_df = self._load_data_file(test_path)
        
        # Load RUL if requested
        rul_df = None
        if include_rul:
            if not rul_path.exists():
                raise FileNotFoundError(f"RUL file not found: {rul_path}")
            rul_df = pd.read_csv(
                rul_path,
                header=None,
                names=['RUL'],
                delim_whitespace=True
            )
            # Add engine_id to match test data
            rul_df['engine_id'] = range(1, len(rul_df) + 1)
            rul_df = rul_df[['engine_id', 'RUL']]
        
        # Validate loaded data
        self._validate_data(train_df, 'train')
        self._validate_data(test_df, 'test')
        
        return train_df, test_df, rul_df
    
    def _load_data_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a single C-MAPSS data file.
        
        Args:
            filepath: Path to the data file.
        
        Returns:
            DataFrame with properly named columns.
        """
        df = pd.read_csv(
            filepath,
            header=None,
            delim_whitespace=True,
            names=COLUMN_NAMES
        )
        
        # Convert engine_id and cycle to integers
        df['engine_id'] = df['engine_id'].astype(np.int64)
        df['cycle'] = df['cycle'].astype(np.int64)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, data_type: str) -> None:
        """
        Validate the loaded data for schema conformance.
        
        Args:
            df: DataFrame to validate.
            data_type: Description for error messages ('train' or 'test').
        
        Raises:
            ValueError: If validation fails.
        """
        # Check column count
        expected_cols = len(COLUMN_NAMES)
        if len(df.columns) != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} columns in {data_type} data, "
                f"got {len(df.columns)}"
            )
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values in {data_type} data")
        
        # Check for negative cycle values
        if (df['cycle'] < 1).any():
            raise ValueError(f"Invalid cycle values (< 1) in {data_type} data")
        
        # Check engine_id starts from 1
        if df['engine_id'].min() < 1:
            raise ValueError(f"Invalid engine_id values in {data_type} data")
    
    def get_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate summary statistics for a loaded dataset.
        
        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            rul_df: Optional RUL DataFrame.
        
        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            'training': {
                'n_engines': train_df['engine_id'].nunique(),
                'n_samples': len(train_df),
                'min_cycles': train_df.groupby('engine_id')['cycle'].max().min(),
                'max_cycles': train_df.groupby('engine_id')['cycle'].max().max(),
                'avg_cycles': train_df.groupby('engine_id')['cycle'].max().mean(),
            },
            'test': {
                'n_engines': test_df['engine_id'].nunique(),
                'n_samples': len(test_df),
                'min_cycles': test_df.groupby('engine_id')['cycle'].max().min(),
                'max_cycles': test_df.groupby('engine_id')['cycle'].max().max(),
                'avg_cycles': test_df.groupby('engine_id')['cycle'].max().mean(),
            },
            'sensors': {
                'n_operational_settings': 3,
                'n_sensors': 21,
                'columns': COLUMN_NAMES,
            }
        }
        
        if rul_df is not None:
            summary['rul'] = {
                'min_rul': rul_df['RUL'].min(),
                'max_rul': rul_df['RUL'].max(),
                'mean_rul': rul_df['RUL'].mean(),
                'std_rul': rul_df['RUL'].std(),
            }
        
        return summary
    
    def print_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Print a formatted summary of the dataset."""
        summary = self.get_summary(train_df, test_df, rul_df)
        
        print("=" * 60)
        print("NASA C-MAPSS Dataset Summary")
        print("=" * 60)
        
        print("\n📊 TRAINING DATA")
        print(f"  • Number of engines: {summary['training']['n_engines']}")
        print(f"  • Total samples: {summary['training']['n_samples']:,}")
        print(f"  • Lifecycle range: {summary['training']['min_cycles']:.0f} - "
              f"{summary['training']['max_cycles']:.0f} cycles")
        print(f"  • Average lifecycle: {summary['training']['avg_cycles']:.1f} cycles")
        
        print("\n🧪 TEST DATA")
        print(f"  • Number of engines: {summary['test']['n_engines']}")
        print(f"  • Total samples: {summary['test']['n_samples']:,}")
        print(f"  • Observed cycles: {summary['test']['min_cycles']:.0f} - "
              f"{summary['test']['max_cycles']:.0f}")
        
        if 'rul' in summary:
            print("\n⏱️  REMAINING USEFUL LIFE (Ground Truth)")
            print(f"  • RUL range: {summary['rul']['min_rul']:.0f} - "
                  f"{summary['rul']['max_rul']:.0f} cycles")
            print(f"  • Mean RUL: {summary['rul']['mean_rul']:.1f} ± "
                  f"{summary['rul']['std_rul']:.1f} cycles")
        
        print("\n🔧 FEATURES")
        print(f"  • Operational settings: {summary['sensors']['n_operational_settings']}")
        print(f"  • Sensor measurements: {summary['sensors']['n_sensors']}")
        print("=" * 60)


def compute_training_rul(df: pd.DataFrame, cap_rul: int = 125) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for training data.
    
    For training data, we know the full run-to-failure trajectory, so
    RUL = max_cycle - current_cycle for each engine.
    
    A piecewise-linear RUL with a cap is commonly used because:
    1. Early degradation is often not detectable
    2. Capping reduces noise in healthy operation regime
    
    Args:
        df: Training DataFrame with 'engine_id' and 'cycle' columns.
        cap_rul: Maximum RUL value (clips higher values). Default 125.
    
    Returns:
        DataFrame with 'RUL' column added.
    """
    df = df.copy()
    
    # Compute max cycle for each engine
    max_cycles = df.groupby('engine_id')['cycle'].max()
    
    # Map to each row
    df['max_cycle'] = df['engine_id'].map(max_cycles)
    
    # Compute RUL
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Apply piecewise-linear cap
    df['RUL'] = df['RUL'].clip(upper=cap_rul)
    
    # Drop helper column
    df = df.drop(columns=['max_cycle'])
    
    return df


def get_sensor_columns() -> List[str]:
    """Return list of sensor column names."""
    return [f'sensor_{i}' for i in range(1, 22)]


def get_operational_columns() -> List[str]:
    """Return list of operational setting column names."""
    return [f'op_setting_{i}' for i in range(1, 4)]


def get_feature_columns() -> List[str]:
    """Return all feature columns (operational settings + sensors)."""
    return get_operational_columns() + get_sensor_columns()
