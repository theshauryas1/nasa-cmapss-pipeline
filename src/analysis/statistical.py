"""
Statistical Analysis Module for NASA C-MAPSS Data

This module provides comprehensive statistical analysis functions for
understanding sensor behavior, degradation patterns, and feature relationships
in turbofan engine degradation data.

Key analyses include:
- Correlation analysis (sensor-RUL relationships)
- Distribution analysis (health vs. degraded states)
- Trend analysis (monotonicity, stationarity)
- Statistical tests for significance

These analyses are crucial for scientific understanding before applying
machine learning models.

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp, mannwhitneyu

from ..data.ingestion import get_sensor_columns, SENSOR_DESCRIPTIONS


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.
    
    Args:
        df: Input DataFrame.
        columns: Columns to include (default: all sensor columns).
        method: 'pearson' or 'spearman'.
    
    Returns:
        Correlation matrix as DataFrame.
    """
    if columns is None:
        columns = [c for c in get_sensor_columns() if c in df.columns]
    
    if method == 'pearson':
        return df[columns].corr(method='pearson')
    elif method == 'spearman':
        return df[columns].corr(method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'")


def compute_sensor_rul_correlation(
    df: pd.DataFrame,
    rul_column: str = 'RUL',
    method: str = 'both'
) -> pd.DataFrame:
    """
    Compute correlation between each sensor and RUL.
    
    This analysis identifies which sensors are most predictive of
    remaining useful life, supporting feature selection decisions.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        rul_column: Name of RUL column.
        method: 'pearson', 'spearman', or 'both'.
    
    Returns:
        DataFrame with correlation coefficients and p-values.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    if rul_column not in df.columns:
        raise ValueError(f"RUL column '{rul_column}' not found")
    
    results = []
    
    for sensor in sensor_cols:
        row = {'sensor': sensor}
        
        # Remove any NaN values for correlation computation
        valid_mask = ~(df[sensor].isna() | df[rul_column].isna())
        x = df.loc[valid_mask, sensor]
        y = df.loc[valid_mask, rul_column]
        
        if method in ['pearson', 'both']:
            corr, pval = pearsonr(x, y)
            row['pearson_corr'] = corr
            row['pearson_pval'] = pval
        
        if method in ['spearman', 'both']:
            corr, pval = spearmanr(x, y)
            row['spearman_corr'] = corr
            row['spearman_pval'] = pval
        
        # Add sensor description if available
        if sensor in SENSOR_DESCRIPTIONS:
            row['description'] = SENSOR_DESCRIPTIONS[sensor]
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Sort by absolute correlation
    if 'spearman_corr' in result_df.columns:
        result_df['abs_corr'] = result_df['spearman_corr'].abs()
    else:
        result_df['abs_corr'] = result_df['pearson_corr'].abs()
    
    result_df = result_df.sort_values('abs_corr', ascending=False)
    result_df = result_df.drop(columns=['abs_corr'])
    
    return result_df.reset_index(drop=True)


def analyze_sensor_monotonicity(
    df: pd.DataFrame,
    target_col: str = 'RUL'
) -> pd.DataFrame:
    """
    Analyze monotonicity of sensor values with respect to degradation.
    
    Monotonicity is an important property for prognostics - sensors that
    increase or decrease monotonically with degradation are more reliable
    indicators of remaining useful life.
    
    Uses Spearman correlation as a proxy for monotonicity.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        target_col: Target column (RUL or cycle).
    
    Returns:
        DataFrame with monotonicity metrics for each sensor.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    results = []
    
    for sensor in sensor_cols:
        valid_mask = ~(df[sensor].isna() | df[target_col].isna())
        x = df.loc[valid_mask, sensor].values
        y = df.loc[valid_mask, target_col].values
        
        # Spearman correlation as monotonicity measure
        spearman_corr, _ = spearmanr(x, y)
        
        # Count monotonic segments per engine
        mono_scores = []
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id][sensor].values
            if len(engine_data) > 1:
                # Calculate fraction of consistent direction changes
                diffs = np.diff(engine_data)
                if len(diffs) > 0:
                    # Fraction of same-sign differences
                    sign_consistency = np.abs(np.sum(np.sign(diffs))) / len(diffs)
                    mono_scores.append(sign_consistency)
        
        avg_monotonicity = np.mean(mono_scores) if mono_scores else 0
        
        results.append({
            'sensor': sensor,
            'spearman_with_rul': spearman_corr,
            'direction': 'increasing' if spearman_corr > 0 else 'decreasing',
            'monotonicity_score': avg_monotonicity,
            'is_strongly_monotonic': avg_monotonicity > 0.7
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('monotonicity_score', ascending=False)
    
    return result_df.reset_index(drop=True)


def compare_health_stages(
    df: pd.DataFrame,
    rul_threshold: int = 50,
    rul_column: str = 'RUL'
) -> pd.DataFrame:
    """
    Compare sensor distributions between healthy and degraded states.
    
    Uses statistical tests to identify sensors that show significant
    distribution shifts as engines approach failure.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        rul_threshold: RUL value to separate healthy/degraded states.
        rul_column: Name of RUL column.
    
    Returns:
        DataFrame with distribution comparison statistics.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    healthy = df[df[rul_column] > rul_threshold]
    degraded = df[df[rul_column] <= rul_threshold]
    
    results = []
    
    for sensor in sensor_cols:
        healthy_vals = healthy[sensor].dropna()
        degraded_vals = degraded[sensor].dropna()
        
        if len(healthy_vals) == 0 or len(degraded_vals) == 0:
            continue
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_pval = ks_2samp(healthy_vals, degraded_vals)
        
        # Mann-Whitney U test (non-parametric location test)
        mw_stat, mw_pval = mannwhitneyu(
            healthy_vals, degraded_vals, alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (healthy_vals.std()**2 + degraded_vals.std()**2) / 2
        )
        if pooled_std > 0:
            cohens_d = (healthy_vals.mean() - degraded_vals.mean()) / pooled_std
        else:
            cohens_d = 0
        
        results.append({
            'sensor': sensor,
            'healthy_mean': healthy_vals.mean(),
            'degraded_mean': degraded_vals.mean(),
            'mean_shift': degraded_vals.mean() - healthy_vals.mean(),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'mw_pvalue': mw_pval,
            'cohens_d': cohens_d,
            'effect_size': (
                'large' if abs(cohens_d) > 0.8 else
                'medium' if abs(cohens_d) > 0.5 else
                'small' if abs(cohens_d) > 0.2 else
                'negligible'
            )
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('ks_statistic', ascending=False)
    
    return result_df.reset_index(drop=True)


def analyze_sensor_variance(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze variance characteristics of each sensor.
    
    Identifies sensors with:
    - Near-zero variance (constant, uninformative)
    - High variance (potentially noisy)
    - Coefficient of variation (standardized variability)
    
    Args:
        df: DataFrame with sensor columns.
    
    Returns:
        DataFrame with variance statistics.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    results = []
    
    for sensor in sensor_cols:
        values = df[sensor].dropna()
        
        mean_val = values.mean()
        std_val = values.std()
        var_val = values.var()
        
        # Coefficient of variation (meaningful for positive values)
        if mean_val != 0:
            cv = std_val / abs(mean_val)
        else:
            cv = np.inf if std_val > 0 else 0
        
        results.append({
            'sensor': sensor,
            'mean': mean_val,
            'std': std_val,
            'variance': var_val,
            'min': values.min(),
            'max': values.max(),
            'range': values.max() - values.min(),
            'cv': cv,
            'is_constant': var_val < 1e-10,
            'is_high_variance': cv > 1.0
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('variance', ascending=False)
    
    return result_df.reset_index(drop=True)


def compute_failure_proximity_stats(
    df: pd.DataFrame,
    rul_column: str = 'RUL',
    proximity_windows: List[int] = [10, 30, 50, 100]
) -> pd.DataFrame:
    """
    Analyze how sensor statistics change in the final cycles before failure.
    
    This analysis helps understand the precursors to failure and can
    inform threshold-based alerting strategies.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        rul_column: Name of RUL column.
        proximity_windows: RUL thresholds to analyze.
    
    Returns:
        DataFrame with failure-proximate statistics.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    results = []
    
    for sensor in sensor_cols:
        sensor_stats = {'sensor': sensor}
        
        # Far from failure (healthy baseline)
        healthy_data = df[df[rul_column] > max(proximity_windows)][sensor]
        sensor_stats['healthy_mean'] = healthy_data.mean()
        sensor_stats['healthy_std'] = healthy_data.std()
        
        # Stats at each proximity window
        for window in proximity_windows:
            window_data = df[df[rul_column] <= window][sensor]
            sensor_stats[f'mean_rul_le_{window}'] = window_data.mean()
            
            # Percent change from healthy
            if sensor_stats['healthy_mean'] != 0:
                pct_change = (
                    (window_data.mean() - sensor_stats['healthy_mean']) /
                    abs(sensor_stats['healthy_mean']) * 100
                )
            else:
                pct_change = 0
            sensor_stats[f'pct_change_rul_le_{window}'] = pct_change
        
        results.append(sensor_stats)
    
    return pd.DataFrame(results)


def print_statistical_summary(
    df: pd.DataFrame,
    rul_column: str = 'RUL'
) -> None:
    """
    Print a comprehensive statistical summary of the dataset.
    
    Args:
        df: DataFrame with sensor columns and RUL.
        rul_column: Name of RUL column.
    """
    print("=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Sensor-RUL correlation
    print("\n📊 TOP 10 SENSORS BY |CORRELATION WITH RUL|")
    print("-" * 50)
    corr_df = compute_sensor_rul_correlation(df, rul_column)
    for _, row in corr_df.head(10).iterrows():
        print(f"  {row['sensor']:12s}  ρ={row['spearman_corr']:+.3f}  "
              f"(p={row['spearman_pval']:.2e})")
    
    # Variance analysis
    print("\n📉 CONSTANT SENSORS (Zero Variance)")
    print("-" * 50)
    var_df = analyze_sensor_variance(df)
    constant = var_df[var_df['is_constant']]
    if len(constant) > 0:
        for sensor in constant['sensor'].tolist():
            print(f"  • {sensor}")
    else:
        print("  None detected")
    
    # Health stage comparison
    print("\n🔬 SENSORS WITH LARGE EFFECT SIZE (Healthy vs Degraded)")
    print("-" * 50)
    stage_df = compare_health_stages(df, rul_threshold=50, rul_column=rul_column)
    large_effect = stage_df[stage_df['effect_size'] == 'large']
    for _, row in large_effect.iterrows():
        print(f"  {row['sensor']:12s}  d={row['cohens_d']:+.2f}  "
              f"shift={row['mean_shift']:+.3f}")
    
    print("\n" + "=" * 70)
