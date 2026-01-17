"""
Correlation and Feature Relationship Analysis Module

This module provides advanced correlation analysis functions including:
- Inter-sensor correlation matrices
- Time-lagged cross-correlation
- PCA-based dimensionality analysis
- Sensor clustering by behavior patterns

These analyses support understanding of multivariate sensor relationships
and can inform feature selection and dimensionality reduction strategies.

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..data.ingestion import get_sensor_columns


def compute_inter_sensor_correlation(
    df: pd.DataFrame,
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute correlation matrix between all sensor pairs.
    
    Args:
        df: DataFrame with sensor columns.
        method: 'pearson' or 'spearman'.
    
    Returns:
        Correlation matrix as DataFrame.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    return df[sensor_cols].corr(method=method)


def find_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """
    Find sensor pairs with correlation above threshold.
    
    Useful for identifying redundant sensors that could be removed
    to reduce dimensionality without losing information.
    
    Args:
        corr_matrix: Correlation matrix.
        threshold: Minimum |correlation| to report.
    
    Returns:
        List of (sensor1, sensor2, correlation) tuples.
    """
    pairs = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append((cols[i], cols[j], corr))
    
    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return pairs


def compute_lagged_correlation(
    df: pd.DataFrame,
    sensor1: str,
    sensor2: str,
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Compute time-lagged cross-correlation between two sensors.
    
    Analyzes if changes in one sensor precede changes in another,
    which could indicate causal relationships in degradation.
    
    Args:
        df: DataFrame with sensor columns and engine_id.
        sensor1: First sensor name.
        sensor2: Second sensor name.
        max_lag: Maximum lag (in cycles) to compute.
    
    Returns:
        DataFrame with lag and correlation for each lag value.
    """
    results = []
    
    for lag in range(-max_lag, max_lag + 1):
        correlations = []
        
        for engine_id in df['engine_id'].unique():
            engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
            
            s1 = engine_df[sensor1].values
            s2 = engine_df[sensor2].values
            
            if len(s1) <= abs(lag):
                continue
            
            if lag >= 0:
                s1_aligned = s1[lag:]
                s2_aligned = s2[:len(s1) - lag]
            else:
                s1_aligned = s1[:len(s1) + lag]
                s2_aligned = s2[-lag:]
            
            if len(s1_aligned) > 2:
                corr, _ = stats.pearsonr(s1_aligned, s2_aligned)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            results.append({
                'lag': lag,
                'correlation': np.mean(correlations),
                'std': np.std(correlations),
                'n_engines': len(correlations)
            })
    
    return pd.DataFrame(results)


def perform_pca_analysis(
    df: pd.DataFrame,
    n_components: Optional[int] = None
) -> Dict:
    """
    Perform PCA on sensor data for dimensionality analysis.
    
    Helps understand:
    - How many principal components capture most variance
    - Which sensors contribute most to each component
    - Whether dimensionality reduction is feasible
    
    Args:
        df: DataFrame with sensor columns.
        n_components: Number of components (default: all).
    
    Returns:
        Dictionary with PCA results and analysis.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[sensor_cols].dropna())
    
    # Fit PCA
    if n_components is None:
        n_components = len(sensor_cols)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Compute cumulative explained variance
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for 95% variance
    n_for_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=sensor_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Top contributors for each component
    top_contributors = {}
    for i in range(min(5, n_components)):  # First 5 components
        pc = f'PC{i+1}'
        abs_loadings = loadings[pc].abs().sort_values(ascending=False)
        top_contributors[pc] = abs_loadings.head(5).index.tolist()
    
    return {
        'pca': pca,
        'transformed_data': X_pca,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_var,
        'n_components_for_95_var': n_for_95,
        'loadings': loadings,
        'top_contributors': top_contributors,
        'sensor_columns': sensor_cols
    }


def cluster_sensors(
    df: pd.DataFrame,
    n_clusters: int = 5,
    method: str = 'correlation'
) -> pd.DataFrame:
    """
    Cluster sensors based on their behavior patterns.
    
    Groups sensors that behave similarly, which can help:
    - Identify redundant sensors
    - Create sensor groups for analysis
    - Understand sensor subsystems
    
    Args:
        df: DataFrame with sensor columns.
        n_clusters: Number of clusters to create.
        method: 'correlation' or 'euclidean'.
    
    Returns:
        DataFrame with sensor-to-cluster assignments.
    """
    sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
    
    # Compute correlation matrix
    corr_matrix = df[sensor_cols].corr(method='spearman')
    
    if method == 'correlation':
        # Convert correlation to distance (1 - |corr|)
        distance_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(distance_matrix, 0)  # Distance to self is 0
    else:
        # Euclidean distance on standardized data
        scaler = StandardScaler()
        X = scaler.fit_transform(df[sensor_cols].T)
        from scipy.spatial.distance import pdist, squareform
        distance_matrix = squareform(pdist(X, metric='euclidean'))
    
    # Hierarchical clustering
    condensed = squareform(distance_matrix)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    result = pd.DataFrame({
        'sensor': sensor_cols,
        'cluster': clusters
    })
    
    # Add cluster summary
    for cluster_id in range(1, n_clusters + 1):
        cluster_sensors = result[result['cluster'] == cluster_id]['sensor'].tolist()
        print(f"Cluster {cluster_id}: {cluster_sensors}")
    
    return result


def compute_temporal_correlation_evolution(
    df: pd.DataFrame,
    sensor: str,
    target_col: str = 'RUL',
    n_windows: int = 5
) -> pd.DataFrame:
    """
    Analyze how sensor-RUL correlation evolves over the lifecycle.
    
    Sensors may become more or less predictive as degradation progresses.
    This analysis reveals lifecycle-dependent sensor behavior.
    
    Args:
        df: DataFrame with sensor, RUL, and engine_id columns.
        sensor: Sensor to analyze.
        target_col: Target column (usually RUL).
        n_windows: Number of lifecycle windows to analyze.
    
    Returns:
        DataFrame with correlation at different lifecycle stages.
    """
    results = []
    
    # Compute normalized lifecycle position
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df_temp = df.copy()
    df_temp['lifecycle_pct'] = df_temp['cycle'] / max_cycles * 100
    
    # Define windows
    window_edges = np.linspace(0, 100, n_windows + 1)
    
    for i in range(n_windows):
        low, high = window_edges[i], window_edges[i + 1]
        
        mask = (df_temp['lifecycle_pct'] >= low) & (df_temp['lifecycle_pct'] < high)
        window_df = df_temp[mask]
        
        if len(window_df) > 10:
            corr, pval = stats.spearmanr(
                window_df[sensor].dropna(),
                window_df[target_col].dropna()
            )
        else:
            corr, pval = np.nan, np.nan
        
        results.append({
            'lifecycle_window': f'{low:.0f}-{high:.0f}%',
            'window_start': low,
            'window_end': high,
            'n_samples': len(window_df),
            'correlation': corr,
            'p_value': pval
        })
    
    return pd.DataFrame(results)


def print_correlation_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of correlation analysis results.
    
    Args:
        df: DataFrame with sensor columns.
    """
    print("=" * 70)
    print("CORRELATION ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Inter-sensor correlations
    corr_matrix = compute_inter_sensor_correlation(df)
    high_pairs = find_highly_correlated_pairs(corr_matrix, threshold=0.95)
    
    print("\n🔗 HIGHLY CORRELATED SENSOR PAIRS (|ρ| ≥ 0.95)")
    print("-" * 50)
    if high_pairs:
        for s1, s2, corr in high_pairs[:10]:
            print(f"  {s1} ↔ {s2}: ρ = {corr:+.3f}")
    else:
        print("  No pairs found with |ρ| ≥ 0.95")
    
    # PCA summary
    print("\n📐 PCA DIMENSIONALITY ANALYSIS")
    print("-" * 50)
    pca_results = perform_pca_analysis(df)
    print(f"  Components for 95% variance: {pca_results['n_components_for_95_var']}")
    print(f"  Variance explained by PC1: {pca_results['explained_variance'][0]:.1%}")
    print(f"  Variance explained by PC1-3: {pca_results['cumulative_variance'][2]:.1%}")
    
    print("\n  Top contributors to PC1:")
    for sensor in pca_results['top_contributors'].get('PC1', [])[:5]:
        loading = pca_results['loadings'].loc[sensor, 'PC1']
        print(f"    • {sensor}: {loading:+.3f}")
    
    print("\n" + "=" * 70)
