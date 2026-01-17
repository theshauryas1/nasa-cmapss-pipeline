"""
Model Evaluation Module for RUL Prediction

This module provides comprehensive evaluation metrics for RUL prediction
models, including both standard regression metrics and domain-specific
metrics used in prognostics (e.g., NASA scoring function).

Metrics included:
- Standard: RMSE, MAE, R², MAPE
- NASA Scoring Function (asymmetric penalty)
- Early vs. Late prediction analysis
- Engine-level aggregated metrics

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
    
    Returns:
        RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
    
    Returns:
        MAE value.
    """
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
    
    Returns:
        R² value.
    """
    return r2_score(y_true, y_pred)


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0
) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        epsilon: Small value to avoid division by zero.
    
    Returns:
        MAPE value (as percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    y_true_safe = np.maximum(np.abs(y_true), epsilon)
    
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute NASA PHM scoring function (asymmetric penalty).
    
    This scoring function penalizes late predictions (predicting failure
    later than actual) more heavily than early predictions, since late
    predictions are more dangerous in safety-critical systems.
    
    Score formula:
    - If d < 0 (early prediction): exp(-d/13) - 1
    - If d >= 0 (late prediction): exp(d/10) - 1
    
    where d = predicted_RUL - true_RUL
    
    Lower scores are better. A perfect prediction gives score 0.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
    
    Returns:
        NASA score (sum over all samples).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    d = y_pred - y_true  # Prediction error
    
    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # Early prediction (less penalty)
        np.exp(d / 10) - 1    # Late prediction (more penalty)
    )
    
    return np.sum(scores)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
    
    Returns:
        Dictionary with all metric values.
    """
    return {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
        'nasa_score': compute_nasa_score(y_true, y_pred),
        'n_samples': len(y_true)
    }


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rul_bins: List[int] = [0, 25, 50, 75, 100, 125, 200]
) -> pd.DataFrame:
    """
    Analyze prediction errors across different RUL ranges.
    
    This analysis reveals whether the model performs better at
    certain lifecycle stages (e.g., near failure vs. healthy).
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        rul_bins: Bin edges for RUL grouping.
    
    Returns:
        DataFrame with error analysis by RUL bin.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    results = []
    
    for i in range(len(rul_bins) - 1):
        low, high = rul_bins[i], rul_bins[i + 1]
        mask = (y_true >= low) & (y_true < high)
        
        if mask.sum() == 0:
            continue
        
        true_subset = y_true[mask]
        pred_subset = y_pred[mask]
        errors = pred_subset - true_subset
        
        results.append({
            'rul_range': f'{low}-{high}',
            'n_samples': mask.sum(),
            'mean_error': errors.mean(),  # Positive = late prediction
            'rmse': compute_rmse(true_subset, pred_subset),
            'mae': compute_mae(true_subset, pred_subset),
            'pct_early': (errors < 0).mean() * 100,
            'pct_late': (errors > 0).mean() * 100,
            'mean_true_rul': true_subset.mean()
        })
    
    return pd.DataFrame(results)


def compute_engine_level_metrics(
    df: pd.DataFrame,
    y_true_col: str = 'RUL',
    y_pred_col: str = 'RUL_pred',
    engine_id_col: str = 'engine_id'
) -> pd.DataFrame:
    """
    Compute metrics aggregated at the engine level.
    
    For prognostics, we often care about per-engine accuracy,
    especially at the final cycles before failure.
    
    Args:
        df: DataFrame with predictions and true values.
        y_true_col: Column name for true RUL.
        y_pred_col: Column name for predicted RUL.
        engine_id_col: Column name for engine ID.
    
    Returns:
        DataFrame with per-engine metrics.
    """
    results = []
    
    for engine_id in df[engine_id_col].unique():
        engine_df = df[df[engine_id_col] == engine_id]
        
        y_true = engine_df[y_true_col].values
        y_pred = engine_df[y_pred_col].values
        
        # Final prediction (at minimum RUL)
        final_idx = y_true.argmin()
        final_true = y_true[final_idx]
        final_pred = y_pred[final_idx]
        
        results.append({
            'engine_id': engine_id,
            'n_cycles': len(engine_df),
            'rmse': compute_rmse(y_true, y_pred),
            'mae': compute_mae(y_true, y_pred),
            'final_true_rul': final_true,
            'final_pred_rul': final_pred,
            'final_error': final_pred - final_true,
            'nasa_score': compute_nasa_score(y_true, y_pred)
        })
    
    result_df = pd.DataFrame(results)
    
    # Add summary row
    summary = pd.DataFrame([{
        'engine_id': 'MEAN',
        'n_cycles': result_df['n_cycles'].mean(),
        'rmse': result_df['rmse'].mean(),
        'mae': result_df['mae'].mean(),
        'final_true_rul': result_df['final_true_rul'].mean(),
        'final_pred_rul': result_df['final_pred_rul'].mean(),
        'final_error': result_df['final_error'].mean(),
        'nasa_score': result_df['nasa_score'].mean()
    }])
    
    return pd.concat([result_df, summary], ignore_index=True)


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        y_true: True RUL values.
        predictions: Dictionary mapping model names to predictions.
    
    Returns:
        DataFrame comparing all models across metrics.
    """
    results = []
    
    for model_name, y_pred in predictions.items():
        metrics = compute_all_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    result_df = pd.DataFrame(results)
    result_df = result_df[['model', 'rmse', 'mae', 'r2', 'mape', 'nasa_score']]
    
    return result_df.sort_values('rmse')


def print_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> None:
    """
    Print a comprehensive evaluation summary.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        model_name: Name of the model for display.
    """
    print("=" * 60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    
    metrics = compute_all_metrics(y_true, y_pred)
    
    print(f"\n📊 Overall Metrics (n={metrics['n_samples']:,}):")
    print(f"  • RMSE: {metrics['rmse']:.2f} cycles")
    print(f"  • MAE: {metrics['mae']:.2f} cycles")
    print(f"  • R²: {metrics['r2']:.4f}")
    print(f"  • MAPE: {metrics['mape']:.1f}%")
    print(f"  • NASA Score: {metrics['nasa_score']:.2f}")
    
    # Error analysis by RUL range
    print("\n📈 Error Analysis by RUL Range:")
    error_df = analyze_prediction_errors(y_true, y_pred)
    print(error_df.to_string(index=False))
    
    # Prediction bias
    errors = y_pred - y_true
    early_pct = (errors < 0).mean() * 100
    late_pct = (errors > 0).mean() * 100
    
    print(f"\n🎯 Prediction Bias:")
    print(f"  • Early predictions: {early_pct:.1f}%")
    print(f"  • Late predictions: {late_pct:.1f}%")
    print(f"  • Mean bias: {errors.mean():+.2f} cycles")
    
    print("=" * 60)
