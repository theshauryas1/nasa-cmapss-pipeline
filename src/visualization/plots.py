"""
Scientific Visualization Module for NASA C-MAPSS Analysis

This module provides publication-quality visualization functions for
turbofan engine degradation analysis. All plots follow scientific
visualization best practices for clarity and reproducibility.

Plot categories:
- Sensor degradation trends
- RUL prediction analysis
- Model comparison
- Explainability visualization
- Statistical analysis plots

Style: Plots use a consistent scientific theme with:
- Clear axis labels with units
- Legible font sizes
- Colorblind-friendly palettes
- High DPI for publication

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ..data.ingestion import get_sensor_columns, SENSOR_DESCRIPTIONS


# Set scientific plot style
def set_scientific_style():
    """Set matplotlib style for scientific publications."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Initialize style on import
set_scientific_style()


def plot_sensor_degradation(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    engine_ids: Optional[List[int]] = None,
    n_engines: int = 5,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot sensor degradation trends over engine lifecycle.
    
    Shows how sensor values evolve as engines approach failure,
    providing insight into degradation patterns.
    
    Args:
        df: DataFrame with sensor columns, engine_id, and cycle.
        sensors: List of sensors to plot (default: top 6 variable sensors).
        engine_ids: Specific engines to show (default: random sample).
        n_engines: Number of engines to sample if not specified.
        figsize: Figure size.
        save_path: Optional path to save figure.
    """
    if sensors is None:
        # Select sensors with high variance
        sensor_cols = [c for c in get_sensor_columns() if c in df.columns]
        variances = df[sensor_cols].var().sort_values(ascending=False)
        sensors = variances.head(6).index.tolist()
    
    if engine_ids is None:
        engine_ids = np.random.choice(df['engine_id'].unique(), n_engines, replace=False)
    
    n_sensors = len(sensors)
    n_rows = (n_sensors + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(engine_ids)))
    
    for i, sensor in enumerate(sensors):
        ax = axes[i]
        
        for j, engine_id in enumerate(engine_ids):
            engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
            ax.plot(
                engine_df['cycle'],
                engine_df[sensor],
                color=colors[j],
                alpha=0.7,
                linewidth=1.5,
                label=f'Engine {engine_id}' if i == 0 else ''
            )
        
        ax.set_xlabel('Cycle')
        ax.set_ylabel(sensor.replace('_', ' ').title())
        ax.set_title(sensor)
        
        # Add degradation arrow
        ax.annotate(
            '', xy=(0.95, 0.1), xytext=(0.05, 0.1),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)
        )
        ax.text(0.5, 0.02, 'Degradation →', transform=ax.transAxes,
                ha='center', fontsize=9, color='gray')
    
    # Hide unused subplots
    for j in range(len(sensors), len(axes)):
        axes[j].set_visible(False)
    
    # Add legend
    if len(engine_ids) <= 10:
        handles = [mpatches.Patch(color=colors[j], label=f'Engine {engine_ids[j]}')
                   for j in range(len(engine_ids))]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    fig.suptitle('Sensor Degradation Trends Over Engine Lifecycle', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_rul_distribution(
    train_df: pd.DataFrame,
    test_rul: Optional[pd.DataFrame] = None,
    rul_column: str = 'RUL',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot RUL distribution for training and test data.
    
    Args:
        train_df: Training DataFrame with RUL column.
        test_rul: Optional test RUL DataFrame.
        rul_column: Name of RUL column.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Training RUL distribution
    ax1 = axes[0]
    sns.histplot(train_df[rul_column], bins=50, ax=ax1, color='steelblue', alpha=0.7)
    ax1.axvline(train_df[rul_column].mean(), color='red', linestyle='--',
                label=f'Mean: {train_df[rul_column].mean():.1f}')
    ax1.set_xlabel('Remaining Useful Life (cycles)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Training Data RUL Distribution')
    ax1.legend()
    
    # Test RUL distribution (if provided)
    ax2 = axes[1]
    if test_rul is not None:
        sns.histplot(test_rul['RUL'], bins=30, ax=ax2, color='coral', alpha=0.7)
        ax2.axvline(test_rul['RUL'].mean(), color='red', linestyle='--',
                    label=f'Mean: {test_rul["RUL"].mean():.1f}')
        ax2.set_xlabel('Remaining Useful Life (cycles)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Test Data Ground Truth RUL Distribution')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Test RUL not provided', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Test Data RUL Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create prediction vs actual scatter plot with regression line.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        model_name: Name of the model for title.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=20, c='steelblue')
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add ±20% error bands
    ax.fill_between(
        [0, max_val],
        [0, max_val * 0.8],
        [0, max_val * 1.2],
        alpha=0.1, color='green', label='±20% Error Band'
    )
    
    # Metrics annotation
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title(f'{model_name}: Predicted vs Actual RUL')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot residual analysis for RUL predictions.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        model_name: Model name for title.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.3, s=20, c='steelblue')
    ax1.axhline(0, color='red', linestyle='--')
    ax1.axhline(residuals.mean(), color='orange', linestyle=':',
                label=f'Mean: {residuals.mean():.2f}')
    ax1.set_xlabel('Predicted RUL')
    ax1.set_ylabel('Residual (Pred - Actual)')
    ax1.set_title('Residuals vs Predicted')
    ax1.legend()
    
    # Residuals vs actual RUL
    ax2 = axes[1]
    ax2.scatter(y_true, residuals, alpha=0.3, s=20, c='coral')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Actual RUL')
    ax2.set_ylabel('Residual (Pred - Actual)')
    ax2.set_title('Residuals vs Actual RUL')
    
    # Residual distribution
    ax3 = axes[2]
    sns.histplot(residuals, bins=50, ax=ax3, color='steelblue', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(residuals.mean(), color='orange', linestyle=':',
                label=f'Mean: {residuals.mean():.2f}')
    ax3.set_xlabel('Residual (Pred - Actual)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution')
    ax3.legend()
    
    fig.suptitle(f'{model_name}: Residual Analysis', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = 'Sensor Correlation Matrix',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a correlation heatmap with hierarchical clustering.
    
    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create clustered heatmap
    sns.heatmap(
        corr_matrix,
        center=0,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        annot=False,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    ax.set_title(title, fontsize=14)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = 'Random Forest',
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        model_name: Model name for title.
        top_n: Number of top features to show.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top features
    top_features = importance_df.head(top_n).sort_values('importance')
    
    # Color by sensor type
    colors = []
    sensor_cols = get_sensor_columns()
    for feature in top_features['feature']:
        if feature in sensor_cols:
            colors.append('steelblue')
        elif 'roll' in feature or 'diff' in feature or 'ewm' in feature:
            colors.append('coral')
        else:
            colors.append('forestgreen')
    
    bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
    
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title(f'{model_name}: Top {top_n} Feature Importance')
    
    # Legend
    sensor_patch = mpatches.Patch(color='steelblue', label='Sensor')
    derived_patch = mpatches.Patch(color='coral', label='Derived Feature')
    other_patch = mpatches.Patch(color='forestgreen', label='Other')
    ax.legend(handles=[sensor_patch, derived_patch, other_patch], loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'rmse',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Bar chart comparing multiple models.
    
    Args:
        comparison_df: DataFrame with 'model' column and metrics.
        metric: Metric to compare.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))
    
    bars = ax.bar(comparison_df['model'], comparison_df[metric], color=colors)
    
    # Add value labels
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', fontsize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison: {metric.upper()}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_engine_predictions(
    df: pd.DataFrame,
    engine_id: int,
    y_true_col: str = 'RUL',
    y_pred_col: str = 'RUL_pred',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot RUL predictions for a single engine over its lifecycle.
    
    Args:
        df: DataFrame with predictions, true RUL, cycle, engine_id.
        engine_id: Engine to visualize.
        y_true_col: Column for true RUL.
        y_pred_col: Column for predicted RUL.
        figsize: Figure size.
        save_path: Optional path to save.
    """
    engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(engine_df['cycle'], engine_df[y_true_col],
            'b-', linewidth=2, label='True RUL')
    ax.plot(engine_df['cycle'], engine_df[y_pred_col],
            'r--', linewidth=2, label='Predicted RUL')
    
    # Fill error region
    ax.fill_between(
        engine_df['cycle'],
        engine_df[y_true_col],
        engine_df[y_pred_col],
        alpha=0.2, color='gray', label='Prediction Error'
    )
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title(f'Engine {engine_id}: RUL Prediction Over Time')
    ax.legend()
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_summary_dashboard(
    train_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    importance_df: pd.DataFrame,
    model_name: str = 'Random Forest',
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive summary dashboard.
    
    Combines multiple visualizations into a single figure.
    
    Args:
        train_df: Training DataFrame.
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        importance_df: Feature importance DataFrame.
        model_name: Model name.
        save_path: Optional path to save.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # 1. RUL Distribution
    sns.histplot(train_df['RUL'], bins=50, ax=ax1, color='steelblue', alpha=0.7)
    ax1.set_xlabel('RUL')
    ax1.set_title('Training RUL Distribution')
    
    # 2. Prediction vs Actual
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual RUL')
    ax2.set_ylabel('Predicted RUL')
    ax2.set_title('Predicted vs Actual')
    
    # 3. Residual Distribution
    residuals = y_pred - y_true
    sns.histplot(residuals, bins=50, ax=ax3, color='coral', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_xlabel('Residual')
    ax3.set_title('Residual Distribution')
    
    # 4. Top 10 Features
    top_10 = importance_df.head(10).sort_values('importance')
    ax4.barh(top_10['feature'], top_10['importance'], color='steelblue')
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Features')
    
    # 5. Error by RUL Range
    rul_bins = [0, 30, 60, 100, 150]
    errors_by_bin = []
    for i in range(len(rul_bins) - 1):
        mask = (y_true >= rul_bins[i]) & (y_true < rul_bins[i + 1])
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
            errors_by_bin.append(rmse)
        else:
            errors_by_bin.append(0)
    
    bin_labels = [f'{rul_bins[i]}-{rul_bins[i+1]}' for i in range(len(rul_bins) - 1)]
    ax5.bar(bin_labels, errors_by_bin, color='coral')
    ax5.set_xlabel('RUL Range')
    ax5.set_ylabel('RMSE')
    ax5.set_title('Error by RUL Range')
    
    # 6. Metrics Summary
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    ax6.axis('off')
    metrics_text = (
        f"Model: {model_name}\n\n"
        f"RMSE: {rmse:.2f} cycles\n"
        f"MAE: {mae:.2f} cycles\n"
        f"R²: {r2:.4f}\n\n"
        f"Samples: {len(y_true):,}\n"
        f"Bias: {residuals.mean():+.2f} cycles"
    )
    ax6.text(0.5, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=14, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax6.set_title('Performance Summary')
    
    fig.suptitle(f'NASA C-MAPSS Analysis Dashboard: {model_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
