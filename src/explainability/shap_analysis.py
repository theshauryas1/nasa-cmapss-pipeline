"""
SHAP Explainability Analysis Module

This module provides model-agnostic explainability using SHAP (SHapley
Additive exPlanations) values. SHAP values provide both global and local
interpretability for machine learning predictions.

Key capabilities:
- Global feature importance (summary plots)
- Local explanation for individual predictions
- Feature importance evolution over degradation lifecycle
- Failure case analysis with explanations

Why SHAP matters for scientific computing:
- Theoretically grounded (game theory)
- Model-agnostic applicability
- Additive feature attributions (contributions sum to prediction)
- Supports both global and local interpretability

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from ..data.ingestion import get_sensor_columns


class SHAPExplainer:
    """
    SHAP-based explainability for RUL prediction models.
    
    This class wraps SHAP functionality with domain-specific analyses
    for turbofan engine degradation.
    
    Supports:
    - Tree-based models (Random Forest, Gradient Boosting)
    - Linear models
    - Generic models (via KernelSHAP, slower)
    
    Example:
        >>> explainer = SHAPExplainer(model, X_train)
        >>> shap_values = explainer.compute_shap_values(X_test)
        >>> explainer.plot_summary()
    """
    
    def __init__(
        self,
        model,
        background_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        model_type: str = 'auto'
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model with predict() method.
            background_data: Training data sample for computing expected values.
            feature_names: Optional list of feature names.
            model_type: 'tree', 'linear', 'kernel', or 'auto'.
        """
        self.model = model
        
        if isinstance(background_data, pd.DataFrame):
            self.feature_names = background_data.columns.tolist()
            self.background = background_data.values
        else:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(background_data.shape[1])]
            self.background = background_data
        
        self.model_type = model_type
        self.explainer_ = None
        self.shap_values_ = None
        self.X_explain_ = None
        
        self._create_explainer()
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer based on model type."""
        model_type = self.model_type
        
        # Auto-detect model type
        if model_type == 'auto':
            model_class = type(self.model).__name__
            
            # Check if it's a wrapper class
            if hasattr(self.model, 'model_'):
                inner_model = self.model.model_
            else:
                inner_model = self.model
            
            inner_class = type(inner_model).__name__
            
            if 'Forest' in inner_class or 'Boosting' in inner_class or 'XGB' in inner_class:
                model_type = 'tree'
            elif 'Linear' in inner_class or 'Ridge' in inner_class or 'Lasso' in inner_class:
                model_type = 'linear'
            else:
                model_type = 'kernel'
        
        # Get the actual prediction model
        if hasattr(self.model, 'model_'):
            predict_model = self.model.model_
        else:
            predict_model = self.model
        
        # Create appropriate explainer
        if model_type == 'tree':
            self.explainer_ = shap.TreeExplainer(predict_model)
        elif model_type == 'linear':
            self.explainer_ = shap.LinearExplainer(predict_model, self.background)
        else:
            # Sample background for efficiency
            n_background = min(100, len(self.background))
            background_sample = shap.sample(self.background, n_background)
            self.explainer_ = shap.KernelExplainer(predict_model.predict, background_sample)
    
    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for the given data.
        
        Args:
            X: Data to explain.
            max_samples: Maximum samples to compute (for efficiency).
        
        Returns:
            SHAP values array.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        self.X_explain_ = X
        self.shap_values_ = self.explainer_.shap_values(X)
        
        return self.shap_values_
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Returns:
            DataFrame with mean absolute SHAP value per feature.
        """
        if self.shap_values_ is None:
            raise ValueError("Must compute SHAP values first")
        
        mean_abs_shap = np.abs(self.shap_values_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)
    
    def explain_instance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        index: int = 0
    ) -> pd.DataFrame:
        """
        Explain a single prediction.
        
        Args:
            X: Data containing the instance.
            index: Index of the instance to explain.
        
        Returns:
            DataFrame with feature contributions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.shap_values_ is None:
            self.compute_shap_values(X)
        
        if index >= len(self.shap_values_):
            raise ValueError(f"Index {index} out of range")
        
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': self.X_explain_[index],
            'shap_value': self.shap_values_[index],
            'abs_contribution': np.abs(self.shap_values_[index])
        })
        
        contributions = contributions.sort_values('abs_contribution', ascending=False)
        
        return contributions.reset_index(drop=True)
    
    def plot_summary(
        self,
        plot_type: str = 'bar',
        max_features: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a SHAP summary plot.
        
        Args:
            plot_type: 'bar' for importance, 'beeswarm' for detailed.
            max_features: Maximum number of features to show.
            figsize: Figure size.
            save_path: Optional path to save the figure.
        """
        if self.shap_values_ is None:
            raise ValueError("Must compute SHAP values first")
        
        plt.figure(figsize=figsize)
        
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values_,
                self.X_explain_,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_features,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values_,
                self.X_explain_,
                feature_names=self.feature_names,
                max_display=max_features,
                show=False
            )
        
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_force(
        self,
        index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Create a force plot for a single prediction.
        
        Args:
            index: Index of instance to explain.
            save_path: Optional path to save.
        
        Returns:
            SHAP force plot object.
        """
        if self.shap_values_ is None:
            raise ValueError("Must compute SHAP values first")
        
        force_plot = shap.force_plot(
            self.explainer_.expected_value,
            self.shap_values_[index],
            self.X_explain_[index],
            feature_names=self.feature_names
        )
        
        if save_path:
            shap.save_html(save_path, force_plot)
        
        return force_plot
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = 'auto',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a dependence plot for a feature.
        
        Shows how SHAP values depend on feature values.
        
        Args:
            feature: Feature to analyze.
            interaction_feature: Feature for coloring (or 'auto').
            figsize: Figure size.
            save_path: Optional path to save.
        """
        if self.shap_values_ is None:
            raise ValueError("Must compute SHAP values first")
        
        plt.figure(figsize=figsize)
        
        feature_idx = self.feature_names.index(feature)
        
        shap.dependence_plot(
            feature_idx,
            self.shap_values_,
            self.X_explain_,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence: {feature}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def analyze_feature_importance_by_rul(
    model,
    X: pd.DataFrame,
    y_rul: np.ndarray,
    rul_bins: List[int] = [0, 30, 60, 100, 150],
    n_samples_per_bin: int = 100
) -> pd.DataFrame:
    """
    Analyze how feature importance changes across the degradation lifecycle.
    
    This is critical for understanding if certain sensors become more
    important as failure approaches.
    
    Args:
        model: Trained model.
        X: Feature DataFrame.
        y_rul: RUL values for each sample.
        rul_bins: Bin edges for RUL grouping.
        n_samples_per_bin: Number of samples per bin for SHAP.
    
    Returns:
        DataFrame with feature importance by RUL bin.
    """
    results = []
    
    for i in range(len(rul_bins) - 1):
        low, high = rul_bins[i], rul_bins[i + 1]
        mask = (y_rul >= low) & (y_rul < high)
        
        if mask.sum() < 10:
            continue
        
        # Sample from this bin
        X_bin = X[mask]
        if len(X_bin) > n_samples_per_bin:
            X_bin = X_bin.sample(n_samples_per_bin, random_state=42)
        
        # Create explainer and compute SHAP
        explainer = SHAPExplainer(model, X.sample(min(100, len(X)), random_state=42))
        shap_values = explainer.compute_shap_values(X_bin)
        
        # Mean absolute SHAP by feature
        importance = np.abs(shap_values).mean(axis=0)
        
        for j, feature in enumerate(X.columns):
            results.append({
                'rul_range': f'{low}-{high}',
                'rul_low': low,
                'rul_high': high,
                'feature': feature,
                'mean_abs_shap': importance[j]
            })
    
    result_df = pd.DataFrame(results)
    
    # Pivot for easier analysis
    pivot_df = result_df.pivot(
        index='feature',
        columns='rul_range',
        values='mean_abs_shap'
    )
    
    return pivot_df


def detect_overfitting(
    model,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: np.ndarray,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: np.ndarray
) -> Dict:
    """
    Detect potential overfitting through error analysis.
    
    Compares SHAP distributions and prediction errors between
    training and test sets.
    
    Args:
        model: Trained model.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
    
    Returns:
        Dictionary with overfitting diagnostics.
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Errors
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    
    # Compute SHAP for both sets (sample for efficiency)
    n_sample = min(200, len(X_train), len(X_test))
    
    train_indices = np.random.choice(len(X_train), n_sample, replace=False)
    test_indices = np.random.choice(len(X_test), n_sample, replace=False)
    
    # SHAP values
    explainer = SHAPExplainer(model, X_train[train_indices])
    
    shap_train = explainer.compute_shap_values(X_train[train_indices])
    shap_test = explainer.compute_shap_values(X_test[test_indices])
    
    # Compare SHAP distributions
    shap_train_mean = np.abs(shap_train).mean()
    shap_test_mean = np.abs(shap_test).mean()
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'rmse_ratio': test_rmse / train_rmse,
        'shap_train_mean': shap_train_mean,
        'shap_test_mean': shap_test_mean,
        'shap_ratio': shap_test_mean / shap_train_mean,
        'likely_overfitting': test_rmse / train_rmse > 1.5,
        'overfitting_severity': 'high' if test_rmse / train_rmse > 2.0 else (
            'moderate' if test_rmse / train_rmse > 1.5 else 'low'
        )
    }


def print_shap_summary(explainer: SHAPExplainer) -> None:
    """
    Print a summary of SHAP analysis results.
    
    Args:
        explainer: Fitted SHAPExplainer with computed SHAP values.
    """
    print("=" * 60)
    print("SHAP EXPLAINABILITY SUMMARY")
    print("=" * 60)
    
    importance = explainer.get_feature_importance()
    
    print(f"\n📊 Global Feature Importance (Top 10):")
    for _, row in importance.head(10).iterrows():
        bar_len = int(row['mean_abs_shap'] / importance['mean_abs_shap'].max() * 20)
        bar = '█' * bar_len
        print(f"  {row['rank']:2d}. {row['feature']:20s}  {bar}  {row['mean_abs_shap']:.4f}")
    
    # Identify sensors vs other features
    sensor_cols = get_sensor_columns()
    sensor_imp = importance[importance['feature'].isin(sensor_cols)]
    other_imp = importance[~importance['feature'].isin(sensor_cols)]
    
    print(f"\n🔧 Feature Type Breakdown:")
    print(f"  • Sensors in top 10: {len(sensor_imp[sensor_imp['rank'] <= 10])}")
    print(f"  • Non-sensors in top 10: {len(other_imp[other_imp['rank'] <= 10])}")
    
    print("=" * 60)
