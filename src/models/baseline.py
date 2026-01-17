"""
Baseline Machine Learning Models for RUL Prediction

This module provides baseline models for Remaining Useful Life (RUL)
prediction using simple, interpretable approaches. These baselines
establish performance benchmarks and provide insights through
coefficient analysis.

Models included:
- Linear Regression (with regularization options)
- Elastic Net (L1+L2 regularization)

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict

from ..data.ingestion import get_feature_columns


class LinearRULPredictor:
    """
    Linear regression model for RUL prediction.
    
    This serves as an interpretable baseline that allows analysis of
    feature importance through coefficient inspection.
    
    Supports regularization options:
    - 'none': Standard OLS
    - 'ridge': L2 regularization
    - 'lasso': L1 regularization
    - 'elasticnet': Combined L1+L2
    
    Example:
        >>> model = LinearRULPredictor(regularization='ridge', alpha=1.0)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> coefficients = model.get_feature_importance()
    """
    
    def __init__(
        self,
        regularization: str = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        standardize: bool = True
    ):
        """
        Initialize the linear predictor.
        
        Args:
            regularization: Type of regularization ('none', 'ridge', 'lasso', 'elasticnet').
            alpha: Regularization strength.
            l1_ratio: L1 ratio for ElasticNet (0=Ridge, 1=Lasso).
            standardize: Whether to standardize features before fitting.
        """
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.standardize = standardize
        
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def _create_model(self):
        """Create the underlying sklearn model."""
        if self.regularization == 'none':
            return LinearRegression()
        elif self.regularization == 'ridge':
            return Ridge(alpha=self.alpha)
        elif self.regularization == 'lasso':
            return Lasso(alpha=self.alpha)
        elif self.regularization == 'elasticnet':
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'LinearRULPredictor':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix.
            y: Target values (RUL).
            feature_names: Optional list of feature names.
        
        Returns:
            self (for method chaining).
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Fit model
        self.model_ = self._create_model()
        self.model_.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict RUL for new data.
        
        Args:
            X: Feature matrix.
        
        Returns:
            Predicted RUL values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.standardize:
            X = self.scaler_.transform(X)
        
        predictions = self.model_.predict(X)
        
        # Clip to non-negative values (RUL can't be negative)
        return np.clip(predictions, 0, None)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.
        
        Returns:
            DataFrame with feature names, coefficients, and absolute importance.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        coefficients = self.model_.coef_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix.
            y: Target values.
            cv: Number of CV folds.
            scoring: Scoring metric.
        
        Returns:
            Dictionary with CV results.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        model = self._create_model()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative RMSE to positive
        if 'neg_' in scoring:
            scores = -scores
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'cv_folds': cv
        }
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'regularization': self.regularization,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'standardize': self.standardize
        }


def train_linear_baseline(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'RUL',
    regularization: str = 'ridge',
    alpha: float = 1.0
) -> Tuple[LinearRULPredictor, Dict]:
    """
    Convenience function to train a linear baseline model.
    
    Args:
        train_df: Training DataFrame.
        feature_cols: List of feature columns (default: all sensors + settings).
        target_col: Target column name.
        regularization: Regularization type.
        alpha: Regularization strength.
    
    Returns:
        Tuple of (trained model, CV results).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    model = LinearRULPredictor(regularization=regularization, alpha=alpha)
    cv_results = model.cross_validate(X, y)
    model.fit(X, y)
    
    return model, cv_results


def print_linear_model_summary(
    model: LinearRULPredictor,
    cv_results: Optional[Dict] = None
) -> None:
    """
    Print summary of a trained linear model.
    
    Args:
        model: Trained LinearRULPredictor.
        cv_results: Optional cross-validation results.
    """
    print("=" * 60)
    print("LINEAR MODEL SUMMARY")
    print("=" * 60)
    
    params = model.get_params()
    print(f"\n📋 Configuration:")
    print(f"  • Regularization: {params['regularization']}")
    print(f"  • Alpha: {params['alpha']}")
    print(f"  • Standardized: {params['standardize']}")
    
    if cv_results:
        print(f"\n📊 Cross-Validation Results ({cv_results['cv_folds']}-fold):")
        print(f"  • RMSE: {cv_results['mean']:.2f} ± {cv_results['std']:.2f}")
    
    print("\n🔝 Top 10 Features by |Coefficient|:")
    importance = model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:20s}  coef={row['coefficient']:+.4f}")
    
    print("=" * 60)
