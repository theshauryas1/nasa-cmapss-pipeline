"""
Ensemble Machine Learning Models for RUL Prediction

This module provides ensemble methods for Remaining Useful Life (RUL)
prediction, with emphasis on Random Forest as the primary model.
These models generally offer better predictive performance than linear
baselines while still providing feature importance insights.

Models included:
- Random Forest Regressor
- Gradient Boosting Regressor (optional)

Author: Scientific Data Pipeline Project
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from ..data.ingestion import get_feature_columns


class RandomForestRUL:
    """
    Random Forest model for RUL prediction.
    
    Random Forest is well-suited for RUL prediction because:
    - Handles non-linear relationships
    - Robust to outliers and noise
    - Provides feature importance rankings
    - Built-in out-of-bag error estimation
    
    Example:
        >>> model = RandomForestRUL(n_estimators=100, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples to split an internal node.
            min_samples_leaf: Minimum samples in a leaf node.
            max_features: Number of features for best split ('sqrt', 'log2', or int).
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def _create_model(self) -> RandomForestRegressor:
        """Create the Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True  # Enable out-of-bag score
        )
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'RandomForestRUL':
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
        
        predictions = self.model_.predict(X)
        
        # Clip to non-negative values
        return np.clip(predictions, 0, None)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the Random Forest.
        
        Uses the built-in impurity-based feature importance.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.model_.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df.reset_index(drop=True)
    
    def get_oob_score(self) -> float:
        """
        Get the out-of-bag R² score.
        
        OOB score provides an unbiased estimate of test error.
        
        Returns:
            OOB R² score.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        return self.model_.oob_score_
    
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
        
        model = self._create_model()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative scores to positive
        if 'neg_' in scoring:
            scores = -scores
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'cv_folds': cv
        }
    
    def hyperparameter_search(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Feature matrix.
            y: Target values.
            param_grid: Dictionary of parameters to search.
            cv: Number of CV folds.
        
        Returns:
            Dictionary with best parameters and scores.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=1
        )
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert to positive RMSE
            'cv_results': grid_search.cv_results_
        }
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state
        }


class GradientBoostingRUL:
    """
    Gradient Boosting model for RUL prediction.
    
    Gradient Boosting often achieves better accuracy than Random Forest
    but may be more prone to overfitting. Use for comparison.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Shrinkage factor.
            max_depth: Maximum depth of individual trees.
            min_samples_split: Minimum samples to split.
            min_samples_leaf: Minimum samples in leaf.
            random_state: Random seed.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'GradientBoostingRUL':
        """Fit the model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict RUL values."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return np.clip(self.model_.predict(X), 0, None)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.model_.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)


def train_random_forest(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'RUL',
    **kwargs
) -> Tuple[RandomForestRUL, Dict]:
    """
    Convenience function to train a Random Forest model.
    
    Args:
        train_df: Training DataFrame.
        feature_cols: List of feature columns.
        target_col: Target column name.
        **kwargs: Additional arguments for RandomForestRUL.
    
    Returns:
        Tuple of (trained model, CV results).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    model = RandomForestRUL(**kwargs)
    cv_results = model.cross_validate(X, y)
    model.fit(X, y)
    
    return model, cv_results


def print_rf_model_summary(
    model: RandomForestRUL,
    cv_results: Optional[Dict] = None
) -> None:
    """
    Print summary of a trained Random Forest model.
    
    Args:
        model: Trained RandomForestRUL.
        cv_results: Optional cross-validation results.
    """
    print("=" * 60)
    print("RANDOM FOREST MODEL SUMMARY")
    print("=" * 60)
    
    params = model.get_params()
    print(f"\n📋 Configuration:")
    print(f"  • Trees: {params['n_estimators']}")
    print(f"  • Max Depth: {params['max_depth']}")
    print(f"  • Min Samples Split: {params['min_samples_split']}")
    print(f"  • Max Features: {params['max_features']}")
    
    print(f"\n📊 Performance:")
    print(f"  • OOB R² Score: {model.get_oob_score():.4f}")
    
    if cv_results:
        print(f"  • CV RMSE ({cv_results['cv_folds']}-fold): "
              f"{cv_results['mean']:.2f} ± {cv_results['std']:.2f}")
    
    print("\n🔝 Top 10 Features by Importance:")
    importance = model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:20s}  "
              f"importance={row['importance']:.4f}  "
              f"(cumulative: {row['cumulative_importance']:.2%})")
    
    # Features needed for 90% importance
    n_for_90 = (importance['cumulative_importance'] < 0.9).sum() + 1
    print(f"\n  → {n_for_90} features capture 90% of total importance")
    
    print("=" * 60)
