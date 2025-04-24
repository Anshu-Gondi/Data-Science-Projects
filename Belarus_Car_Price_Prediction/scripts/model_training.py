import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from joblib import parallel_backend


def train_decision_tree(X_train, y_train):
    """
    Trains a DecisionTreeRegressor with hyperparameter tuning using RandomizedSearchCV
    and returns the trained model.
    """
    dtr = DecisionTreeRegressor()

    # Define a smaller hyperparameter space for faster search
    params = {
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt'],
        'random_state': [42]
    }

    # Use RandomizedSearchCV (faster than GridSearchCV)
    with parallel_backend('loky'):
        grid = RandomizedSearchCV(dtr, param_distributions=params,
                                  cv=3, n_iter=10, verbose=1, n_jobs=-1, random_state=42)
        grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("✅ Best Parameters:", best_params)

    # Retrain model with best parameters
    dtr = DecisionTreeRegressor(**best_params)
    dtr.fit(X_train, y_train)

    return dtr
