from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator
import copy



def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    folds = np.array_split(indices, cv)

    train_scores = []
    validation_scores = []
    for val_idx in folds:
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # use a fresh copy of the estimator for each fold
        est = copy.deepcopy(estimator)
        est.fit(X_train, y_train)

        train_scores.append(est.loss(X_train, y_train))
        validation_scores.append(est.loss(X_val, y_val))

    return float(np.mean(train_scores)), float(np.mean(validation_scores))