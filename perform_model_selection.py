import numpy as np
import os
import plotly.graph_objects as go
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    output_dir = "/Users/amitmuller/Git projects/iml projects/graphs/ex3_iml"
    os.makedirs(output_dir, exist_ok=True)

    # Question 1 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:n_samples], indices[n_samples:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    ridge_lambdas = np.linspace(0.001, 5, n_evaluations)
    ridge_train_err, ridge_val_err = [], []
    for lam in ridge_lambdas:
        est = RidgeRegression(lam=lam, include_intercept=True)
        tr, va = cross_validate(est, X_train, y_train, cv=5)
        ridge_train_err.append(tr)
        ridge_val_err.append(va)

    # Lasso
    lasso_lambdas = np.linspace(0.001, 5, n_evaluations)
    lasso_train_err, lasso_val_err = [], []
    for lam in lasso_lambdas:
        est = Lasso(alpha=lam, include_intercept=True)
        tr, va = cross_validate(est, X_train, y_train, cv=5)
        lasso_train_err.append(tr)
        lasso_val_err.append(va)

    # --- plot Lasso
    fig_lasso = go.Figure()
    fig_lasso.add_trace(go.Scatter(
        x=lasso_lambdas, y=lasso_train_err, name="Train Error", mode="lines"
    ))
    fig_lasso.add_trace(go.Scatter(
        x=lasso_lambdas, y=lasso_val_err, name="Validation Error", mode="lines"
    ))
    fig_lasso.update_layout(
        width=600, height=400,
        title={"x": 0.5, "text": "lasso - train and validation error as function of λ"},
        xaxis_title="lambda",
        yaxis_title="mse"
    )
    fig_lasso.write_image(os.path.join(output_dir, "lasso_train_validation_vs_lambda.png"))

    # --- plot Ridge
    fig_ridge = go.Figure()
    fig_ridge.add_trace(go.Scatter(
        x=ridge_lambdas, y=ridge_train_err, name="Train Error", mode="lines"
    ))
    fig_ridge.add_trace(go.Scatter(
        x=ridge_lambdas, y=ridge_val_err, name="Validation Error", mode="lines"
    ))
    fig_ridge.update_layout(
        width=600, height=400,
        title={"x": 0.5, "text": "ridge: train and validation error as function of λ"},
        xaxis_title="lambda",
        yaxis_title="mse"
    )
    fig_ridge.write_image(os.path.join(output_dir, "ridge_train_validation_vs_lambda.png"))

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_idx = int(np.argmin(ridge_val_err))
    best_lasso_idx = int(np.argmin(lasso_val_err))
    best_ridge_lambda = ridge_lambdas[best_ridge_idx]
    best_lasso_lambda = lasso_lambdas[best_lasso_idx]

    # final models on all training data
    ridge_final = RidgeRegression(lam=best_ridge_lambda, include_intercept=True)
    ridge_final.fit(X_train, y_train)
    ridge_test_mse = ridge_final.loss(X_test, y_test)

    lasso_final = Lasso(alpha=best_lasso_lambda, include_intercept=True)
    lasso_final.fit(X_train, y_train)
    lasso_test_mse = lasso_final.loss(X_test, y_test)

    ls_final = LinearRegression()
    ls_final.fit(X_train, y_train)
    ls_test_mse = ls_final.loss(X_test, y_test)

    # print results
    print(f"Best Ridge λ: {best_ridge_lambda:.5f}")
    print(f"Best Lasso λ: {best_lasso_lambda:.5f}")
    print(f"Ridge test MSE: {ridge_test_mse:.2f}")
    print(f"Lasso test MSE: {lasso_test_mse:.2f}")
    print(f"Least Squares test MSE: {ls_test_mse:.2f}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
