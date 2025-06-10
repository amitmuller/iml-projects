import os
import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

OUTPUT_DIR = "/Users/amitmuller/Git projects/iml projects/graphs/ex3_iml"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    """
    X = np.random.rand(n, 2) * 2 - 1
    y = np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    flip_indices = np.random.choice(n, int(noise_ratio * n), replace=False)
    y[flip_indices] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    """
    Train AdaBoost on one noisy dataset and evaluate misclassification error
    as a function of the number of weak learners, using the same train/test split.
    """
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    train_error = []
    for t in range(1, n_learners + 1):
        train_error.append(model.partial_loss(train_X, train_y, t))

    test_error = []
    for t in range(1, n_learners + 1):
        test_error.append(model.partial_loss(test_X, test_y, t))

    fig1 = go.Figure(
        data=[
            go.Scatter(x=list(range(1, n_learners + 1)), y=train_error, name="train err", mode="lines"),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_error, name="test err", mode="lines")
        ],
        layout=go.Layout(
            width=600, height=400,
            title={"x": 0.5, "text": r"$\text{adaboost misclassification as function of number of learners}$"},
            xaxis_title=r"$\text{num of learners}$",
            yaxis_title=r"$\text{misclassification error}$",
            margin=dict(l=40, r=40, t=60, b=40)
        )
    )
    fig1.write_image(os.path.join(OUTPUT_DIR, f"adaboost_____{noise}.png"))

    # Question 2: decision surfaces
    T = [5, 50, 100, 250]
    subplot_titles = []
    for t in T:
        subplot_titles.append(rf"$\text{{{t} classifiers}}$")

    lims = (
        np.array([np.r_[train_X, test_X].min(axis=0),
                  np.r_[train_X, test_X].max(axis=0)])
        .T + np.array([-.1, .1])
    )
    fig2 = make_subplots(rows=1, cols=4, subplot_titles=subplot_titles)
    for i, t in enumerate(T):
        fig2.add_traces(
            [
                decision_surface(
                    lambda X: model.partial_predict(X, t),
                    lims[0], lims[1],
                    density=60, showscale=False
                ),
                go.Scatter(
                    x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"))
                )
            ],
            rows=1, cols=i + 1
        )
    fig2.update_layout(width=1200, height=300, margin=dict(l=20, r=20, t=40, b=20)) \
         .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.write_image(os.path.join(OUTPUT_DIR, f"adaboost_{noise}_decision_boundaries.png"))

    # Question 3: best-performing ensemble
    best_t = np.argmin(test_error) + 1
    fig3 = go.Figure(
        [
            decision_surface(
                lambda X: model.partial_predict(X, best_t),
                lims[0], lims[1],
                density=60, showscale=False
            ),
            go.Scatter(
                x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"))
            )
        ],
        layout=go.Layout(
            width=400, height=400,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title=(
                f"best performing ensemble<br>"
                f"T: {best_t}, accuracy: {1 - round(test_error[best_t - 1], 2)}"
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    )
    fig3.write_image(os.path.join(OUTPUT_DIR, f"adaboost_{noise}_best_over_test.png"))

    # Question 4: weighted samples
    D = 20 * model.D_ / model.D_.max()
    fig4 = go.Figure(
        [
            decision_surface(model.predict, lims[0], lims[1], density=60, showscale=False),
            go.Scatter(
                x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                marker=dict(size=D, color=train_y, symbol=np.where(train_y == 1, "circle", "x"))
            )
        ],
        layout=go.Layout(
            width=400, height=600,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title="adaboost sample distribution",
            margin=dict(l=20, r=20, t=40, b=20)
        )
    )
    fig4.write_image(os.path.join(OUTPUT_DIR, f"adaboost_{noise}_weighted_samples.png"))


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
