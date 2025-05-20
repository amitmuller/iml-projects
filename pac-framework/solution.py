import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


sns.set(style="whitegrid")


#######     helper functions    ########
def _generate_data(m: int, mean=None, cov=None, seed: int = 0):
    """
    draw m samples from N(mean, cov) and label them
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(2) if mean is None else mean
    cov = np.array([[1, 0.5], [0.5, 1]]) if cov is None else cov
    X = rng.multivariate_normal(mean, cov, size=m)
    w = np.array([-0.6, 0.4])
    y = np.sign(X @ w)
    return X, y.astype(int), w


def _plot_svm(ax, X, y, sep_true, svm_clf, m, C):
    """
    the true separator and the SVM decision boundary learned.
    """
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color="blue")
    ax.scatter(X[y == 1, 0],  X[y == 1, 1],  color="red")

    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, 200)

    ax.plot(x, -(sep_true[0] / sep_true[1]) * x,
            linestyle="--", linewidth=2, color="black", label="true seperator")

    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    ax.plot(x, -(w[0] * x + b) / w[1],
            linestyle="-", linewidth=2, color="green", label="SVM boundary")

    ax.set_title(f"SVM (m={m}, C={C})")

def _meshgrid(X, h=0.02, pad=1.0):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx_ret, yy_ret = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx_ret, yy_ret


def _plot_decision(ax, clf, X_train, y_train, xx, yy, title):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")

    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               color="blue", edgecolor="k")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               color="red", edgecolor="k")

    ax.set_title(title)




########  Exercise Solution   ###########
def pratical_1_runner(save_path=None):
    """
    generates the plots
    """
    ms = [5, 10, 20, 100]
    Cs = [0.1, 1, 5, 10, 100]

    for m in ms:
        X, y, w_true = _generate_data(m)
        for C in Cs:
            clf = SVC(kernel="linear", C=C)
            clf.fit(X, y)

            fig, ax = plt.subplots(figsize=(5, 4))
            _plot_svm(ax, X, y, w_true, clf, m, C)
            plt.tight_layout()

            if save_path is None:
                plt.show()
            else:
                os.makedirs(save_path, exist_ok=True)
                file_name = f"svm_m{m}_C{C}.png"
                plt.savefig(os.path.join(save_path, file_name), dpi=150)
                plt.close(fig)


def practical_2_runner(save_path=None):
    """
    Generates boundary-plots
    """
    rng = np.random.default_rng(0)
    datasets = {
        "moons": make_moons(n_samples=200, noise=0.2, random_state=0),
        "circles": make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=0),
        "gaussians": (
            np.vstack((
                rng.multivariate_normal(mean=[-1, -1],
                                        cov=[[0.5, 0.2], [0.2, 0.5]], size=100),
                rng.multivariate_normal(mean=[1, 1],
                                        cov=[[0.5, 0.2], [0.2, 0.5]], size=100)
            )),
            np.hstack((np.zeros(100, dtype=int), np.ones(100, dtype=int)))
        )
    }

    classifiers = {
        "SVM λ=5": SVC(kernel="linear", C=1/5, gamma="scale"),
        "Tree depth 7": DecisionTreeClassifier(max_depth=7, random_state=0),
        "KNN k=5": KNeighborsClassifier(n_neighbors=5)
    }

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for d_name, (X, y) in datasets.items():
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        xx, yy = _meshgrid(X_train)

        for c_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            accuracy = accuracy_score(y_train, clf.predict(X_train))
            fig, ax = plt.subplots(figsize=(5, 4))
            _plot_decision(ax, clf, X_train, y_train, xx, yy,
                           title=f"{d_name}" + "-" + f"{c_name}" +f" accuracy: {accuracy}")
            plt.tight_layout()

            if save_path is None:
                plt.show()
            else:
                file_name = f"{d_name}-{c_name.replace(' ', '_')}.png"
                plt.savefig(os.path.join(save_path, file_name), dpi=150)
                plt.close(fig)



if __name__ == "__main__":
    path = "/Users/amitmuller/Git projects/iml projects/graphs/ex2 iml"
    pratical_1_runner(save_path=path)
    practical_2_runner(save_path=path) 