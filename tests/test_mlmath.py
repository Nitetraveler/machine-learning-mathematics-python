"""Unit tests for the from-scratch mathematical implementations."""
import numpy as np
import pandas as pd
import pytest

from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix as sk_cm,
    f1_score,
    precision_score,
    r2_score as sk_r2,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from mlmath.gd import (
    batch_gradient_descent,
    mse_gradient,
    mse_loss,
    numerical_gradient,
    unscale_weights,
)
from mlmath.linreg import (
    add_bias,
    fit_lstsq,
    fit_normal_equation,
    r2_score,
)
from mlmath.logreg import LogisticRegressionGD, loss_and_grad, sigmoid
from mlmath.metrics import (
    auc_trapezoid,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc_rank,
    roc_curve,
)
from mlmath.model_selection import (
    kfold_indices,
    polynomial_features,
    ridge_fit,
)
from mlmath.pca import PCA, n_components_for_variance
from mlmath.prep import Standardizer, impute_median


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (
        1.5 + X @ np.array([2.0, -1.0, 0.5])
        + rng.normal(0, 0.1, 200)
    )
    return X, y


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    p = sigmoid(0.3 + X @ np.array([1.5, -2.0, 0.7]))
    return X, (rng.uniform(size=500) < p).astype(int)


def test_impute_median_returns_copy():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out, med = impute_median(df, ["a"])

    assert med["a"] == 2.0
    assert out["a"].isna().sum() == 0
    assert df["a"].isna().sum() == 1


def test_standardizer_matches_sklearn(reg_data):
    X, _ = reg_data

    np.testing.assert_allclose(
        Standardizer().fit_transform(X),
        StandardScaler().fit_transform(X),
        atol=1e-12
    )


def test_normal_equation_matches_sklearn(reg_data):
    X, y = reg_data

    w = fit_normal_equation(add_bias(X), y)
    sk = LinearRegression().fit(X, y)

    np.testing.assert_allclose(
        w, np.r_[sk.intercept_, sk.coef_], atol=1e-10
    )

    np.testing.assert_allclose(
        w, fit_lstsq(add_bias(X), y), atol=1e-10
    )


def test_r2_matches_sklearn(reg_data):
    X, y = reg_data
    yhat = add_bias(X) @ fit_normal_equation(add_bias(X), y)

    assert r2_score(y, yhat) == pytest.approx(
        sk_r2(y, yhat), abs=1e-12
    )


def test_mse_gradient_matches_numerical(reg_data):
    X, y = reg_data
    Xb = add_bias(X)
    w = np.array([0.1, -0.2, 0.3, 0.4])

    g = mse_gradient(Xb, y, w)
    gn = numerical_gradient(lambda v: mse_loss(Xb, y, v), w)

    assert np.linalg.norm(g - gn) / np.linalg.norm(g + gn) < 1e-6


def test_gradient_descent_reaches_ols(reg_data):
    X, y = reg_data

    sc = Standardizer().fit(X)
    Xb = add_bias(sc.transform(X))

    w, hist = batch_gradient_descent(
        Xb, y, lr=0.1, epochs=10000, tol=1e-14
    )

    np.testing.assert_allclose(
        unscale_weights(w, sc.mean_, sc.std_),
        fit_normal_equation(add_bias(X), y),
        atol=1e-5
    )

    assert hist[-1] < hist[0]


def test_sigmoid_is_stable():
    with np.errstate(over="raise", invalid="raise"):
        p = sigmoid(np.array([-1000., 0., 1000.]))

    np.testing.assert_allclose(p, [0., 0.5, 1.])


def test_logistic_gradient_and_sklearn(clf_data):
    X, y = clf_data
    Xb = add_bias(X)
    w = np.array([0.1, -0.2, 0.3, 0.4])

    loss, g = loss_and_grad(Xb, y, w, l2=0.05)

    gn = numerical_gradient(
        lambda v: loss_and_grad(Xb, y, v, l2=0.05)[0],
        w
    )

    assert np.linalg.norm(g - gn) / np.linalg.norm(g + gn) < 1e-6

    for l2 in [0., 0.05]:
        ours = LogisticRegressionGD(
            lr=1., epochs=20000, l2=l2
        ).fit(X, y)

        if l2 == 0:
            sk = LogisticRegression(
                penalty=None, max_iter=10000, tol=1e-8
            ).fit(X, y)
        else:
            sk = LogisticRegression(
                C=1.0 / (len(y) * l2),
                max_iter=10000, tol=1e-8
            ).fit(X, y)

        np.testing.assert_allclose(
            ours.w_,
            np.r_[sk.intercept_, sk.coef_.ravel()],
            atol=1e-3
        )


def test_classification_metrics_match_sklearn():
    y = np.array([0, 0, 1, 1, 1])
    yp = np.array([0, 1, 1, 1, 0])

    np.testing.assert_array_equal(
        confusion_matrix(y, yp), sk_cm(y, yp)
    )

    assert precision(y, yp) == pytest.approx(precision_score(y, yp))
    assert recall(y, yp) == pytest.approx(recall_score(y, yp))
    assert f1(y, yp) == pytest.approx(f1_score(y, yp))
    assert precision(np.array([0, 1]), np.array([0, 0])) == 0


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)

    y = rng.integers(0, 2, 200)
    p = np.round(rng.uniform(size=200), 1)

    expected = roc_auc_score(y, p)

    fpr, tpr, _ = roc_curve(y, p)

    assert roc_auc_rank(y, p) == pytest.approx(expected, abs=1e-10)
    assert auc_trapezoid(fpr, tpr) == pytest.approx(expected, abs=1e-10)


def test_kfold_partitions_indices():
    folds = list(kfold_indices(23, k=5, seed=0))

    assert len(folds) == 5

    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == 23

    np.testing.assert_array_equal(
        np.sort(np.concatenate([va for _, va in folds])),
        np.arange(23)
    )


def test_ridge_matches_sklearn(reg_data):
    X, y = reg_data

    for lam in [0.1, 0.5, 10.0]:
        w = ridge_fit(add_bias(X), y, lam)
        sk = Ridge(alpha=lam).fit(X, y)

        np.testing.assert_allclose(
            w, np.r_[sk.intercept_, sk.coef_], atol=1e-10
        )

    np.testing.assert_allclose(
        polynomial_features([-2., 0., 2.], 3),
        [[-2., 4., -8.], [0., 0., 0.], [2., 4., 8.]]
    )


def test_pca_matches_sklearn():
    rng = np.random.default_rng(5)

    X = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6))

    ours, sk = PCA(3).fit(X), SkPCA(3).fit(X)

    np.testing.assert_allclose(
        ours.explained_variance_ratio_,
        sk.explained_variance_ratio_,
        atol=1e-10
    )

    np.testing.assert_allclose(
        np.abs(np.sum(ours.components_ * sk.components_, axis=1)),
        1.0,
        atol=1e-8
    )

    assert n_components_for_variance(
        np.array([0.5, 0.3, 0.15, 0.05]), 0.9
    ) == 3


def test_pca_reconstruction_identity():
    rng = np.random.default_rng(6)

    X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))

    n, d, k = X.shape[0], X.shape[1], 2
    full, pk = PCA().fit(X), PCA(k).fit(X)

    rec = np.mean(
        (X - pk.inverse_transform(pk.transform(X))) ** 2
    )

    expected = (
        (n - 1) / (n * d)
        * full.explained_variance_all_[k:].sum()
    )

    assert rec == pytest.approx(expected, rel=1e-8)


def test_newton_matches_sklearn(clf_data):
    from mlmath.newton import fit_newton

    X, y = clf_data
    Xb = add_bias(X)

    w, n_iter = fit_newton(Xb, y)

    sk = LogisticRegression(
        penalty=None,
        max_iter=10000,
        tol=1e-12
    ).fit(X, y)

    np.testing.assert_allclose(
        w,
        np.r_[sk.intercept_, sk.coef_.ravel()],
        atol=1e-6,
        rtol=0
    )

    _, grad = loss_and_grad(Xb, y, w)

    assert n_iter < 10
    assert np.linalg.norm(grad) < 1e-12
