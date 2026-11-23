"""Production churn pipeline.

Author: Tariq Hussain
"""
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlmath.prep import Standardizer
from mlmath.logreg import LogisticRegressionGD
from mlmath.metrics import roc_auc_rank

FEATURES = ["tenure_months", "monthly_charge", "support_calls"]

df = pd.read_csv("data/churn.csv")
X = df[FEATURES]
y = df["churned"].to_numpy(int)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=11
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, tol=1e-8))
])

search = GridSearchCV(
    pipe,
    {"clf__C": [0.001, 0.01, 0.1, 1., 10., 100.]},
    cv=StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ),
    scoring="roc_auc",
    n_jobs=1
)

search.fit(Xtr, ytr)

print("C        Mean CV AUC     Std")

for C, m, s in zip(
    search.cv_results_["param_clf__C"],
    search.cv_results_["mean_test_score"],
    search.cv_results_["std_test_score"]
):
    print(f"{C:8g} {m:12.4f} {s:8.4f}")

best = search.best_estimator_
best_C = float(search.best_params_["clf__C"])

p_te = best.predict_proba(Xte)[:, 1]
test_auc = float(roc_auc_score(yte, p_te))

sc = Standardizer().fit(Xtr.to_numpy(float))

ours = LogisticRegressionGD(
    lr=1.0,
    epochs=20000,
    l2=1.0 / (len(ytr) * best_C)
)

ours.fit(sc.transform(Xtr.to_numpy(float)), ytr)

p_ours = ours.predict_proba(
    sc.transform(Xte.to_numpy(float))
)

max_prob_diff = float(np.max(np.abs(p_ours - p_te)))

results = {
    "best_C": best_C,
    "cv_auc": float(search.best_score_),
    "test_auc": test_auc,
    "auc_ours": float(roc_auc_rank(yte, p_ours)),
    "max_prob_diff": max_prob_diff
}

joblib.dump({
    "model": best,
    "features": FEATURES,
    "threshold": 0.5,
    "sklearn_version": sklearn.__version__
}, "models/churn_model.joblib")

with open("outputs/churn_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("Saved models/churn_model.joblib")
