"""Newton/IRLS challenge experiment.

Author: Tariq Hussain
"""
import io
import json
import re
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlmath.linreg import add_bias, train_test_split_idx
from mlmath.prep import Standardizer
from mlmath.logreg import LogisticRegressionGD, loss_and_grad
from mlmath.newton import fit_newton

df = pd.read_csv("data/churn.csv")

features = [
    "tenure_months",
    "monthly_charge",
    "support_calls"
]

X = df[features].to_numpy(float)
y = df["churned"].to_numpy(int)

tr, te = train_test_split_idx(len(y), seed=11)

sc = Standardizer().fit(X[tr])
Z = sc.transform(X[tr])
Xb = add_bias(Z)
yt = y[tr]

print("Training samples:", len(tr))
print("Test samples:", len(te))

print("\n========== NEWTON ITERATIONS ==========")

buffer = io.StringIO()

with redirect_stdout(buffer):
    w, n_iter = fit_newton(Xb, yt)

trace = buffer.getvalue()
print(trace, end="")

norms = [
    float(v)
    for v in re.findall(r"\|\|grad\|\|=([0-9.eE+-]+)", trace)
]

_, grad = loss_and_grad(Xb, yt, w)
final_norm = float(np.linalg.norm(grad))

print("\n========== GRADIENT DESCENT COMPARISON ==========")

# Use a separate higher-precision comparison for the challenge.
gd = LogisticRegressionGD(
    lr=1.0,
    epochs=20000,
    l2=0.0,
    tol=0.0
).fit(Z, yt)

gd_diff = float(np.max(np.abs(w - gd.w_)))

print("GD iterations:", gd.n_iter_)
print("Newton iterations:", n_iter)
print("Maximum Newton/GD coefficient difference:", gd_diff)

print("\n========== SKLEARN COMPARISON ==========")

sk = LogisticRegression(
    penalty=None,
    max_iter=10000,
    tol=1e-12
).fit(Z, yt)

w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]

sk_diff = float(np.max(np.abs(w - w_sk)))

print("Maximum sklearn coefficient difference:", sk_diff)
print("Final Newton gradient norm:", final_norm)

print("\n========== QUADRATIC CONVERGENCE ==========")

ratios = []

for a, b in zip(norms[:-1], norms[1:]):
    if a > 1e-7 and b > 1e-14:
        ratio = b / (a * a)
        ratios.append(float(ratio))
        print(f"{a:.6e} -> {b:.6e}; ratio={ratio:.6e}")

quadratic = any(
    1e-7 < a < 1e-2 and b < a * a * 100
    for a, b in zip(norms[:-1], norms[1:])
)

results = {
    "author": "Tariq Hussain",
    "training_samples": int(len(tr)),
    "newton_iterations": int(n_iter),
    "gd_iterations": int(gd.n_iter_),
    "max_diff_vs_gd": gd_diff,
    "max_diff_vs_sklearn": sk_diff,
    "final_gradient_norm": final_norm,
    "gradient_norm_history": norms,
    "quadratic_ratios": ratios,
    "quadratic_convergence_observed": quadratic,
    "newton_coefficients": w.tolist(),
    "gd_coefficients": gd.w_.tolist()
}

with open("outputs/challenge_newton.json", "w") as f:
    json.dump(results, f, indent=2)

assert n_iter < 10
assert gd_diff < 1e-6
assert sk_diff < 1e-6
assert final_norm < 1e-12
assert quadratic

print("\nPASS: Newton converged in fewer than 10 iterations")
print("PASS: Newton matches gradient descent")
print("PASS: Newton matches sklearn")
print("PASS: Final gradient norm below 1e-12")
print("PASS: Numerical evidence of quadratic convergence")
print("PASS: CHALLENGE EXPERIMENT")
