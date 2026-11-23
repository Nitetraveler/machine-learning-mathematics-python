"""Final validation of all lab tasks."""
import json
import subprocess
import sys
from pathlib import Path

results = []


def check(name, fn):
    try:
        ok = bool(fn())
    except Exception as exc:
        ok, name = False, f"{name} ({type(exc).__name__}: {exc})"
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


def j(path):
    return json.loads(Path(path).read_text())


def png(path):
    p = Path(path)
    return p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def setup():
    import numpy, sklearn
    return numpy.__version__ == "1.26.4" and sklearn.__version__ == "1.5.1"


def t1():
    import pandas as pd
    d = j("outputs/task1.json")
    return (pd.read_csv("data/housing.csv")["age_years"].isna().sum() == 12
            and pd.read_csv("data/housing_clean.csv").isna().sum().sum() == 0
            and d["rank"] == 5 and d["cond_raw"] > 100 and d["cond_std"] < 1.5)


def t2():
    d = j("outputs/task2_ols.json")
    return (d["max_diff_sklearn"] < 1e-6 and d["test_r2"] > 0.97 and 13 < d["test_rmse"] < 17
            and d["collinear_rank"] == 5
            and abs(d["collinear_size_sum"] - d["coef_normal_eq"][1]) < 1e-6)


def t3():
    d = j("outputs/task3_gd.json")
    return (d["grad_rel_error"] < 1e-6 and d["raw_diverged"] and d["below_bound_converged"]
            and d["above_bound_diverged"] and d["max_diff_vs_ols"] < 1e-4 and png("outputs/gd_lr_sweep.png"))


def t4():
    d = j("outputs/task4_logreg.json")
    return (d["max_diff_sklearn_noreg"] < 1e-3 and d["max_diff_sklearn_l2"] < 1e-3
            and 0.65 < d["test_accuracy"] < 0.90 and Path("outputs/churn_test_scores.csv").exists())


def t5():
    d = j("outputs/task5_metrics.json")
    return (d["cm_equal"] and all(abs(d["ours"][k] - d["sklearn"][k]) < 1e-12 for k in d["ours"])
            and abs(d["auc_rank"] - d["auc_sklearn"]) < 1e-10
            and abs(d["auc_trapezoid"] - d["auc_sklearn"]) < 1e-10
            and d["recall80_recall"] >= 0.8 and d["paradox_recall"] == 0.0 and png("outputs/roc_churn.png"))


def t6():
    d = j("outputs/task6_ridge.json")
    r, b = d["degree_results"], str(d["best_degree_cv"])
    return (d["max_diff_sklearn"] < 1e-6 and d["norm_monotone"] and d["best_degree_cv"] in (3, 5, 9)
            and r["15"]["cv_mse"] > r[b]["cv_mse"] and d["test_mse_best"] < 0.15
            and png("outputs/bias_variance.png"))


def t7():
    d = j("outputs/task7_pca.json")
    return (d["n90_ours"] == d["n90_sklearn"] and d["max_ratio_diff"] < 1e-8
            and d["min_abs_cos_top10"] > 0.9999 and d["orthonormal_err"] < 1e-10
            and d["recon_identity_err"] < 1e-8 and png("outputs/pca_digits.png"))


def t8():
    import joblib
    import pandas as pd
    b = joblib.load("models/churn_model.joblib")
    d = j("outputs/churn_metrics.json")
    s = pd.read_csv("outputs/scored.csv").set_index("customer_id")
    return ("features" in b and d["test_auc"] > 0.75 and d["max_prob_diff"] < 1e-3
            and s.loc["C001", "flag_for_retention"] == 1 and s.loc["C002", "flag_for_retention"] == 0)


def tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
    return r.returncode == 0


check("Setup: pinned numpy 1.26.4 / scikit-learn 1.5.1", setup)
check("Task 1: data cleaned, rank and conditioning diagnosed", t1)
check("Task 2: OLS normal equation, orthogonality, collinearity", t2)
check("Task 3: gradient check, divergence, learning-rate bound", t3)
check("Task 4: logistic regression matches sklearn (with and without L2)", t4)
check("Task 5: metrics/AUC match sklearn, threshold policy, paradox", t5)
check("Task 6: ridge matches sklearn, CV selects sensible model", t6)
check("Task 7: PCA matches sklearn and reconstruction theory", t7)
check("Task 8: tuned pipeline persisted and CLI scoring correct", t8)
check("Task 8: pytest suite passes", tests)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
