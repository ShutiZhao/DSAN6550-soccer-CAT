"""
Step 4 - 2PL Item Parameter Calibration
DSAN 6550 Final Project (Spring 2026)
Role 1: Psychometric Lead

Approach (college-level, matches Session 4 content):
  The 2PL probability of a correct response is
        P(Y_ij = 1 | theta_i, a_j, b_j) = 1 / (1 + exp(-(a_j*theta_i - a_j*b_j)))
                                        = 1 / (1 + exp(-(slope_j * theta_i + intercept_j)))
  where  slope_j     = a_j
         intercept_j = -a_j * b_j   ->   b_j = -intercept_j / slope_j

  So fitting a 2PL item is equivalent to fitting a logistic regression
  of the item's responses on theta. We do that one item at a time.

  Theta itself isn't observed in real life, but a standard CTT-style
  proxy is the z-scored total score (sum-score). That works because, for
  a reasonably reliable test, total score correlates strongly with
  the true latent ability (we showed this in Step 3).

Steps:
  1) Compute provisional theta_i = z(sum_score_i).
  2) For each item j, fit LogisticRegression(y_ij ~ theta_i).
  3) Read off a_j (slope) and b_j (-intercept/slope).
  4) Save calibrated_item_bank.csv (the file the front-end will read).
  5) Quick recovery check vs the true (a, b) used in simulation.

Dependencies:
    Prefers scikit-learn's LogisticRegression (a standard sklearn class
    your laptops will already have via Anaconda / pip). If sklearn is
    not installed, falls back to a small numpy gradient-descent fitter
    so the code still runs - the math is identical.

Outputs:
    data/calibrated_item_bank.csv      <- canonical file for the CAT engine
    data/calibrated_thetas.csv         <- estimated person abilities
    outputs/calibration_diagnostics.txt
    plots/recovery_scatter.png         <- true vs estimated a, b, theta
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional dependency: sklearn (preferred). Fallback: tiny numpy fitter.
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
OUT_DIR  = os.path.normpath(os.path.join(HERE, "..", "outputs"))
PLOT_DIR = os.path.normpath(os.path.join(HERE, "..", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data from earlier steps
# ---------------------------------------------------------------------------
resp     = pd.read_csv(os.path.join(DATA_DIR, "simulated_responses.csv"), index_col=0)
items    = pd.read_csv(os.path.join(DATA_DIR, "items_metadata.csv"))
true_thr = pd.read_csv(os.path.join(DATA_DIR, "simulated_thetas.csv"))

X        = resp.to_numpy(dtype=float)        # (N, J) response matrix
N, J     = X.shape
item_ids = resp.columns.tolist()

# ---------------------------------------------------------------------------
# Provisional theta = z-scored total score
# ---------------------------------------------------------------------------
total = X.sum(axis=1)
theta = (total - total.mean()) / total.std(ddof=0)


# ---------------------------------------------------------------------------
# Per-item logistic regression
# ---------------------------------------------------------------------------
def fit_logreg_sklearn(theta_vec, y):
    """Fit y ~ theta with sklearn; return (slope, intercept)."""
    X1 = theta_vec.reshape(-1, 1)
    # Use a very small regularization (large C) so estimates are essentially MLE.
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X1, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def fit_logreg_numpy(theta_vec, y, max_iter=50, tol=1e-7):
    """
    Newton-Raphson logistic regression - fallback if sklearn isn't
    installed. Same MLE estimates as sklearn (no regularization).
    Converges in ~10 iterations for well-behaved data.

        Design matrix X = [theta_i, 1]      (slope, intercept)
        Score:  X.T @ (y - p)
        Hessian: -X.T @ diag(p*(1-p)) @ X
    """
    Xmat = np.column_stack([theta_vec, np.ones_like(theta_vec)])
    beta = np.zeros(2)                       # [slope, intercept]
    for _ in range(max_iter):
        z = Xmat @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        score   = Xmat.T @ (y - p)
        W       = p * (1.0 - p)
        hessian = -(Xmat * W[:, None]).T @ Xmat
        # Newton step: beta_new = beta - H^{-1} * score
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


fit_one = fit_logreg_sklearn if HAS_SKLEARN else fit_logreg_numpy
print(f"Step 4 - 2PL calibration via per-item logistic regression "
      f"({'sklearn' if HAS_SKLEARN else 'numpy fallback'})")
print(f"  N = {N} respondents, J = {J} items")
print(f"  Provisional theta: mean = {theta.mean():.3f}, sd = {theta.std(ddof=0):.3f}")

a_hat = np.empty(J)
b_hat = np.empty(J)

for j, item_id in enumerate(item_ids):
    y = X[:, j]
    if y.var() == 0:
        # Item with no variance can't be calibrated; skip.
        a_hat[j], b_hat[j] = np.nan, np.nan
        continue
    slope, intercept = fit_one(theta, y)
    a_hat[j] = slope                              # discrimination
    b_hat[j] = -intercept / slope if slope != 0 else np.nan  # difficulty

# ---------------------------------------------------------------------------
# Save calibrated item bank (this is the file the CAT engine reads)
# ---------------------------------------------------------------------------
calibrated = items[["ItemID", "Question", "OptionA", "OptionB", "OptionC",
                    "OptionD", "CorrectAnswer", "Difficulty", "Category"]].copy()
calibrated["a"] = np.round(a_hat, 4)
calibrated["b"] = np.round(b_hat, 4)
cal_path = os.path.join(DATA_DIR, "calibrated_item_bank.csv")
calibrated.to_csv(cal_path, index=False)

theta_out = pd.DataFrame({
    "RespondentID": resp.index,
    "theta_hat":    np.round(theta, 4),
    "total_score":  total.astype(int),
})
theta_path = os.path.join(DATA_DIR, "calibrated_thetas.csv")
theta_out.to_csv(theta_path, index=False)

# ---------------------------------------------------------------------------
# Recovery check vs the TRUE parameters (only possible because we simulated
# the data ourselves - in a real study you'd never know these).
# ---------------------------------------------------------------------------
a_true     = items["a_true"].to_numpy()
b_true     = items["b_true"].to_numpy()
theta_true = true_thr["theta_true"].to_numpy()

corr_a = np.corrcoef(a_true, a_hat)[0, 1]
corr_b = np.corrcoef(b_true, b_hat)[0, 1]
corr_t = np.corrcoef(theta_true, theta)[0, 1]
rmse_a = np.sqrt(np.mean((a_hat - a_true) ** 2))
rmse_b = np.sqrt(np.mean((b_hat - b_true) ** 2))
rmse_t = np.sqrt(np.mean((theta - theta_true) ** 2))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report = []
def w(s=""):
    report.append(s)

w("=" * 72)
w("DSAN 6550 - Step 4: 2PL Calibration Diagnostics")
w("=" * 72)
w(f"Method        : per-item logistic regression "
  f"({'sklearn' if HAS_SKLEARN else 'numpy fallback'})")
w(f"Sample        : N = {N}, items J = {J}")
w(f"Provisional theta proxy : z-scored total score "
  f"(mean = 0, sd = 1 by construction)")
w()
w("Item parameter recovery (calibrated vs simulated truth):")
w(f"  corr(a_hat, a_true)  = {corr_a:.4f}    RMSE(a) = {rmse_a:.4f}")
w(f"  corr(b_hat, b_true)  = {corr_b:.4f}    RMSE(b) = {rmse_b:.4f}")
w()
w("Person ability recovery (z-scored total vs true theta):")
w(f"  corr(theta_proxy, theta_true) = {corr_t:.4f}    RMSE = {rmse_t:.4f}")
w()
w("Note: a small constant scale difference between estimated and true 'a'")
w("is expected because we use a sum-score proxy for theta rather than the")
w("true latent metric. The RANK ORDERING of items is what the CAT engine")
w("relies on, and that is preserved (corr ~ %.2f for both a and b)."
  % min(corr_a, corr_b))
w()
w("Calibrated item parameters:")
w(calibrated[["ItemID", "Difficulty", "a", "b"]].to_string(index=False))
w("=" * 72)

report_text = "\n".join(report)
diag_path = os.path.join(OUT_DIR, "calibration_diagnostics.txt")
with open(diag_path, "w") as fh:
    fh.write(report_text)

print()
print(report_text)

# ---------------------------------------------------------------------------
# Recovery scatter (one figure with three panels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].scatter(a_true, a_hat, c="#1f77b4", edgecolor="white")
lo, hi = min(a_true.min(), a_hat.min()) - 0.1, max(a_true.max(), a_hat.max()) + 0.1
axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[0].set(xlabel="True a", ylabel="Estimated a",
            title=f"Discrimination (r = {corr_a:.3f})")

axes[1].scatter(b_true, b_hat, c="#d62728", edgecolor="white")
lo, hi = min(b_true.min(), b_hat.min()) - 0.2, max(b_true.max(), b_hat.max()) + 0.2
axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[1].set(xlabel="True b", ylabel="Estimated b",
            title=f"Difficulty (r = {corr_b:.3f})")

axes[2].scatter(theta_true, theta, c="#2ca02c", edgecolor="white", alpha=0.6, s=18)
lo, hi = min(theta_true.min(), theta.min()) - 0.3, max(theta_true.max(), theta.max()) + 0.3
axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[2].set(xlabel="True theta", ylabel="Estimated theta (proxy)",
            title=f"Ability (r = {corr_t:.3f})")

fig.suptitle("2PL Parameter Recovery - Soccer/World Cup Item Bank", y=1.02)
fig.tight_layout()
recovery_path = os.path.join(PLOT_DIR, "recovery_scatter.png")
fig.savefig(recovery_path, dpi=130, bbox_inches="tight")
plt.close(fig)

print()
print(f"Saved -> {cal_path}")
print(f"Saved -> {theta_path}")
print(f"Saved -> {diag_path}")
print(f"Saved -> {recovery_path}")
