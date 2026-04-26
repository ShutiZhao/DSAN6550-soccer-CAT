"""
Step 4b - OPTIONAL: 2PL Marginal Maximum Likelihood (MML) via Bock-Aitkin EM
DSAN 6550 Final Project (Spring 2026)
Role 1: Psychometric Lead

============================================================================
THIS SCRIPT IS OPTIONAL. The primary calibration is step4_calibration_2pl.py
(per-item logistic regression at the level of Session 4 / Hambleton et al).
Only run this file if you want to discuss MML/EM in the report or compare.
============================================================================

Why include MML at all?
  The simple per-item logistic regression in step4 uses a sum-score proxy
  for theta, which works well in practice but is an approximation.
  Marginal MLE integrates theta out under a N(0, 1) prior, which is the
  professional-grade calibration method (e.g., the R 'mirt' package and
  Python 'girth' package both use it). Once items are calibrated this
  way, individual thetas are recovered with EAP (expected a posteriori).

Algorithm (Bock & Aitkin 1981):
  1) Discretize theta on a Gauss-Hermite-like grid X_k with weights w_k.
  2) E-step: for each respondent compute posterior weights p(theta_k | y_i).
     Aggregate to expected counts r_jk (correct) and n_jk (total) at
     each quadrature point per item.
  3) M-step: for each item, fit 2PL by Newton on the weighted log-lik
     using (X_k, r_jk, n_jk).
  4) Repeat until log-likelihood converges.

Numpy-only implementation (works without scipy).

Outputs (../data/, ../outputs/):
    calibrated_item_bank_mml.csv     - MML item parameters
    calibrated_thetas_mml.csv        - EAP person abilities
    calibration_diagnostics_mml.txt  - convergence + recovery report
    plots/recovery_scatter_mml.png   - true vs estimated scatter

The driver script (run_pipeline.py) selects the better-fitting
calibration as the final 'calibrated_item_bank.csv'.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
OUT_DIR  = os.path.normpath(os.path.join(HERE, "..", "outputs"))
PLOT_DIR = os.path.normpath(os.path.join(HERE, "..", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
resp     = pd.read_csv(os.path.join(DATA_DIR, "simulated_responses.csv"), index_col=0)
items    = pd.read_csv(os.path.join(DATA_DIR, "items_metadata.csv"))
true_thr = pd.read_csv(os.path.join(DATA_DIR, "simulated_thetas.csv"))

X        = resp.to_numpy(dtype=float)         # (N, J)
N, J     = X.shape
item_ids = resp.columns.tolist()

# ---------------------------------------------------------------------------
# Quadrature: 41 equally spaced points across [-4, 4] with N(0,1) weights
# (rectangular quadrature is fine here; classic Bock-Aitkin choice).
# ---------------------------------------------------------------------------
K = 41
nodes  = np.linspace(-4.0, 4.0, K)
prior  = np.exp(-0.5 * nodes ** 2) / np.sqrt(2 * np.pi)
weights = prior / prior.sum()              # normalized prior weights

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    ez        = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
a = np.ones(J)
b = np.zeros(J)

def item_p_at_nodes(a_vec, b_vec):
    """Returns P_jk = P(correct | theta_k) for every (j, k).  Shape (J, K)."""
    Z = a_vec[:, None] * (nodes[None, :] - b_vec[:, None])
    return sigmoid(Z)

def marginal_loglik(X, a_vec, b_vec):
    P_jk = item_p_at_nodes(a_vec, b_vec)                # (J, K)
    # log P(y_i | theta_k) = sum_j y_ij log P_jk + (1 - y_ij) log(1 - P_jk)
    logP   = np.log(P_jk + 1e-12)
    log1mP = np.log(1.0 - P_jk + 1e-12)
    # (N, K) matrix of conditional log-likelihoods
    cond_ll = X @ logP + (1 - X) @ log1mP                # (N, K)
    log_w   = np.log(weights + 1e-300)
    # logsumexp over k
    M = (cond_ll + log_w[None, :]).max(axis=1, keepdims=True)
    ll_per_person = M.squeeze(1) + np.log(np.exp(cond_ll + log_w[None, :] - M).sum(axis=1))
    return ll_per_person.sum()

def posterior_weights(X, a_vec, b_vec):
    P_jk = item_p_at_nodes(a_vec, b_vec)
    logP   = np.log(P_jk + 1e-12)
    log1mP = np.log(1.0 - P_jk + 1e-12)
    cond_ll = X @ logP + (1 - X) @ log1mP                 # (N, K)
    log_w   = np.log(weights + 1e-300)[None, :]
    log_post = cond_ll + log_w
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post                                            # (N, K)

def m_step_item(r_jk, n_jk, a_j, b_j, max_iter=20, tol=1e-6):
    """
    Newton's method for 2PL on weighted (theta_k, r_jk, n_jk).
    Maximizes sum_k [r_jk log p_k + (n_jk - r_jk) log(1 - p_k)]
    where p_k = sigmoid(a_j (X_k - b_j)).
    """
    for _ in range(max_iter):
        z   = a_j * (nodes - b_j)
        p   = sigmoid(z)
        wp  = n_jk * p * (1.0 - p)
        res = r_jk - n_jk * p

        # gradient
        gA = float(((nodes - b_j) * res).sum())
        gB = float(-a_j * res.sum())

        # negative Hessian (info matrix for max)
        HAA = float(((nodes - b_j) ** 2 * wp).sum()) + 1e-8
        HBB = float(a_j ** 2 * wp.sum()) + 1e-8
        HAB = float(-a_j * ((nodes - b_j) * wp).sum() + res.sum() * 0.0)
        # Note: cross derivative simplifies because at the optimum sum(res) -> 0.
        # Use a stable approximation:
        det = HAA * HBB - HAB * HAB
        if det < 1e-10:
            HAB = 0.0
            det = HAA * HBB

        # Newton step:  delta = H^{-1} g
        dA = (HBB * gA - HAB * gB) / det
        dB = (-HAB * gA + HAA * gB) / det

        # bounded step
        dA = float(np.clip(dA, -0.5, 0.5))
        dB = float(np.clip(dB, -0.5, 0.5))

        a_new = a_j + dA
        b_new = b_j + dB
        a_new = float(np.clip(a_new, 0.05, 4.0))
        b_new = float(np.clip(b_new, -5.0, 5.0))

        if abs(a_new - a_j) < tol and abs(b_new - b_j) < tol:
            a_j, b_j = a_new, b_new
            break
        a_j, b_j = a_new, b_new
    return a_j, b_j

# ---------------------------------------------------------------------------
# EM iterations
# ---------------------------------------------------------------------------
print("Step 4b - 2PL Marginal MLE (Bock-Aitkin EM)")
print(f"  N = {N}, J = {J}, quadrature K = {K}")
prev_ll = -np.inf
for it in range(1, 81):
    # E-step: posterior weights
    post = posterior_weights(X, a, b)                       # (N, K)
    # expected counts per item per node
    n_jk = post.sum(axis=0)                                  # (K,) - same for every item
    r_jk_mat = X.T @ post                                    # (J, K)

    # M-step: per-item Newton
    for j in range(J):
        a[j], b[j] = m_step_item(r_jk_mat[j], n_jk, a[j], b[j])

    ll = marginal_loglik(X, a, b)
    if it % 5 == 0 or it == 1:
        print(f"  iter {it:3d}: marginal log-lik = {ll:.3f}  (delta = {ll - prev_ll:+.4f})")
    if abs(ll - prev_ll) < 1e-3 and it > 5:
        print(f"  Converged at iter {it}.")
        break
    prev_ll = ll

# ---------------------------------------------------------------------------
# EAP person abilities
# ---------------------------------------------------------------------------
post = posterior_weights(X, a, b)
theta_hat = post @ nodes
theta_se  = np.sqrt(post @ (nodes ** 2) - theta_hat ** 2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
calibrated = items[["ItemID", "Question", "OptionA", "OptionB", "OptionC",
                    "OptionD", "CorrectAnswer", "Difficulty", "Category"]].copy()
calibrated["a"] = np.round(a, 4)
calibrated["b"] = np.round(b, 4)
cal_path = os.path.join(DATA_DIR, "calibrated_item_bank_mml.csv")
calibrated.to_csv(cal_path, index=False)
# NOTE: This script does NOT overwrite calibrated_item_bank.csv.
# The canonical calibration is step4_calibration_2pl.py.

theta_path = os.path.join(DATA_DIR, "calibrated_thetas_mml.csv")
pd.DataFrame({
    "RespondentID": resp.index,
    "theta_hat": np.round(theta_hat, 4),
    "se_theta":  np.round(theta_se, 4),
}).to_csv(theta_path, index=False)

# ---------------------------------------------------------------------------
# Recovery diagnostics
# ---------------------------------------------------------------------------
a_true = items["a_true"].to_numpy()
b_true = items["b_true"].to_numpy()
theta_true = true_thr["theta_true"].to_numpy()

corr_a = np.corrcoef(a_true, a)[0, 1]
corr_b = np.corrcoef(b_true, b)[0, 1]
corr_t = np.corrcoef(theta_true, theta_hat)[0, 1]
rmse_a = np.sqrt(np.mean((a - a_true) ** 2))
rmse_b = np.sqrt(np.mean((b - b_true) ** 2))
rmse_t = np.sqrt(np.mean((theta_hat - theta_true) ** 2))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report = []
def w(s=""): report.append(s)
w("=" * 72)
w("DSAN 6550 - Step 4b: Marginal MLE (Bock-Aitkin) Calibration")
w("=" * 72)
w(f"Sample size N = {N}, items J = {J}, quadrature K = {K}")
w(f"Final marginal log-likelihood : {prev_ll:.2f}")
w()
w("Item parameter recovery (calibrated vs true):")
w(f"  corr(a_hat, a_true)  = {corr_a:.4f}    RMSE(a) = {rmse_a:.4f}")
w(f"  corr(b_hat, b_true)  = {corr_b:.4f}    RMSE(b) = {rmse_b:.4f}")
w()
w("Person ability recovery (EAP vs true theta):")
w(f"  corr(theta_hat, theta_true) = {corr_t:.4f}    RMSE = {rmse_t:.4f}")
w()
w("Calibrated item parameters (MML):")
w(calibrated[["ItemID", "Difficulty", "a", "b"]].to_string(index=False))
w("=" * 72)
report_text = "\n".join(report)
diag_path = os.path.join(OUT_DIR, "calibration_diagnostics_mml.txt")
with open(diag_path, "w") as fh:
    fh.write(report_text)
print()
print(report_text)

# ---------------------------------------------------------------------------
# Recovery scatter
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(a_true, a, c="#1f77b4", edgecolor="white")
lo, hi = min(a_true.min(), a.min()) - 0.1, max(a_true.max(), a.max()) + 0.1
axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[0].set(xlabel="True a", ylabel="Estimated a", title=f"Discrimination (r={corr_a:.3f})")

axes[1].scatter(b_true, b, c="#d62728", edgecolor="white")
lo, hi = min(b_true.min(), b.min()) - 0.2, max(b_true.max(), b.max()) + 0.2
axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[1].set(xlabel="True b", ylabel="Estimated b", title=f"Difficulty (r={corr_b:.3f})")

axes[2].scatter(theta_true, theta_hat, c="#2ca02c", edgecolor="white", alpha=0.6, s=18)
lo, hi = min(theta_true.min(), theta_hat.min()) - 0.3, max(theta_true.max(), theta_hat.max()) + 0.3
axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
axes[2].set(xlabel="True theta", ylabel="EAP theta_hat", title=f"Ability (r={corr_t:.3f})")

fig.suptitle("2PL MML/EM Parameter Recovery - Soccer/World Cup Item Bank", y=1.02)
fig.tight_layout()
recovery_path = os.path.join(PLOT_DIR, "recovery_scatter_mml.png")
fig.savefig(recovery_path, dpi=130, bbox_inches="tight")
plt.close(fig)

print()
print(f"Saved -> {cal_path}")
print(f"Saved -> {theta_path}")
print(f"Saved -> {diag_path}")
print(f"Saved -> {recovery_path}")
