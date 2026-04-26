"""
Step 2 - Response Simulation (2PL)
DSAN 6550 Final Project (Spring 2026)
Role 1: Psychometric Lead

Simulates 500 respondents answering the 30 soccer items under the 2PL model:

        P(Y=1 | theta, a, b) = 1 / (1 + exp(-a * (theta - b)))

Design choices:
  * Person abilities theta_i ~ N(0, 1)              (i = 1, ..., 500)
  * True item parameters (a_true, b_true) come from step 1.
    Because b values are stratified by difficulty tag, easy items get
    HIGHER p-correct on average and hard items get LOWER p-correct,
    matching the assignment hint that 'easy items should be answered
    correctly by more people.'
  * Responses are drawn as Bernoulli(P) - NOT a simple deterministic
    threshold - so we keep realistic noise.

Outputs (../data/):
    simulated_responses.csv   - 500 x 30 binary matrix (RespondentID rows)
    simulated_thetas.csv      - true theta for each respondent (held out;
                                used only to validate calibration recovery)
    response_summary.csv      - per-item p-correct + true (a, b) for QA
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

# ---------------------------------------------------------------------------
# Load the true item bank produced by step 1
# ---------------------------------------------------------------------------
items_meta = pd.read_csv(os.path.join(DATA_DIR, "items_metadata.csv"))
item_ids   = items_meta["ItemID"].tolist()
a_true     = items_meta["a_true"].to_numpy()
b_true     = items_meta["b_true"].to_numpy()
n_items    = len(item_ids)

# ---------------------------------------------------------------------------
# Sample 500 respondent abilities
# ---------------------------------------------------------------------------
N_RESPONDENTS = 500
RNG = np.random.default_rng(seed=2026)
theta = RNG.normal(loc=0.0, scale=1.0, size=N_RESPONDENTS)

# ---------------------------------------------------------------------------
# 2PL probability and Bernoulli response draw
# ---------------------------------------------------------------------------
def p_2pl(theta_vec, a, b):
    """P(correct) for a single item across all respondents."""
    return 1.0 / (1.0 + np.exp(-a * (theta_vec - b)))

# probability matrix shape (N_RESPONDENTS, n_items)
P = np.empty((N_RESPONDENTS, n_items))
for j in range(n_items):
    P[:, j] = p_2pl(theta, a_true[j], b_true[j])

# Bernoulli draw
U = RNG.random(size=P.shape)
responses = (U < P).astype(int)

# ---------------------------------------------------------------------------
# Build dataframes
# ---------------------------------------------------------------------------
resp_df = pd.DataFrame(
    responses,
    index=[f"R{i+1}" for i in range(N_RESPONDENTS)],
    columns=item_ids,
)
resp_df.index.name = "RespondentID"

theta_df = pd.DataFrame({
    "RespondentID": [f"R{i+1}" for i in range(N_RESPONDENTS)],
    "theta_true": theta,
})

# Per-item QA: empirical p-correct vs true b
summary = items_meta[["ItemID", "Difficulty", "Category", "a_true", "b_true"]].copy()
summary["p_correct"] = resp_df.mean(axis=0).values
# Point-biserial-ish: corr(item, total_score) - quick discrimination sanity
total_score = resp_df.sum(axis=1).to_numpy()
summary["item_total_corr"] = [
    np.corrcoef(resp_df[itm].to_numpy(), total_score)[0, 1] for itm in item_ids
]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
resp_path    = os.path.join(DATA_DIR, "simulated_responses.csv")
theta_path   = os.path.join(DATA_DIR, "simulated_thetas.csv")
summary_path = os.path.join(DATA_DIR, "response_summary.csv")

resp_df.to_csv(resp_path)
theta_df.to_csv(theta_path, index=False)
summary.to_csv(summary_path, index=False)

# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
print("Step 2 complete: 500 responses simulated under 2PL.")
print(f"  Response matrix: {resp_df.shape}")
print(f"  Mean total score: {total_score.mean():.2f} / {n_items}")
print(f"  theta: mean={theta.mean():.3f}, sd={theta.std():.3f}")
print()
print("Per-difficulty p-correct (sanity check):")
print(summary.groupby("Difficulty")["p_correct"]
              .agg(["mean", "min", "max", "count"]).round(3))
print()
print("Top-5 most-answered (easiest in practice):")
print(summary.nlargest(5, "p_correct")[["ItemID", "Difficulty", "p_correct", "b_true"]].to_string(index=False))
print()
print("Bottom-5 (hardest in practice):")
print(summary.nsmallest(5, "p_correct")[["ItemID", "Difficulty", "p_correct", "b_true"]].to_string(index=False))
print()
print(f"Saved -> {resp_path}")
print(f"Saved -> {theta_path}")
print(f"Saved -> {summary_path}")
