"""
Step 3 - Reliability and Validity Checks
DSAN 6550 Final Project (Spring 2026)
Role 1: Psychometric Lead

Reliability:
  * Cronbach's alpha (KR-20 equivalent for 0/1 items)
  * Split-half reliability with Spearman-Brown correction
  * 'Alpha-if-item-deleted' diagnostics

Validity (classical-test-theory item analysis):
  * p-value (item difficulty in CTT sense)
  * Corrected item-total correlation (CITC) -> point-biserial discrimination
  * Flags for items with poor discrimination or extreme p-values

Outputs (../outputs/):
    reliability_report.txt    - human-readable summary for the report
    item_analysis.csv         - per-item CTT statistics
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.normpath(os.path.join(HERE, "..", "data"))
OUTPUT_DIR  = os.path.normpath(os.path.join(HERE, "..", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
resp_df  = pd.read_csv(os.path.join(DATA_DIR, "simulated_responses.csv"), index_col=0)
items    = pd.read_csv(os.path.join(DATA_DIR, "items_metadata.csv"))
X        = resp_df.to_numpy(dtype=float)            # (N, J)
N, J     = X.shape
item_ids = resp_df.columns.tolist()
total    = X.sum(axis=1)

# ---------------------------------------------------------------------------
# Reliability: Cronbach's alpha
# ---------------------------------------------------------------------------
def cronbach_alpha(matrix):
    item_var  = matrix.var(axis=0, ddof=1)
    total_var = matrix.sum(axis=1).var(ddof=1)
    k = matrix.shape[1]
    return (k / (k - 1.0)) * (1.0 - item_var.sum() / total_var)

alpha = cronbach_alpha(X)

# 'Alpha-if-deleted' for each item
alpha_if_deleted = []
for j in range(J):
    keep = [k for k in range(J) if k != j]
    alpha_if_deleted.append(cronbach_alpha(X[:, keep]))
alpha_if_deleted = np.array(alpha_if_deleted)

# Split-half reliability (odd vs even items) with Spearman-Brown
odd_idx  = np.arange(0, J, 2)
even_idx = np.arange(1, J, 2)
score_odd  = X[:, odd_idx].sum(axis=1)
score_even = X[:, even_idx].sum(axis=1)
r_half = np.corrcoef(score_odd, score_even)[0, 1]
sb     = (2.0 * r_half) / (1.0 + r_half)

# ---------------------------------------------------------------------------
# Validity / item analysis (classical test theory)
# ---------------------------------------------------------------------------
p_values = X.mean(axis=0)
sd_items = X.std(axis=0, ddof=1)

# Corrected item-total correlation: correlate each item with the
# total score MINUS that item (avoids spurious self-correlation).
citc = np.empty(J)
for j in range(J):
    rest = total - X[:, j]
    if X[:, j].std(ddof=1) == 0:
        citc[j] = np.nan
    else:
        citc[j] = np.corrcoef(X[:, j], rest)[0, 1]

# Flagging rules
flags = []
for j in range(J):
    f = []
    if p_values[j] < 0.10 or p_values[j] > 0.95:
        f.append("extreme_p")
    if citc[j] < 0.20:
        f.append("low_discrimination")
    flags.append(",".join(f) if f else "ok")

item_analysis = pd.DataFrame({
    "ItemID": item_ids,
    "Difficulty": items["Difficulty"].values,
    "Category": items["Category"].values,
    "p_value": np.round(p_values, 4),
    "sd": np.round(sd_items, 4),
    "CITC": np.round(citc, 4),
    "alpha_if_deleted": np.round(alpha_if_deleted, 4),
    "flag": flags,
})

# ---------------------------------------------------------------------------
# Validity by ability gradient (low/medium/high theta groups)
# ---------------------------------------------------------------------------
thetas = pd.read_csv(os.path.join(DATA_DIR, "simulated_thetas.csv"))["theta_true"].to_numpy()
low_mask  = thetas <= np.quantile(thetas, 1.0 / 3.0)
high_mask = thetas >= np.quantile(thetas, 2.0 / 3.0)
mid_mask  = ~(low_mask | high_mask)

p_low  = X[low_mask].mean(axis=0)
p_mid  = X[mid_mask].mean(axis=0)
p_high = X[high_mask].mean(axis=0)
gradient_ok = (p_high > p_mid) & (p_mid > p_low)
item_analysis["p_low_third"]  = np.round(p_low, 3)
item_analysis["p_mid_third"]  = np.round(p_mid, 3)
item_analysis["p_high_third"] = np.round(p_high, 3)
item_analysis["monotone_in_theta"] = gradient_ok

# Discrimination index (D) - classical CTT: p_high - p_low
item_analysis["D_index"] = np.round(p_high - p_low, 3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ia_path = os.path.join(OUTPUT_DIR, "item_analysis.csv")
item_analysis.to_csv(ia_path, index=False)

# Reliability report
report_lines = []
def w(s=""):
    report_lines.append(s)

w("=" * 72)
w("DSAN 6550 Final Project - Step 3: Reliability & Validity Report")
w("Topic: Soccer / FIFA World Cup CAT")
w("=" * 72)
w()
w(f"Sample size            : N = {N} respondents")
w(f"Number of items        : J = {J}")
w(f"Mean total score       : {total.mean():.2f} / {J}  (SD = {total.std(ddof=1):.2f})")
w()
w("-" * 72)
w("Reliability")
w("-" * 72)
w(f"Cronbach's alpha (KR-20)              : alpha = {alpha:.4f}")
w(f"Split-half (odd vs even, raw)         : r     = {r_half:.4f}")
w(f"Spearman-Brown corrected split-half   : r_SB  = {sb:.4f}")
w()
if alpha >= 0.90:
    interp = "excellent (>=0.90)"
elif alpha >= 0.80:
    interp = "good (>=0.80)"
elif alpha >= 0.70:
    interp = "acceptable (>=0.70)"
else:
    interp = "below conventional threshold (<0.70)"
w(f"Interpretation of alpha: {interp}")
w()
w("-" * 72)
w("Item-level validity (corrected item-total correlation, CITC)")
w("-" * 72)
w(f"CITC mean = {np.nanmean(citc):.3f}, min = {np.nanmin(citc):.3f}, max = {np.nanmax(citc):.3f}")
w(f"Items with CITC < 0.20 (low discrimination): "
  f"{(citc < 0.20).sum()}")
w(f"Items with extreme p-value (<.10 or >.95)  : "
  f"{((p_values < 0.10) | (p_values > 0.95)).sum()}")
w()
w("Per-difficulty summary:")
by_diff = item_analysis.groupby("Difficulty")[["p_value", "CITC", "D_index"]].mean().round(3)
w(by_diff.to_string())
w()
w("Validity by ability tertile (sanity check that p-correct is monotone in theta):")
w(f"  All 30 items monotone (p_low < p_mid < p_high)?  "
  f"{int(item_analysis['monotone_in_theta'].sum())} / {J}")
w(f"  Mean discrimination index D = p_high - p_low  : "
  f"{item_analysis['D_index'].mean():.3f}")
w()
w("-" * 72)
w("Flags")
w("-" * 72)
flagged = item_analysis[item_analysis["flag"] != "ok"]
if flagged.empty:
    w("No items flagged. All 30 items pass CTT screening.")
else:
    w("Flagged items (review before retaining in operational pool):")
    w(flagged[["ItemID", "Difficulty", "p_value", "CITC", "flag"]].to_string(index=False))
w()
w("-" * 72)
w("Conclusion")
w("-" * 72)
w(f"The 30-item soccer/World Cup pool shows {interp.split(' ')[0]} internal consistency")
w(f"(alpha = {alpha:.3f}). The CITC distribution and the monotone p-correct gradient")
w("across theta tertiles support criterion validity: items rank-order respondents")
w("of different ability levels in the expected direction.")
w("=" * 72)

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "reliability_report.txt")
with open(report_path, "w") as fh:
    fh.write(report_text)

print(report_text)
print()
print(f"Saved -> {ia_path}")
print(f"Saved -> {report_path}")
