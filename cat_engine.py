"""
cat_engine.py
=============
Pure-Python 2PL Computerized Adaptive Testing engine.
DSAN 6550 Spring 2026 Final Project (Soccer / FIFA World Cup CAT).

This module has no Streamlit dependencies, so we can test it on its own.
The Streamlit dashboard (cat_dashboard.py) imports the functions from here.

Public functions
----------------
prob_at(theta, a, b)               -> 2PL P(correct)
info_at(theta, a, b)               -> 2PL item information
se_at(theta, item_params)          -> standard error of theta_hat
next_item(theta_hat, bank, used)   -> the unused item with maximum information
update_theta(responses, items)     -> Newton-Raphson MLE of theta
should_stop(n, se)                 -> True if SE < 0.30 OR n >= 20

Formulas come from the methodology document written by Role 1 and from
Hambleton, Swaminathan & Rogers (1991), the textbook listed in the
DSAN 6550 syllabus.
"""

import math
import os

import numpy as np
import pandas as pd


# ----- Constants from the project spec --------------------------------------

THETA_MIN, THETA_MAX = -4.0, 4.0   # cap to keep MLE finite when all-right or all-wrong
SE_STOP   = 0.30                   # stop when standard error drops below this
MAX_ITEMS = 20                     # stop after this many items
NEWTON_ITERS = 25                  # max Newton-Raphson iterations (usually converges in 5-8)
NEWTON_TOL   = 1e-6                # early stop when |delta theta| < tol


# ----- Core 2PL formulas ----------------------------------------------------

def prob_at(theta, a, b):
    """
    2PL probability of a correct response:

        P(theta) = 1 / (1 + exp(-a * (theta - b)))

    theta, a, b can be floats or numpy arrays (broadcast as usual).
    The np.clip on z just avoids harmless overflow warnings for huge |z|;
    the logistic saturates at 0 or 1 there anyway.
    """
    z = a * (theta - b)
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def info_at(theta, a, b):
    """
    2PL item (Fisher) information:

        I(theta) = a^2 * P(theta) * (1 - P(theta))
    """
    p = prob_at(theta, a, b)
    return (a ** 2) * p * (1.0 - p)


def se_at(theta, item_params):
    """
    Standard error of theta_hat from the items administered so far:

        SE(theta) = 1 / sqrt(sum_j I_j(theta))

    item_params: list of dicts with keys 'a' and 'b'.
    Returns +inf if no items have been administered yet.
    """
    if len(item_params) == 0:
        return float("inf")
    a_arr = np.array([p["a"] for p in item_params])
    b_arr = np.array([p["b"] for p in item_params])
    total_info = float(np.sum(info_at(theta, a_arr, b_arr)))
    if total_info <= 0:
        return float("inf")
    return 1.0 / math.sqrt(total_info)


# ----- Item selection (CAT next-item rule) ----------------------------------

def next_item(theta_hat, item_bank, used_ids):
    """
    Pick the unused item with the maximum information at theta_hat.
    Ties are broken by ItemID (alphabetical) so the demo is reproducible.

    Returns a dict with:
        item_id   -- chosen ItemID
        info      -- I_j(theta_hat) for the chosen item
        a, b      -- parameters of the chosen item
        all_infos -- DataFrame of (ItemID, info, a, b) for ALL unused items,
                     sorted by info descending. Used by the bar chart and
                     the "why this item" explanation in the dashboard.
    """
    used = set(used_ids)
    avail = item_bank.loc[~item_bank["ItemID"].isin(used)].copy()
    if len(avail) == 0:
        raise RuntimeError("next_item: no unused items left in the bank")

    avail["info"] = info_at(theta_hat,
                            avail["a"].to_numpy(),
                            avail["b"].to_numpy())
    avail = avail.sort_values(["info", "ItemID"],
                              ascending=[False, True]).reset_index(drop=True)
    winner = avail.iloc[0]
    return {
        "item_id":   str(winner["ItemID"]),
        "info":      float(winner["info"]),
        "a":         float(winner["a"]),
        "b":         float(winner["b"]),
        "all_infos": avail[["ItemID", "info", "a", "b"]].copy(),
    }


# ----- Theta update (Newton-Raphson MLE) ------------------------------------

def update_theta(responses, item_params, theta_init=0.0):
    """
    Newton-Raphson update for the MLE of theta given the responses so far.

    Score:    U(theta) = sum_j a_j * (y_j - P_j(theta))
    Info:     J(theta) = sum_j a_j^2 * P_j(theta) * (1 - P_j(theta))
    Update:   theta_new = theta_old + U / J

    With an all-correct or all-wrong pattern the MLE diverges to +/- infinity,
    so we (a) shortcut directly to the cap and (b) clamp every intermediate
    step to [THETA_MIN, THETA_MAX]. This is the standard fix-up; see the
    methodology doc (Role 1) and Hambleton et al. (1991), ch. 3.
    """
    if len(responses) != len(item_params):
        raise ValueError("responses and item_params must have the same length")
    if len(responses) == 0:
        return 0.0   # no information yet -> prior mean

    y = np.asarray(responses, dtype=float)
    a = np.array([p["a"] for p in item_params])
    b = np.array([p["b"] for p in item_params])

    # All-correct / all-wrong: MLE is at the cap; skip iteration.
    if np.all(y == 1):
        return THETA_MAX
    if np.all(y == 0):
        return THETA_MIN

    theta = float(theta_init)
    for _ in range(NEWTON_ITERS):
        p = prob_at(theta, a, b)
        score = float(np.sum(a * (y - p)))
        info  = float(np.sum((a ** 2) * p * (1.0 - p)))
        if info <= 1e-9:
            break
        theta_new = theta + score / info
        # Clamp so subsequent P_j stay finite and we don't diverge.
        if theta_new > THETA_MAX:
            theta_new = THETA_MAX
        elif theta_new < THETA_MIN:
            theta_new = THETA_MIN
        if abs(theta_new - theta) < NEWTON_TOL:
            theta = theta_new
            break
        theta = theta_new

    return float(theta)


# ----- Stopping rule --------------------------------------------------------

def should_stop(n_administered, se):
    """
    Stopping rule from the project spec:
    stop when SE < 0.30 OR at least 20 items have been administered.
    """
    if n_administered >= MAX_ITEMS:
        return True
    if n_administered >= 1 and se < SE_STOP:
        return True
    return False


# ----- Smoke test (run with `python cat_engine.py`) -------------------------

if __name__ == "__main__":
    # Verify the engine on a known respondent: load R250 from the simulated
    # responses, run the CAT, and compare to her recorded theta_hat.
    here = os.path.dirname(os.path.abspath(__file__))
    bank = pd.read_csv(os.path.join(here, "data", "calibrated_item_bank.csv"))
    resp = pd.read_csv(os.path.join(here, "data", "simulated_responses.csv"))
    th   = pd.read_csv(os.path.join(here, "data", "calibrated_thetas.csv"))

    print(f"Loaded bank: {bank.shape}, responses: {resp.shape}")
    print()

    target_id = "R250"
    if target_id not in resp["RespondentID"].values:
        target_id = resp["RespondentID"].iloc[249]
    recorded = float(th.loc[th["RespondentID"] == target_id,
                            "theta_hat"].iloc[0])
    print(f"Smoke test on {target_id} (recorded theta_hat = {recorded:+.3f})")

    sim = resp.loc[resp["RespondentID"] == target_id].iloc[0]
    theta = 0.0
    used, administered = [], []
    for step in range(1, MAX_ITEMS + 1):
        sel = next_item(theta, bank, used)
        y = int(sim[sel["item_id"]])
        used.append(sel["item_id"])
        administered.append({"a": sel["a"], "b": sel["b"], "y": y})
        theta = update_theta([d["y"] for d in administered],
                             administered, theta_init=theta)
        se = se_at(theta, administered)
        print(f"  step {step:2d}: {sel['item_id']:>4} y={y}  "
              f"theta_hat={theta:+.3f}  SE={se:.3f}")
        if should_stop(step, se):
            break

    print()
    print(f"Final theta_hat: {theta:+.3f}")
    print(f"Recorded:        {recorded:+.3f}")
    print(f"Difference:      {abs(theta - recorded):.3f}")
