# Role 2 — AI Build Brief

*Paste this whole file into ChatGPT / Claude / Cursor as a single message,*
*along with the four data files listed below, to build the front-end + CAT engine.*

---

## Project context

I'm building Role 2 of a Computerized Adaptive Test (CAT) for my
DSAN 6550 final project. The topic is **soccer / FIFA World Cup
knowledge**. My partner already built Role 1 (the back-end / data
pipeline) and gave me four files in a `data/` folder:

- `calibrated_item_bank.csv` — 30 items with columns:
  `ItemID, Question, OptionA, OptionB, OptionC, OptionD,
  CorrectAnswer (letter), Difficulty (Easy/Medium/Hard), Category,
  a (discrimination), b (difficulty)`. **This is the only file the
  CAT engine reads at runtime.**
- `simulated_responses.csv` — 500 × 30 binary matrix of fake
  responses (RespondentID rows like `R1, R2, ...`). Use for offline
  testing.
- `simulated_thetas.csv` — true θ for each simulated respondent. Use
  to pick three demo respondents at low / medium / high ability.
- `methodology_section.md` — has the IRT formulas (already summarized
  below).

The presentation is in **2 days**, so I need this to actually work,
not be perfect.

## What to build

A Streamlit app (`cat_dashboard.py`) plus a separate `cat_engine.py`
module so the CAT logic is testable on its own.

**Three pages:**

1. **Intro** — project description and a "Start Test" button, with a
   sidebar option to switch between **Live mode** (real human
   answers) and **Demo mode** (pick one of three pre-selected
   simulated respondents and auto-answer using their recorded
   responses).
2. **Test** — shows the current item and four options, then after
   each answer shows the current θ̂ estimate, its SE, the next item's
   ICC and IIC plots, and a one-line "why this item was chosen"
   explanation.
3. **Results** — final θ̂ and SE, a history table of
   `(item ID, response, θ̂ after)`, a θ̂ trajectory plot, and a CSV
   download.

## CAT engine logic (put in `cat_engine.py`)

**The 2PL formulas:**

- Probability: $P_j(\theta) = 1 / (1 + \exp(-a_j (\theta - b_j)))$
- Item information: $I_j(\theta) = a_j^2 \cdot P_j(\theta) \cdot (1 - P_j(\theta))$
- Standard error of θ̂: $\mathrm{SE} = 1 / \sqrt{\sum_j I_j(\hat\theta)}$
  summed over administered items

**Algorithm:**

1. Initialize `θ̂ = 0` and an empty `used_items` set.
2. For every unused item, compute `I_j(θ̂)`. Pick the item with the
   maximum information (break ties by ItemID). One item is
   administered only once per respondent.
3. Show the item, collect the response `y ∈ {0, 1}`.
4. Update θ̂ from all responses so far. The simplest method is
   **Newton-Raphson MLE**:
   $\hat\theta_\text{new} = \hat\theta + \frac{\sum_j a_j (y_j - P_j)}{\sum_j a_j^2 P_j (1 - P_j)}$,
   iterate ~10 times. **Cap θ̂ at [-4, 4]** to handle the "all wrong"
   or "all right" edge cases (otherwise MLE diverges to ±∞).
5. Compute SE.
6. **Stop** if `SE < 0.30` OR 20 items have been administered,
   whichever comes first.

## Visualizations

Matplotlib is fine, doesn't need to be pretty. After each answer,
show four panels:

1. **Bar chart of information** at θ̂ for all unused items, with the
   chosen one highlighted in green — this is the "why" visual.
2. **ICC** of the selected item (P vs θ) with a vertical line at
   current θ̂.
3. **IIC** of the selected item (I vs θ) with the same vertical line.
4. **Trajectory** of θ̂ vs item number so far, with a ±1 SE band.

## Three demo respondents (project requirement #6)

From `simulated_thetas.csv`, pre-pick three respondents whose true θ
is approximately **−1.5, 0.0, and +1.5**. Hard-code their
RespondentIDs in the app. In demo mode, the app auto-answers each
item using the value already in `simulated_responses.csv` for that
respondent and item. This lets me deterministically show three
different ability levels in class.

## Explainability (project requirement #7)

After every item is selected, render a one-sentence explanation like:

> Selected Q15 (b = -0.21, a = 1.32) because it has the highest
> information (I = 0.43) at your current ability estimate
> θ̂ = -0.30. The next-best alternative was Q12 with I = 0.41.

## Files to deliver

- **`cat_engine.py`** — pure-Python module with these functions:
  - `next_item(theta_hat, item_bank, used_ids)`
  - `update_theta(responses, item_params)`
  - `info_at(theta, a, b)`
  - `prob_at(theta, a, b)`
  - `se_at(theta, item_params)`
- **`cat_dashboard.py`** — the Streamlit app that imports from
  `cat_engine.py`.
- A short docstring at the top of each file explaining what it does.
- Test the engine first with the simulated data: load R250 from
  `simulated_responses.csv`, run the engine on her responses,
  confirm it converges to roughly her true θ.

## Tech

- Streamlit, matplotlib, numpy, pandas.
- Soccer-themed colors are nice but optional (green `#0a7c2c` on
  white).
- Run with `streamlit run cat_dashboard.py`.
- Use `st.session_state` to track theta, responses, and used items
  across reruns.

## What I want back

Both Python files, a list of any assumptions you made, and a
one-paragraph guide to running the demo so I can rehearse it before
Monday.
