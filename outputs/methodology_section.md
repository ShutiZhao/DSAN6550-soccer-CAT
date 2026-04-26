# Methodology — Soccer / FIFA World Cup CAT
*DSAN 6550, Spring 2026 — Final Project*
*Role 1: Psychometric Lead (Emily Zhao)*

This section documents how the item bank was built, how the simulated
response data were generated, how data quality was verified, and how
the 2PL item parameters were calibrated. The terminology and methods
follow Hambleton, Swaminathan, and Rogers (1991, *Fundamentals of
Item Response Theory*) and the lectures from Sessions 2–6 of DSAN 6550.

---

## 1. Topic and Target Audience

Our content domain is **soccer / FIFA World Cup knowledge**. The 2026
men's FIFA World Cup is approximately two months from our presentation
date, which makes this a topical and engaging demo. Following the
project guidelines, we defined our target population as
**adult soccer / World Cup fans whose knowledge ranges from casual to
expert**. This wide ability range is exactly what a CAT is designed to
serve: a single fixed test cannot efficiently measure both casual fans
and historians, but an adaptive test can.

## 2. Item Generation

Items were drafted with AI assistance (ChatGPT-style generation, then
edited and verified by hand). The bank contains **30 four-option
multiple-choice items** distributed across three a priori difficulty
strata and five content categories:

| Difficulty | # items | Content categories represented            |
|------------|--------:|-------------------------------------------|
| Easy       |      10 | rules, basics, history, players           |
| Medium     |      10 | history, players, records                 |
| Hard       |      10 | history, records, players, format         |

Each item is stored with the metadata required by the CAT engine:
question text, four options (A–D), correct answer letter, difficulty
tag, and content category. The full item table is in
`data/items_metadata.csv`.

For each item we also pre-assigned a *true* discrimination $a_j$ and
*true* difficulty $b_j$ — these are the "ground-truth" parameters the
simulator uses. We drew $a_j$ from a clipped log-normal distribution
(mean ≈ 1.3, range [0.8, 2.5]) and $b_j$ on a smooth gradient within
each difficulty stratum:

- Easy: $b_j \in [-2.2, -0.8]$
- Medium: $b_j \in [-0.5, 0.5]$
- Hard: $b_j \in [0.8, 2.2]$

A small Gaussian jitter was added so the difficulty curve is monotone
without being mechanically linear. This stratified design is what
guarantees the assignment hint, "easier items should be answered
correctly by more people."

## 3. Response Simulation (2PL)

We simulated **N = 500 respondents** under the two-parameter logistic
(2PL) model. Each respondent's latent ability was drawn from the
standard-normal prior typical for IRT calibration:

$$\theta_i \;\sim\; \mathcal{N}(0, 1), \qquad i = 1, 2, \ldots, 500.$$

For each respondent–item pair $(i, j)$, the probability of a correct
response is

$$P(Y_{ij} = 1 \mid \theta_i, a_j, b_j) \;=\; \frac{1}{1 + \exp\!\big(-a_j (\theta_i - b_j)\big)},$$

and the observed response $Y_{ij} \in \{0, 1\}$ is drawn as a Bernoulli
variate with that probability. Because $b_j$ is stratified by
difficulty tag, the resulting response patterns reproduce the expected
difficulty curve:

| Stratum | Mean $p$-correct | Range            |
|---------|-----------------:|------------------|
| Easy    | 0.81             | 0.67 – 0.93      |
| Medium  | 0.49             | 0.41 – 0.59      |
| Hard    | 0.22             | 0.13 – 0.37      |

The simulated $\theta$ values are stored in `data/simulated_thetas.csv`
but are treated as **unknown** by the calibration step — they are only
used at the end as a ground-truth reference for parameter recovery.

## 4. Data Quality (Reliability and Validity)

### 4.1 Reliability — internal consistency

We computed Cronbach's alpha on the 500 × 30 binary response matrix.
For dichotomous items, this is equivalent to KR-20:

$$\alpha = \frac{J}{J - 1} \left(1 - \frac{\sum_{j=1}^{J} \mathrm{Var}(X_j)}{\mathrm{Var}\!\left(\sum_{j} X_j\right)}\right).$$

We also computed split-half reliability (odd vs. even items) with the
Spearman-Brown correction for completeness. Results:

| Statistic                       | Value  |
|---------------------------------|-------:|
| Cronbach's α (KR-20)           | **0.868** |
| Raw odd–even split-half $r$     | 0.763  |
| Spearman-Brown corrected $r$    | 0.865  |

Both estimates are in the **good** range (≥ 0.80), well above the
conventional 0.70 cutoff for instrument use.

### 4.2 Validity — item analysis

For each item we computed:

- **p-value** (proportion correct).
- **Corrected item-total correlation (CITC)** — the point-biserial
  correlation between the item and the total score *excluding that
  item*. Items with CITC < 0.20 are flagged as poorly discriminating.
- **Discrimination index $D = p_\text{high} - p_\text{low}$** —
  difference in p-correct between the top and bottom thirds of total
  score. This is a CTT analog of the IRT discrimination parameter.
- **Monotonicity in θ** — within the simulated $\theta$ tertiles, we
  checked that $p_\text{low} < p_\text{mid} < p_\text{high}$.

Results:

- CITC mean = **0.39** (range 0.23 – 0.55), all 30 items above the
  0.20 cutoff.
- Mean discrimination index $D = 0.41$.
- All 30 items are monotone in $\theta$ across tertiles.
- **No items were flagged**: all p-values are within $[0.10, 0.95]$
  and all CITC values are above 0.20.

The full per-item table is in `outputs/item_analysis.csv` and the
narrative report is in `outputs/reliability_report.txt`.

## 5. Item Parameter Calibration (2PL)

### 5.1 Method — per-item logistic regression

We chose the 2PL model (Hambleton et al., 1991, ch. 2). The 2PL
probability of a correct response can be re-written as a logistic
regression of $Y_{ij}$ on $\theta_i$:

$$P(Y_{ij} = 1) = \frac{1}{1 + \exp\!\big(-(\beta_{1j}\, \theta_i + \beta_{0j})\big)},$$

with the IRT–logistic mapping

$$a_j = \beta_{1j}, \qquad b_j = -\, \frac{\beta_{0j}}{\beta_{1j}}.$$

Calibration therefore reduces to fitting $J = 30$ separate logistic
regressions of each item's responses on $\theta$. We use
`sklearn.linear_model.LogisticRegression` with negligible
regularisation (`C = 1e6`) so the estimates are essentially the
maximum-likelihood solutions. (A small Newton–Raphson fallback in
pure NumPy is provided for environments without scikit-learn; it
returns identical estimates.)

Because the true $\theta_i$ are not known in real applications, we
follow the standard CTT practice and use a **provisional theta** equal
to the z-scored total score:

$$\hat{\theta}_i = \frac{\sum_{j} Y_{ij} - \overline{S}}{\mathrm{sd}(S)}.$$

For a reasonably reliable test the proxy correlates strongly with the
true latent ability. Empirically, in our simulation,
$\mathrm{corr}(\hat\theta_\text{proxy}, \theta_\text{true}) = 0.93$,
which validates this choice.

### 5.2 Parameter recovery

Because we generated the data ourselves, we can compare the calibrated
$(\hat a_j, \hat b_j)$ with the true $(a_j, b_j)$:

| Quantity                                  | Pearson $r$ | RMSE  |
|-------------------------------------------|------------:|------:|
| Discrimination $a$                        | **0.954**   | 0.235 |
| Difficulty $b$                            | **0.993**   | 0.163 |
| Person ability $\theta$ (proxy vs true)   | 0.933       | 0.380 |

The rank ordering of items — which is what the CAT's
maximum-information rule actually uses — is preserved with very high
fidelity for both parameters. The recovery scatterplots are in
`plots/recovery_scatter.png`.

### 5.3 Optional advanced extension (Marginal MLE)

For students who wish to discuss it, we also implemented a Bock–Aitkin
EM solver (Marginal Maximum Likelihood) in
`scripts/step4b_calibration_mml.py`. MML integrates $\theta$ out under
its $\mathcal{N}(0,1)$ prior using a 41-point quadrature grid, and
recovers the items even more cleanly (RMSE on $a$ drops from 0.24 to
0.15, RMSE on $\theta$ EAP drops from 0.38 to 0.37). This is the
algorithm used by the R `mirt` package and the Python `girth` package.
**It is optional and not required for the project**; the per-item
logistic regression in §5.1 is the primary calibration the CAT engine
consumes.

## 6. Deliverables for Role 2

The Systems Lead (Zhengyuan) will read **`data/calibrated_item_bank.csv`**.
Each row contains everything the CAT engine needs:

- `ItemID`, `Question`, `OptionA`–`OptionD`, `CorrectAnswer`
- `Difficulty` (Easy/Medium/Hard tag — useful for showing the user
  what the system thinks the item's difficulty is)
- `Category` (rules / history / players / records / format)
- `a` — calibrated discrimination
- `b` — calibrated difficulty

For the CAT engine, item information at ability $\theta$ under 2PL is

$$I_j(\theta) = a_j^2 \cdot P_j(\theta)\,\big(1 - P_j(\theta)\big),$$

and the next item should be chosen as
$\arg\max_{j \notin \text{used}} I_j(\hat\theta)$ (Session 6). Standard
error of $\hat\theta$ at any point is
$\mathrm{SE}(\hat\theta) = 1 / \sqrt{\sum_j I_j(\hat\theta)}$, summed
over administered items.

## 7. Limitations

1. **Simulated, not real, respondents.** We have no field-test data;
   true psychometric properties depend on real test-takers.
2. **Theta proxy.** Using z-scored total score as $\theta$ is a CTT
   approximation; a fully marginal MLE (our optional §5.3) is more
   defensible for a production CAT. With $N = 500$ we found this had
   limited practical effect on item rankings.
3. **2PL only.** A 3PL model would explicitly account for guessing on
   four-option items (lower asymptote ≈ 0.25). With $N = 500$ the
   guessing parameter is hard to identify stably, so we kept 2PL.
4. **Unidimensionality assumption.** 2PL assumes a single underlying
   trait. Soccer knowledge has subdomains (rules, history, players,
   tactics, records); a multi-dimensional model could capture this
   but is outside the scope of a course final project.
5. **Item content coverage.** 30 items is the project minimum; an
   operational CAT would typically need 100+ items to support large
   item exposure constraints.

## References

Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991).
*Fundamentals of item response theory* (Vol. 2). Sage.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood
estimation of item parameters: Application of an EM algorithm.
*Psychometrika*, 46(4), 443–459.

Lord, F. M. (1980). *Applications of item response theory to
practical testing problems*. Erlbaum.
