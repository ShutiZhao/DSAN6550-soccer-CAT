# DSAN 6550 Final Project — Soccer / FIFA World Cup CAT
*Emily Zhao & Zhengyuan Wang*

A Computerized Adaptive Test on **soccer / FIFA World Cup knowledge**.
This README covers **Emily Zhao** (back-end / data).

## Folder layout

```
DSAN_6550_Final_Project_2026/
|-- README.md                          (this file)
|-- scripts/
|   |-- step1_generate_items.py        # 30 items + true (a, b)
|   |-- step2_simulate_responses.py    # 500 simulated responses (2PL)
|   |-- step3_reliability_validity.py  # Cronbach's alpha, item analysis
|   |-- step4_calibration_2pl.py       # PRIMARY: per-item logistic regression
|   `-- step4b_calibration_mml.py      # OPTIONAL: Bock-Aitkin EM (advanced) 
|-- data/
|   |-- items_metadata.csv             # questions, options, true (a, b)
|   |-- item_bank_true.csv             # ground-truth parameters only
|   |-- simulated_responses.csv        # 500 x 30 binary matrix
|   |-- simulated_thetas.csv           # true theta per respondent
|   |-- response_summary.csv           # per-item p-correct + true (a, b)
|   |-- calibrated_item_bank.csv       # <- FRONT-END READS THIS
|   |-- calibrated_thetas.csv          # estimated theta per respondent
|   |-- calibrated_item_bank_mml.csv   # MML alternative (optional)
|   `-- calibrated_thetas_mml.csv
|-- outputs/
|   |-- reliability_report.txt         # narrative quality report
|   |-- item_analysis.csv              # per-item CTT statistics
|   |-- calibration_diagnostics.txt    # primary calibration report
|   |-- calibration_diagnostics_mml.txt
|   `-- methodology_section.md         # writeup for the final report
`-- plots/
    |-- recovery_scatter.png           # primary calibration recovery
    `-- recovery_scatter_mml.png       # MML calibration recovery
```

## How to run (any order from a fresh checkout)

```bash
cd scripts
python3 step1_generate_items.py
python3 step2_simulate_responses.py
python3 step3_reliability_validity.py
python3 step4_calibration_2pl.py        # primary
python3 step4b_calibration_mml.py       # optional
```

Required Python packages: `numpy`, `pandas`, `matplotlib`,
`scikit-learn` (the calibration script will fall back to a tiny NumPy
Newton–Raphson if scikit-learn is missing).

## Headline results

| Metric                                     | Value      |
|--------------------------------------------|------------|
| Items in pool                              | 30         |
| Simulated respondents                      | 500        |
| Cronbach's alpha (KR-20)                  | 0.868      |
| Spearman-Brown split-half reliability      | 0.865      |
| Items flagged in CTT screening             | 0          |
| Recovery: corr(â, a_true)                  | 0.95       |
| Recovery: corr(b̂, b_true)                  | 0.99       |
| Recovery: corr(θ̂, θ_true)                  | 0.93       |
| Easy / Medium / Hard mean p-correct        | 0.81 / 0.49 / 0.22 |

See `outputs/methodology_section.md` for the full methodology writeup
suitable for inclusion in the final report.

## Hand-off to Role 2

Read **`data/calibrated_item_bank.csv`**. The columns are:

`ItemID, Question, OptionA, OptionB, OptionC, OptionD, CorrectAnswer, Difficulty, Category, a, b`

For the CAT engine:
- 2PL probability: $P_j(\theta) = 1 / (1 + \exp(-a_j(\theta - b_j)))$
- Item information: $I_j(\theta) = a_j^2 \cdot P_j(\theta) (1 - P_j(\theta))$
- Pick next item by $\arg\max_j I_j(\hat\theta)$ over unused items
- Standard error: $\mathrm{SE}(\hat\theta) = 1 / \sqrt{\sum_j I_j(\hat\theta)}$


