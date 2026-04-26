"""
Step 1 - Item Generation
DSAN 6550 Final Project (Spring 2026)
Role 1: Psychometric Lead

Generates the soccer / FIFA World Cup item bank with full metadata
plus the TRUE 2PL parameters (a, b) used for the response simulation.

Topic: Soccer / FIFA World Cup (timely framing for the 2026 World Cup).
Target audience: World Cup fans / casual-to-expert soccer fans.

Output (saved to ../data/):
    items_metadata.csv     - human-readable items: question, options, correct answer,
                             difficulty tag, content category
    item_bank_true.csv     - the same items with the TRUE a, b parameters used
                             to generate responses (ground truth for calibration recovery)
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 30 soccer / World Cup items, balanced across Easy / Medium / Hard
# Categories: rules, basics, history, players, records, format
# Each row: (Question, A, B, C, D, CorrectLetter, DifficultyTag, Category)
# ---------------------------------------------------------------------------

ITEMS = [
    # ---------- EASY (10) ----------
    ("How many players from one team are on the field at the start of a regulation soccer match?",
     "9", "10", "11", "12", "C", "Easy", "rules"),
    ("How long is a regulation soccer match (excluding stoppage and extra time)?",
     "60 minutes", "80 minutes", "90 minutes", "120 minutes", "C", "Easy", "rules"),
    ("Which country has won the most FIFA World Cup titles (men)?",
     "Germany", "Italy", "Argentina", "Brazil", "D", "Easy", "history"),
    ("How often is the FIFA World Cup held?",
     "Every 2 years", "Every 3 years", "Every 4 years", "Every 5 years", "C", "Easy", "basics"),
    ("In which country was the 2022 FIFA World Cup hosted?",
     "Russia", "Qatar", "United Arab Emirates", "Saudi Arabia", "B", "Easy", "history"),
    ("Which player is commonly known by the nickname 'CR7'?",
     "Lionel Messi", "Cristiano Ronaldo", "Neymar", "Kylian Mbappe", "B", "Easy", "players"),
    ("What does a red card mean in a soccer match?",
     "A formal warning", "A goal is scored", "The player is sent off", "An offside call", "C", "Easy", "rules"),
    ("How many teams played in the final tournament of the 2022 FIFA World Cup?",
     "16", "24", "32", "48", "C", "Easy", "format"),
    ("Who scored the famous 'Hand of God' goal at the 1986 World Cup?",
     "Pele", "Diego Maradona", "Zinedine Zidane", "Johan Cruyff", "B", "Easy", "history"),
    ("Which governing body organizes the FIFA World Cup?",
     "UEFA", "FIFA", "CONMEBOL", "IOC", "B", "Easy", "basics"),

    # ---------- MEDIUM (10) ----------
    ("Who won the Golden Ball (best player) at the 2022 FIFA World Cup?",
     "Kylian Mbappe", "Lionel Messi", "Luka Modric", "Antoine Griezmann", "B", "Medium", "history"),
    ("Which country hosted the first FIFA World Cup in 1930?",
     "Brazil", "Italy", "Uruguay", "France", "C", "Medium", "history"),
    ("As of 2024, which player has won the most Ballon d'Or awards?",
     "Cristiano Ronaldo", "Lionel Messi", "Michel Platini", "Johan Cruyff", "B", "Medium", "records"),
    ("Which African nation reached the semifinals of the 2022 World Cup, a first for the continent?",
     "Senegal", "Morocco", "Cameroon", "Ghana", "B", "Medium", "history"),
    ("Which European country won UEFA Euro 2020 (played in 2021)?",
     "England", "France", "Italy", "Portugal", "C", "Medium", "history"),
    ("Who is the all-time leading goalscorer for the Germany men's national team?",
     "Gerd Muller", "Miroslav Klose", "Thomas Muller", "Jurgen Klinsmann", "B", "Medium", "players"),
    ("The 7-1 defeat in a 2014 World Cup semifinal was suffered by which host nation?",
     "South Africa", "Brazil", "Russia", "Qatar", "B", "Medium", "history"),
    ("In which year did goal-line technology debut at a FIFA World Cup?",
     "2010", "2014", "2018", "2022", "B", "Medium", "history"),
    ("Pele won his first World Cup with Brazil at age 17. In which year?",
     "1954", "1958", "1962", "1970", "B", "Medium", "history"),
    ("Who won the Golden Glove (best goalkeeper) at the 2022 World Cup?",
     "Hugo Lloris", "Thibaut Courtois", "Emiliano Martinez", "Yassine Bounou", "C", "Medium", "history"),

    # ---------- HARD (10) ----------
    ("Who won the Golden Boot at the 1978 FIFA World Cup?",
     "Mario Kempes", "Paolo Rossi", "Teofilo Cubillas", "Rob Rensenbrink", "A", "Hard", "history"),
    ("Which player holds the record for most goals scored in a single World Cup tournament?",
     "Gerd Muller", "Just Fontaine", "Pele", "Ronaldo (Brazil)", "B", "Hard", "records"),
    ("Lev Yashin remains the only player at his position ever to win the Ballon d'Or. What position did he play?",
     "Striker", "Midfielder", "Defender", "Goalkeeper", "D", "Hard", "players"),
    ("Hakan Sukur scored the fastest goal in World Cup history (2002). At how many seconds?",
     "11 seconds", "17 seconds", "23 seconds", "30 seconds", "A", "Hard", "records"),
    ("Who is the all-time top scorer in men's FIFA World Cup finals (across all editions)?",
     "Ronaldo (Brazil)", "Gerd Muller", "Miroslav Klose", "Just Fontaine", "C", "Hard", "records"),
    ("Which African team was the first from CAF to reach a World Cup quarterfinal, in 1990?",
     "Nigeria", "Senegal", "Ghana", "Cameroon", "D", "Hard", "history"),
    ("The 1950 World Cup did not have a traditional knockout final. How was the champion decided?",
     "Coin toss", "Final round-robin group", "Continuous tournament", "FIFA committee vote", "B", "Hard", "history"),
    ("Who scored the winning goal in the 1986 World Cup final, sealing Argentina's title?",
     "Diego Maradona", "Jorge Burruchaga", "Jorge Valdano", "Sergio Batista", "B", "Hard", "history"),
    ("As of the 2022 tournament, which player has appeared in the most men's FIFA World Cup matches?",
     "Cristiano Ronaldo", "Lionel Messi", "Lothar Matthaus", "Paolo Maldini", "B", "Hard", "records"),
    ("In which year did the men's FIFA World Cup expand to 32 teams in the final tournament?",
     "1982", "1994", "1998", "2002", "C", "Hard", "format"),
]

assert len(ITEMS) == 30, f"Expected 30 items, got {len(ITEMS)}"

# ---------------------------------------------------------------------------
# Build human-readable metadata table
# ---------------------------------------------------------------------------
metadata_rows = []
for i, (q, a, b, c, d, ans, diff, cat) in enumerate(ITEMS, start=1):
    metadata_rows.append({
        "ItemID": f"Q{i}",
        "Question": q,
        "OptionA": a,
        "OptionB": b,
        "OptionC": c,
        "OptionD": d,
        "CorrectAnswer": ans,
        "Difficulty": diff,
        "Category": cat,
    })
metadata = pd.DataFrame(metadata_rows)

# ---------------------------------------------------------------------------
# Assign TRUE 2PL parameters
#   - 'a' (discrimination): drawn from a log-normal-ish distribution clipped
#     to [0.8, 2.5] so all items discriminate reasonably.
#   - 'b' (difficulty): laid out on a smooth gradient WITHIN each difficulty
#     tag so the simulation reflects a logical difficulty curve:
#         Easy:   b in [-2.2, -0.8]
#         Medium: b in [-0.5,  0.5]
#         Hard:   b in [ 0.8,  2.2]
#     This guarantees easy items truly get higher p-correct on average.
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=20260424)  # presentation date for reproducibility

DIFF_RANGES = {
    "Easy":   (-2.2, -0.8),
    "Medium": (-0.5,  0.5),
    "Hard":   ( 0.8,  2.2),
}

a_true = np.empty(len(metadata))
b_true = np.empty(len(metadata))

for diff_tag, (b_lo, b_hi) in DIFF_RANGES.items():
    idx = metadata.index[metadata["Difficulty"] == diff_tag].to_numpy()
    # smooth ramp + a small jitter so b is monotone-ish but not perfectly linear
    ramp = np.linspace(b_lo, b_hi, len(idx))
    jitter = RNG.normal(0.0, 0.08, size=len(idx))
    b_true[idx] = ramp + jitter
    # discrimination: log-normal with mean ~1.4, clipped to [0.8, 2.5]
    a_draw = RNG.lognormal(mean=np.log(1.3), sigma=0.30, size=len(idx))
    a_true[idx] = np.clip(a_draw, 0.8, 2.5)

metadata["a_true"] = np.round(a_true, 4)
metadata["b_true"] = np.round(b_true, 4)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
metadata_path = os.path.join(DATA_DIR, "items_metadata.csv")
true_path     = os.path.join(DATA_DIR, "item_bank_true.csv")

metadata.to_csv(metadata_path, index=False)
metadata[["ItemID", "Difficulty", "Category", "a_true", "b_true"]].to_csv(true_path, index=False)

# Quick console summary
print("Step 1 complete: item bank generated.")
print(f"  Items: {len(metadata)}")
print(f"  Difficulty mix: {metadata['Difficulty'].value_counts().to_dict()}")
print(f"  Category mix:  {metadata['Category'].value_counts().to_dict()}")
print(f"  a_true: min={a_true.min():.2f}, max={a_true.max():.2f}, mean={a_true.mean():.2f}")
print(f"  b_true: min={b_true.min():.2f}, max={b_true.max():.2f}, mean={b_true.mean():.2f}")
print(f"  Saved -> {metadata_path}")
print(f"  Saved -> {true_path}")
