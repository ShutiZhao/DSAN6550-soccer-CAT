"""
cat_dashboard.py
================
Streamlit dashboard for the Soccer / FIFA World Cup CAT.
DSAN 6550 Spring 2026 Final Project (Role 2: Systems Lead).

Three pages:
    1. Intro    -- project description and a Start Test button
    2. Test     -- current item, four diagnostic plots, "why this item"
    3. Results  -- final theta_hat, SE, history table, trajectory, CSV

Sidebar:
    Live mode -- the user answers each question.
    Demo mode -- replay one of three pre-selected simulated respondents
                 (R27 = low ability, R23 = medium, R44 = high).

Run with:
    streamlit run cat_dashboard.py
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import cat_engine as eng


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SOCCER_GREEN = "#0a7c2c"
SOCCER_GREEN_LIGHT = "#5fb87c"
GREY = "#444444"

# Three demo respondents from calibrated_thetas.csv (project requirement #6).
# Their recorded theta_hats are about -1.57, -0.04, and +1.48.
DEMO_RESPONDENTS = {
    "Casual fan (low ability, theta ~ -1.5)":   "R27",
    "Average fan (medium ability, theta ~ 0)":  "R23",
    "Expert fan (high ability, theta ~ +1.5)":  "R44",
}


# ---------------------------------------------------------------------------
# Data loading (cached so the CSVs are only read once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_item_bank():
    return pd.read_csv(os.path.join(DATA_DIR, "calibrated_item_bank.csv"))

@st.cache_data
def load_simulated_responses():
    return pd.read_csv(os.path.join(DATA_DIR, "simulated_responses.csv"))

@st.cache_data
def load_thetas():
    return pd.read_csv(os.path.join(DATA_DIR, "calibrated_thetas.csv"))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    """Initialise (or reset) all per-test session-state variables."""
    st.session_state.page = "intro"
    st.session_state.theta = 0.0
    st.session_state.se = float("inf")
    st.session_state.used_ids = []          # list of ItemIDs already administered
    st.session_state.administered = []      # list of {'a','b','y','item_id'}
    st.session_state.history = []           # list of {step,item_id,response,theta,se}
    st.session_state.current_item = None    # dict from next_item(), or None
    st.session_state.finished = False


def ensure_state():
    if "page" not in st.session_state:
        init_state()


def reset_for_new_test():
    """Clear test state. Mode and demo_choice are owned by their sidebar
    widgets, so we leave those keys alone -- Streamlit preserves them
    automatically across reruns. (Writing to them here raises
    StreamlitAPIException because the widgets have already been created.)"""
    init_state()


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib)
# ---------------------------------------------------------------------------

def _clean(ax):
    """Strip the top and right spines; smaller tick font."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_info_bar(selection, theta_hat):
    """
    Bar chart of information for all unused items at the current theta_hat,
    with the chosen item highlighted in green. This is the 'why' visual.
    """
    df = selection["all_infos"]
    winner = selection["item_id"]
    colors = [SOCCER_GREEN if iid == winner else "#bbbbbb"
              for iid in df["ItemID"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["ItemID"], df["info"], color=colors, edgecolor="white")
    ax.set_title(f"Information at theta_hat = {theta_hat:+.2f}  "
                 f"(green = chosen item)")
    ax.set_xlabel("Item ID")
    ax.set_ylabel("I_j(theta)")
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right", fontsize=7)
    _clean(ax); fig.tight_layout()
    return fig


def plot_icc(a, b, theta_hat, item_id):
    """Item Characteristic Curve P(theta) with a vertical line at theta_hat."""
    grid = np.linspace(-4, 4, 200)
    p = eng.prob_at(grid, a, b)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(grid, p, color=SOCCER_GREEN, linewidth=2)
    ax.axvline(theta_hat, color=GREY, linestyle="--", linewidth=1)
    ax.axhline(0.5, color="#cccccc", linewidth=0.8)
    ax.set_title(f"ICC of {item_id}  (a={a:.2f}, b={b:.2f})")
    ax.set_xlabel("theta")
    ax.set_ylabel("P(theta)")
    ax.set_ylim(-0.02, 1.02)
    _clean(ax); fig.tight_layout()
    return fig


def plot_iic(a, b, theta_hat, item_id):
    """Item Information Curve I(theta) with a vertical line at theta_hat."""
    grid = np.linspace(-4, 4, 200)
    info = eng.info_at(grid, a, b)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(grid, info, color=SOCCER_GREEN, linewidth=2)
    ax.axvline(theta_hat, color=GREY, linestyle="--", linewidth=1)
    ax.set_title(f"IIC of {item_id}  (a={a:.2f}, b={b:.2f})")
    ax.set_xlabel("theta")
    ax.set_ylabel("I(theta)")
    _clean(ax); fig.tight_layout()
    return fig


def plot_trajectory(history):
    """theta_hat vs item number, with a +/- 1 SE band."""
    if not history:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No items administered yet",
                ha="center", va="center", transform=ax.transAxes,
                color="#888888")
        ax.set_xticks([]); ax.set_yticks([])
        return fig
    steps  = [h["step"]  for h in history]
    thetas = [h["theta"] for h in history]
    ses    = [h["se"]    for h in history]
    upper  = [t + s for t, s in zip(thetas, ses)]
    lower  = [t - s for t, s in zip(thetas, ses)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(steps, lower, upper, color=SOCCER_GREEN_LIGHT,
                    alpha=0.25, label="+/- 1 SE")
    ax.plot(steps, thetas, color=SOCCER_GREEN, marker="o",
            linewidth=2, markersize=4, label="theta_hat")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.set_title("Trajectory of theta_hat")
    ax.set_xlabel("Item number administered")
    ax.set_ylabel("theta_hat")
    ax.legend(loc="best", fontsize=8, frameon=False)
    _clean(ax); fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Item flow
# ---------------------------------------------------------------------------

def select_next_item(bank):
    """Pick the next item and stash it (with its full row) in session_state."""
    sel = eng.next_item(st.session_state.theta, bank,
                        st.session_state.used_ids)
    sel["row"] = bank.loc[bank["ItemID"] == sel["item_id"]].iloc[0].to_dict()
    st.session_state.current_item = sel
    return sel


def submit_answer(letter, bank):
    """
    Score the chosen letter against the current item, update theta and SE,
    append to history, then either pick the next item or finish.
    """
    sel = st.session_state.current_item
    item = sel["row"]
    y = int(letter.upper() == str(item["CorrectAnswer"]).upper())

    st.session_state.used_ids.append(sel["item_id"])
    st.session_state.administered.append({
        "a": sel["a"], "b": sel["b"], "y": y, "item_id": sel["item_id"],
    })
    responses = [d["y"] for d in st.session_state.administered]

    # Warm-start Newton from the previous theta -- usually 1-2 iterations.
    st.session_state.theta = eng.update_theta(
        responses, st.session_state.administered,
        theta_init=st.session_state.theta,
    )
    st.session_state.se = eng.se_at(st.session_state.theta,
                                    st.session_state.administered)
    step = len(st.session_state.administered)
    st.session_state.history.append({
        "step":     step,
        "item_id":  sel["item_id"],
        "response": y,
        "theta":    st.session_state.theta,
        "se":       st.session_state.se,
    })

    # Stopping rule: SE < 0.30 OR n >= 20 OR bank exhausted.
    if (eng.should_stop(step, st.session_state.se)
            or len(st.session_state.used_ids) >= len(bank)):
        st.session_state.finished = True
        st.session_state.page = "results"
        st.session_state.current_item = None
        return

    # Otherwise pick the next item.
    select_next_item(bank)


# ---------------------------------------------------------------------------
# Page 1: Intro
# ---------------------------------------------------------------------------

def page_intro():
    st.title("Soccer / FIFA World Cup: Computerized Adaptive Test")
    st.markdown(
        """
        **DSAN 6550, Spring 2026. Final Project by Emily Zhao & Zhengyuan Wang.**

        This dashboard demonstrates a 2PL Computerized Adaptive Test
        for soccer and FIFA World Cup knowledge. After every answer,
        the system re-estimates the test-taker's ability theta_hat and
        picks the next question whose **item information**
        I_j(theta_hat) is highest at that ability. In other words, it
        chooses the question whose answer would tell us the most about
        the test-taker right now.

        ### How this demo works
        - The item bank holds **30 calibrated items**, with parameters
          a and b estimated by Role 1 from 500 simulated respondents.
        - The engine picks the next item by maximum information, then
          updates theta_hat with Newton-Raphson MLE after each answer.
        - The test stops when SE(theta_hat) drops below 0.30, or after
          20 items.

        ### Two modes (sidebar)
        - **Live mode**: the user answers each question.
        - **Demo mode**: replays a pre-recorded simulated respondent
          (low, medium, or high ability) so the class can see the
          system handle all three ability levels.
        """
    )
    st.divider()

    bank = load_item_bank()
    c1, c2, c3 = st.columns(3)
    c1.metric("Items in bank", len(bank))
    c2.metric("a range", f"{bank['a'].min():.2f} to {bank['a'].max():.2f}")
    c3.metric("b range", f"{bank['b'].min():.2f} to {bank['b'].max():.2f}")

    st.divider()
    if st.button("Start Test", type="primary", use_container_width=True):
        select_next_item(bank)
        st.session_state.page = "test"
        st.rerun()


# ---------------------------------------------------------------------------
# Page 2: Test
# ---------------------------------------------------------------------------

def page_test():
    bank = load_item_bank()
    if st.session_state.current_item is None:
        select_next_item(bank)

    sel = st.session_state.current_item
    item = sel["row"]
    step = len(st.session_state.administered) + 1

    # ----- Header: progress and current estimate
    st.title("Adaptive Test in Progress")
    c1, c2, c3 = st.columns(3)
    c1.metric("Question #", f"{step}  /  up to {eng.MAX_ITEMS}")
    c2.metric("Current theta_hat", f"{st.session_state.theta:+.3f}")
    se_disp = ("inf" if not np.isfinite(st.session_state.se)
               else f"{st.session_state.se:.3f}")
    c3.metric("SE(theta_hat)", se_disp,
              help="Test stops when SE < 0.30")

    st.divider()

    # ----- The item itself
    st.subheader(f"{sel['item_id']}.  {item['Question']}")
    st.caption(f"Tag: {item['Difficulty']}   "
               f"Category: {item['Category']}   "
               f"Calibrated  a={sel['a']:.2f},  b={sel['b']:.2f}")

    options = {"A": item["OptionA"], "B": item["OptionB"],
               "C": item["OptionC"], "D": item["OptionD"]}

    if st.session_state.mode == "Live":
        # Live mode: the user picks an option and clicks Submit.
        choice = st.radio(
            "Your answer:",
            list(options.keys()),
            format_func=lambda L: f"{L}.  {options[L]}",
            key=f"radio_{sel['item_id']}",
            index=None,
        )
        if st.button("Submit answer", type="primary",
                     disabled=(choice is None)):
            submit_answer(choice, bank)
            st.rerun()

    else:
        # Demo mode: pull the simulated respondent's recorded answer.
        rid = DEMO_RESPONDENTS[st.session_state.demo_choice]
        sim_row = load_simulated_responses().loc[
            load_simulated_responses()["RespondentID"] == rid].iloc[0]
        recorded_y = int(sim_row[sel["item_id"]])
        correct_letter = str(item["CorrectAnswer"]).upper()
        # The simulation only stored 0/1, not which distractor was chosen.
        # For display, show the correct letter when y=1, otherwise just say
        # "wrong option". This affects only the message; scoring uses y.
        if recorded_y == 1:
            sim_letter = correct_letter
            msg = f"{rid} answered {sim_letter} (correct)"
        else:
            sim_letter = next(L for L in "ABCD" if L != correct_letter)
            msg = f"{rid} answered a wrong option (recorded response = 0)"
        st.info(msg)

        c_a, c_b = st.columns(2)
        with c_a:
            if st.button("Submit (use simulated answer)", type="primary"):
                submit_answer(sim_letter, bank)
                st.rerun()
        with c_b:
            if st.button("Auto-run to completion"):
                # Just call submit_answer in a loop until the stopping rule fires.
                while not st.session_state.finished:
                    cur = st.session_state.current_item
                    if cur is None:
                        break
                    cur_correct = str(cur["row"]["CorrectAnswer"]).upper()
                    sub_letter = (cur_correct
                                  if int(sim_row[cur["item_id"]]) == 1
                                  else next(L for L in "ABCD"
                                            if L != cur_correct))
                    submit_answer(sub_letter, bank)
                st.rerun()

    st.divider()

    # ----- "Why this item" sentence (project requirement #7)
    df_info = sel["all_infos"]
    if len(df_info) >= 2:
        runner = df_info.iloc[1]
        runner_msg = (f" The next-best alternative was {runner['ItemID']} "
                      f"with I = {runner['info']:.3f}.")
    else:
        runner_msg = ""
    st.markdown(
        f"**Why this item?** Selected **{sel['item_id']}** "
        f"(a={sel['a']:.2f}, b={sel['b']:.2f}) because it has the "
        f"**highest information** I = {sel['info']:.3f} at the current "
        f"ability estimate theta_hat = {st.session_state.theta:+.2f}."
        f"{runner_msg}"
    )

    # ----- Four diagnostic panels (project requirement #7)
    st.subheader("Diagnostics")
    top_l, top_r = st.columns(2)
    with top_l:
        st.pyplot(plot_info_bar(sel, st.session_state.theta),
                  use_container_width=True)
    with top_r:
        st.pyplot(plot_trajectory(st.session_state.history),
                  use_container_width=True)
    bot_l, bot_r = st.columns(2)
    with bot_l:
        st.pyplot(plot_icc(sel["a"], sel["b"], st.session_state.theta,
                           sel["item_id"]), use_container_width=True)
    with bot_r:
        st.pyplot(plot_iic(sel["a"], sel["b"], st.session_state.theta,
                           sel["item_id"]), use_container_width=True)
    plt.close("all")  # don't leak figure objects across reruns


# ---------------------------------------------------------------------------
# Page 3: Results
# ---------------------------------------------------------------------------

def page_results():
    st.title("Test Complete: Results")

    if not st.session_state.history:
        st.warning("No items were administered. Restart and try again.")
        if st.button("Back to start"):
            reset_for_new_test()
            st.rerun()
        return

    final_theta = st.session_state.theta
    final_se    = st.session_state.se
    n_items     = len(st.session_state.history)
    n_correct   = sum(h["response"] for h in st.session_state.history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final theta_hat", f"{final_theta:+.3f}")
    c2.metric("Final SE", f"{final_se:.3f}")
    c3.metric("Items administered", n_items)
    c4.metric("Correct", f"{n_correct} / {n_items}")

    # Why we stopped
    if final_se < eng.SE_STOP:
        reason = f"SE fell below the {eng.SE_STOP:.2f} cutoff."
    elif n_items >= eng.MAX_ITEMS:
        reason = f"Reached the maximum length of {eng.MAX_ITEMS} items."
    else:
        reason = "Item bank was exhausted."
    st.info(f"**Why the test stopped:** {reason}")

    # In demo mode, show how close the CAT got to the recorded full-test value.
    if st.session_state.mode == "Demo":
        rid = DEMO_RESPONDENTS[st.session_state.demo_choice]
        recorded = float(load_thetas().loc[
            load_thetas()["RespondentID"] == rid, "theta_hat"].iloc[0])
        st.success(
            f"Simulated respondent {rid} had a recorded theta_hat from "
            f"the full 30-item test of {recorded:+.3f}. This adaptive run "
            f"estimated {final_theta:+.3f} after only {n_items} items "
            f"(absolute difference {abs(final_theta - recorded):.3f})."
        )

    st.divider()
    st.subheader("Trajectory of theta_hat")
    st.pyplot(plot_trajectory(st.session_state.history),
              use_container_width=True)

    st.subheader("Item-by-item history")
    hist_df = pd.DataFrame(st.session_state.history).rename(columns={
        "step":     "Step",
        "item_id":  "ItemID",
        "response": "Response (1=correct)",
        "theta":    "Theta_hat (after)",
        "se":       "SE (after)",
    })
    st.dataframe(hist_df.style.format({
        "Theta_hat (after)": "{:+.3f}",
        "SE (after)":        "{:.3f}",
    }), use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    hist_df.to_csv(csv_buf, index=False)
    st.download_button("Download history as CSV", data=csv_buf.getvalue(),
                       file_name="cat_history.csv", mime="text/csv")

    plt.close("all")
    st.divider()
    if st.button("Run another test", type="primary"):
        reset_for_new_test()
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar():
    st.sidebar.header("Mode")
    st.sidebar.radio(
        "How should answers be entered?",
        ["Live", "Demo"],
        index=0 if st.session_state.get("mode", "Live") == "Live" else 1,
        key="mode",
        help="Live: the user answers each question. Demo: replay a simulated respondent.",
    )
    if st.session_state.mode == "Demo":
        st.sidebar.selectbox(
            "Simulated respondent",
            list(DEMO_RESPONDENTS.keys()),
            index=list(DEMO_RESPONDENTS.keys()).index(
                st.session_state.get("demo_choice",
                                     list(DEMO_RESPONDENTS.keys())[1])),
            key="demo_choice",
        )

    st.sidebar.divider()
    st.sidebar.header("Stopping rule")
    st.sidebar.write(f"- SE(theta_hat) < {eng.SE_STOP}")
    st.sidebar.write(f"- OR {eng.MAX_ITEMS} items administered")

    st.sidebar.divider()
    if st.sidebar.button("Reset test"):
        reset_for_new_test()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Soccer CAT: DSAN 6550",
                       page_icon="O", layout="wide")
    ensure_state()
    sidebar()
    page = st.session_state.page
    if page == "intro":
        page_intro()
    elif page == "test":
        page_test()
    elif page == "results":
        page_results()


if __name__ == "__main__":
    main()
