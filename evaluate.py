"""
evaluate.py
============================
Evaluates the copilot on real
lab trajectory data from online_arm_trajectories.csv.

KEY DIFFERENCE FROM evaluate_method1_on_real_data.py
------------------------------------------------------
This script runs entirely outside the env loop.  It directly simulates
equation (13) from Lee et al.:

    cursor[t+1] = cursor[t] + real_decoder_vel[t] + copilot_vel[t]

where:
  - real_decoder_vel[t]  = (vx, vy) from the CSV at tick t  (already in
                           normalised [-1,1] space, radius = 432 px)
  - copilot_vel[t]       = copilot_output[t] * COPILOT_VEL
                           direct (vx, vy) scaled by 0.02 (Run6+)

SUCCESS METRIC
--------------
A trial is a success if the angle between the final cursor position vector
and the target direction vector is the smallest among all 8 targets — i.e.

    argmax_j  dot(cursor_final / |cursor_final|, target_dir_j)  ==  target_label

This exactly matches how your lab computes arm_prediction_label in the CSV
(verified at ~97% agreement on non-tie cases).

BASELINE
--------
The raw decoder baseline applies the same angle metric to the CSV cursor
positions as-is (no copilot). This reproduces the 46.5% reported in the CSV.

RUN
---
From arm-bci-copilot/ :

    python evaluate_method1_optionB.py

Adjust CSV_PATH and MODEL_PATH at the top if needed.
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "SJtools/copilot/Training Data/online_arm_trajectories.csv"
MODEL_PATH = "SJtools/copilot/runs/LAB_realData_run5/best_model.zip"

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX   = 432.0
COPILOT_VEL = 0.02   # copilot velocity scale (matches training)
N_HIST      = 5      # history length (from best_model.yaml: history [5, 20, pos])
HIST_INTV   = 20     # history interval (ticks)

# 8 target unit-direction vectors, indexed by label 0–7
# (from lab_dir8.yaml, normalised)
LABEL_TO_DIR = np.array([
    [-0.707,  0.707],   # 0  NW
    [ 0.000,  1.000],   # 1  N
    [ 0.707,  0.707],   # 2  NE
    [ 1.000,  0.000],   # 3  E
    [ 0.707, -0.707],   # 4  SE
    [ 0.000, -1.000],   # 5  S
    [-0.707, -0.707],   # 6  SW
    [-1.000,  0.000],   # 7  W
], dtype=np.float32)

LABEL_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

# Run6+: action space is direct (vx, vy) — no target sorting needed.


# ── angle-based prediction ────────────────────────────────────────────────────
def angle_pred(cursor: np.ndarray) -> int:
    """
    Return the label (0–7) whose target direction is most aligned with cursor.
    Returns -1 if cursor is at the origin (undefined direction).
    Matches how the lab computes arm_prediction_label.
    """
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))


# ── direct velocity copilot (Run6+) ─────────────────────────────────────────
def calc_copilot_vel(copilot_output: np.ndarray, cursor_pos: np.ndarray) -> np.ndarray:
    """
    Apply copilot action as direct (vx, vy) velocity.

    copilot_output: shape (2,), raw PPO output in [-1, 1]
    cursor_pos:     shape (2,), unused — kept for API compatibility
    Returns:        shape (2,), velocity in normalised space

    Run6+ uses direct velocity output instead of chargeTargets.
    The policy outputs (vx, vy) in [-1, 1]; we scale by COPILOT_VEL=0.02.
    This is position-independent: the same action always produces the same
    cursor displacement regardless of where the cursor is.
    """
    vx = float(np.clip(copilot_output[0], -1.0, 1.0))
    vy = float(np.clip(copilot_output[1], -1.0, 1.0))
    return np.array([vx, vy], dtype=np.float32) * COPILOT_VEL


# ── observation builder ───────────────────────────────────────────────────────
class HistoryQueue:
    """
    Reproduces CircularQueue([5, 20, 'pos'], obs_size=2) from the env.
    Stores cursor positions only (velReplaceSoftmax with 'pos' history).
    """
    def __init__(self):
        self.n = (N_HIST - 1) * HIST_INTV + 1  # 81
        self.data = np.zeros((self.n, 2), dtype=np.float32)
        self.i = 0

    def reset(self, last_pos: np.ndarray = None):
        if last_pos is not None:
            # historyReset='last': fill all history with the last cursor pos
            self.data[:] = last_pos
        else:
            self.data[:] = 0.0
        self.i = 0

    def add_get(self, pos: np.ndarray) -> np.ndarray:
        self.i = (self.i + 1) % self.n
        self.data[self.i] = pos
        selected = self.i - np.arange(N_HIST) * HIST_INTV
        return self.data[selected]   # shape (5, 2) → flatten → (10,)


def build_obs(cursor_pos: np.ndarray,
              vel: np.ndarray,
              hist_queue: HistoryQueue) -> np.ndarray:
    """
    Reproduce TaskCopilotObservation.get_env_obs with:
      velReplaceSoftmax=True → obs = [cursorPos(2), softmax_to_vel(2)]
      history=[5,20,'pos']   → append 4 past cursor positions (8 values)
    Total: 12 elements.

    vel here is softmax_to_vel(real_softmax), i.e. the (vx, vy) decoded
    from the normalised real cursor velocity fed as softmax to the model.
    """
    base_obs = np.concatenate([cursor_pos, vel]).astype(np.float32)   # (4,)

    # Add current pos to queue and get history
    hist = hist_queue.add_get(cursor_pos)   # (5, 2)
    # history_obs[2:] = past 4 positions (skip index 0 = current)
    past_obs = hist[1:].flatten()           # (8,)

    return np.concatenate([base_obs, past_obs]).astype(np.float32)   # (12,)


# ── velocity → softmax encoding ───────────────────────────────────────────────
def vel_to_softmax(vx: float, vy: float) -> np.ndarray:
    """
    Same encoding used during training: the real (vx,vy) is normalised to
    unit magnitude, then split into four non-negative channels:
      s[0]=left (-x), s[1]=right (+x), s[2]=up (+y), s[3]=down (-y), s[4]=still
    softmax_to_vel() recovers (vx, vy) = (s[1]-s[0], s[2]-s[3]).
    """
    speed = np.sqrt(vx ** 2 + vy ** 2)
    if speed > 1e-6:
        vx /= speed
        vy /= speed
    s = np.zeros(5, dtype=np.float32)
    s[1] = max(vx,  0.0)
    s[0] = max(-vx, 0.0)
    s[2] = max(vy,  0.0)
    s[3] = max(-vy, 0.0)
    return s

def softmax_to_vel(s: np.ndarray) -> np.ndarray:
    """Inverse of vel_to_softmax (used inside the env obs)."""
    vx = float(s[1] - s[0])
    vy = float(s[2] - s[3])
    return np.clip(np.array([vx, vy], dtype=np.float32), -1, 1)


# ── trial loader ─────────────────────────────────────────────────────────────
def load_trials(csv_path: str) -> list:
    """
    Load CSV and return list of trial dicts:
      subject_id, target_label, vel_seq (list of np.ndarray shape (2,))
    vel_seq[t] = (vx, vy) in normalised space for tick t.
    """
    df = pd.read_csv(csv_path)
    group_cols = ['subject_id', 'session_number', 'run_number',
                  'trial_number', 'inner_trial_number']
    trials = []
    for _, grp in df.groupby(group_cols):
        grp = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        label = int(grp['target_label'].iloc[0])
        if label not in range(8):
            continue

        cx = grp['cursor_pos_x'].values.astype(np.float64) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float64) / RADIUS_PX

        vx = np.diff(cx).astype(np.float32)
        vy = np.diff(cy).astype(np.float32)
        if len(vx) == 0:
            continue

        # Also store raw cursor positions for the baseline
        trials.append({
            'subject_id':   grp['subject_id'].iloc[0],
            'target_label': label,
            'vel_seq':      list(zip(vx, vy)),
            'cursor_seq':   list(zip(cx.astype(np.float32),
                                     cy.astype(np.float32))),
        })
    return trials


# ── single trial simulation ───────────────────────────────────────────────────
def simulate_copilot_trial(model, trial: dict,
                            hist_queue: HistoryQueue,
                            prev_cursor_end: np.ndarray) -> tuple:
    """
    Run one trial with copilot assistance.

    Cursor update per tick:
        cursor[t+1] = cursor[t] + real_vel[t] + copilot_vel[t]

    Returns (success: bool, final_cursor: np.ndarray)
    """
    vel_seq = trial['vel_seq']
    target_label = trial['target_label']

    # Reset history; use end position of previous trial (historyReset='last')
    hist_queue.reset(last_pos=prev_cursor_end)

    cursor = np.zeros(2, dtype=np.float32)
    episode_start = True
    _states = None

    for tick, (vx, vy) in enumerate(vel_seq):
        # Build observation
        encoded_vel = softmax_to_vel(vel_to_softmax(vx, vy))
        obs = build_obs(cursor, encoded_vel, hist_queue)

        # Get copilot action
        action, _states = model.predict(
            obs, state=_states, deterministic=True,
            episode_start=episode_start
        )
        episode_start = False

        # Compute real decoder contribution (raw normalised velocity)
        real_vel = np.array([vx, vy], dtype=np.float32)

        # Compute copilot contribution (chargeTargets)
        cop_vel = calc_copilot_vel(action, cursor)

        # Combined cursor update
        cursor = cursor + real_vel + cop_vel
        cursor = np.clip(cursor, -1.5, 1.5)   # soft clip; targets at r=1.0

    final_pred = angle_pred(cursor)
    success = (final_pred == target_label)
    return success, cursor


# ── main evaluation ───────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 65)
    print("Copilot Evaluation angle-based metric")
    print("=" * 65)
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {CSV_PATH}")
    print()

    # Load model
    print("Loading model...")
    model = PPO.load(MODEL_PATH)
    print("Done.\n")

    # Load trials
    print("Loading trials from CSV...")
    trials = load_trials(CSV_PATH)
    n_subj = len(set(t['subject_id'] for t in trials))
    print(f"Loaded {len(trials)} trials from {n_subj} subjects.\n")

    # ── Baseline: raw decoder (no copilot) ────────────────────────────────────
    baseline_correct = 0
    for t in trials:
        # Final tick cursor position
        cx, cy = t['cursor_seq'][-1]
        if angle_pred(np.array([cx, cy])) == t['target_label']:
            baseline_correct += 1
    baseline_acc = baseline_correct / len(trials)
    print(f"Raw decoder baseline (angle @ final tick): "
          f"{baseline_correct}/{len(trials)}  ({baseline_acc*100:.1f}%)")
    print()

    # ── Copilot evaluation ────────────────────────────────────────────────────
    hist_queue = HistoryQueue()
    results_by_subject = {}
    results_by_direction = {i: [0, 0] for i in range(8)}

    subjects = sorted(set(t['subject_id'] for t in trials))
    for subj in subjects:
        subj_trials = [t for t in trials if t['subject_id'] == subj]
        successes = 0
        prev_cursor_end = np.zeros(2, dtype=np.float32)

        for trial in subj_trials:
            ok, final_cursor = simulate_copilot_trial(
                model, trial, hist_queue, prev_cursor_end
            )
            prev_cursor_end = final_cursor
            successes += int(ok)
            lbl = trial['target_label']
            results_by_direction[lbl][0] += int(ok)
            results_by_direction[lbl][1] += 1

        acc = successes / len(subj_trials)
        results_by_subject[subj] = (successes, len(subj_trials), acc)
        print(f"  {subj}: {successes}/{len(subj_trials)}  ({acc*100:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_success = sum(r[0] for r in results_by_subject.values())
    total_trials  = sum(r[1] for r in results_by_subject.values())
    copilot_acc   = total_success / total_trials
    delta         = copilot_acc - baseline_acc
    sign          = '+' if delta >= 0 else ''

    print()
    print("=" * 65)
    print(f"OVERALL: {total_success}/{total_trials}  ({copilot_acc*100:.1f}%)")
    print(f"Raw decoder baseline :  {baseline_acc*100:.1f}%")
    print(f"Copilot (Method 1)   :  {copilot_acc*100:.1f}%")
    print(f"Improvement          :  {sign}{delta*100:.1f} pp")
    print("=" * 65)

    # ── Per-direction breakdown ───────────────────────────────────────────────
    print()
    print("Per-direction breakdown:")
    print(f"{'Dir':>4}  {'Copilot':>10}  {'Baseline':>10}")
    print("-" * 30)

    df = pd.read_csv(CSV_PATH)
    group_cols = ['subject_id', 'session_number', 'run_number',
                  'trial_number', 'inner_trial_number']
    last = df.groupby(group_cols).last().reset_index()
    for lbl in range(8):
        s, n = results_by_direction[lbl]
        cop_pct = s / n * 100
        base_subset = last[last['target_label'] == lbl]
        base_pct = (base_subset['arm_prediction_label'] == lbl).mean() * 100
        diff = cop_pct - base_pct
        sign_d = '+' if diff >= 0 else ''
        print(f"  {LABEL_NAMES[lbl]:>2} (label {lbl}): "
              f"{cop_pct:5.1f}%   {base_pct:5.1f}%   {sign_d}{diff:.1f} pp")

    print()


if __name__ == '__main__':
    run_evaluation()
