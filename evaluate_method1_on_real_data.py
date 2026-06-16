"""
evaluate_method1_on_real_data.py
=================================
Evaluates the Method 1 copilot (LAB_CS0.75_verify2/best_model.zip) against
real subject trajectories from online_arm_trajectories.csv.

For each trial in the CSV, the real cursor velocity sequence is replayed as
the decoder softmax input (via velReplaceSoftmax encoding). The trained
copilot acts on top of this signal, and we measure whether the cursor
successfully reaches the target.

This directly answers: "Does the copilot improve performance on our lab data?"

Bugs fixed in this version
--------------------------
Bug 1 — Wrong target (root cause of 0% success):
  env.reset() pops targets randomly from an internal shuffled queue.
  The trial that runs after reset() uses whatever target the env happens to pick,
  NOT the target in the CSV trial being replayed. Real softmax pointing East fed
  into an env that expects North → always miss.
  Fix: set env.taskGame.nextTargetBucket = [target_name] before env.reset() so
  the env is forced to use the correct target for that trial.

Bug 3 — Cursor not reset to center between trials:
  center_out_back sets resetCursorPos=False; cursor resets only happen when a
  real center trial times out inside the env (which sets resetNextCursorPos=True).
  The eval script bypasses the center trial entirely, so the cursor is never
  returned to [0,0] between trials. Every trial after the first starts with the
  cursor wherever the previous trial left it — far from center and often pointing
  in the wrong direction relative to the new target.
  Fix: after env.reset() returns, manually set cursorPos to [0,0] and clear any
  pending resetNextCursorPos flag.

Bug 2 — Velocity magnitude mismatch:
  The Method 1 model was trained with CS=0.75 surrogate velocities at unit
  magnitude (~1.0 after softmax_to_vel). Real lab data velocities are ~12x
  smaller (mean ~0.04 normalized). The copilot sees a near-frozen decoder.
  Fix: normalize each (vx, vy) tick to unit magnitude before encoding. Preserves
  directional signal (82.9% correct-direction alignment in real data, better than
  the 75% CS=0.75 training condition). Zero-velocity ticks stay zero.

Usage:
    python evaluate_method1_on_real_data.py

Adjust CSV_PATH and MODEL_PATH below if needed.
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# ── paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = "SJtools/copilot/Training Data/online_arm_trajectories.csv"
MODEL_PATH = "SJtools/copilot/runs/lab_training/LAB_CS0.75_verify2/best_model.zip"

# ── environment import ────────────────────────────────────────────────────────
from SJtools.copilot.env import SJ4DirectionsEnv, init_lab_surrogate

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX   = 432.0
N_STATE     = 5          # softmax dim used by env
TICK_LENGTH = 0.125      # seconds per tick at 8 Hz
TARGET_SIZE = 0.2        # normalized target diameter

# Target label → normalized (x, y) position
LABEL_TO_POS = {
    0: np.array([-0.707,  0.707]),  # NW
    1: np.array([ 0.000,  1.000]),  # N
    2: np.array([ 0.707,  0.707]),  # NE
    3: np.array([ 1.000,  0.000]),  # E
    4: np.array([ 0.707, -0.707]),  # SE
    5: np.array([ 0.000, -1.000]),  # S
    6: np.array([-0.707, -0.707]),  # SW
    7: np.array([-1.000,  0.000]),  # W
}

# Target label → env target name (must match lab_dir8.yaml target names)
LABEL_TO_NAME = {
    0: 'nw', 1: 'n', 2: 'ne', 3: 'e',
    4: 'se', 5: 's', 6: 'sw', 7: 'w',
}


def vel_to_softmax(vx: float, vy: float) -> np.ndarray:
    """
    Convert a (vx, vy) cursor velocity to the 5-element softmax format
    used by the environment with velReplaceSoftmax.

    softmax[0] = left  (-x)
    softmax[1] = right (+x)
    softmax[2] = up    (+y)
    softmax[3] = down  (-y)
    softmax[4] = still

    Normalization: the (vx, vy) vector is normalized to unit magnitude before
    encoding. This matches the speed scale of the CS=0.75 parametric surrogate
    used during training, while preserving the directional information in the
    real data. Zero-velocity ticks (decoder onset delay) pass through as zero.
    """
    speed = np.sqrt(vx ** 2 + vy ** 2)
    if speed > 1e-6:
        vx = vx / speed
        vy = vy / speed

    s = np.zeros(N_STATE, dtype=np.float32)
    if vx > 0:
        s[1] = vx
    else:
        s[0] = -vx
    if vy > 0:
        s[2] = vy
    else:
        s[3] = -vy
    return s


def load_trials(csv_path: str):
    """
    Load CSV and return a list of trial dicts, each containing:
      - subject_id, target_label, target_pos
      - softmax_seq: list of 5-element softmax arrays (one per tick)
    """
    df = pd.read_csv(csv_path)
    group_cols = [
        'subject_id', 'session_number', 'run_number',
        'trial_number', 'inner_trial_number',
    ]

    trials = []
    for keys, grp in df.groupby(group_cols):
        grp = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        label = int(grp['target_label'].iloc[0])
        if label not in LABEL_TO_POS:
            continue

        # Normalize cursor positions
        cx = grp['cursor_pos_x'].values.astype(np.float64) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float64) / RADIUS_PX

        # Compute per-tick velocities
        vx = np.diff(cx)
        vy = np.diff(cy)

        if len(vx) == 0:
            continue

        # Convert to softmax sequences
        softmax_seq = [vel_to_softmax(vx[i], vy[i]) for i in range(len(vx))]

        trials.append({
            'subject_id':   grp['subject_id'].iloc[0],
            'target_label': label,
            'target_pos':   LABEL_TO_POS[label],
            'softmax_seq':  softmax_seq,
        })

    return trials


def simulate_trial(model, trial, env):
    softmax_seq = trial['softmax_seq']
    target_name = LABEL_TO_NAME[trial['target_label']]

    # ── Bug 1 fix: force the correct target ──────────────────────────────────
    # env.reset() pops the next target from taskGame.nextTargetBucket.
    # Without this, the env picks a random target and the real softmax drives the
    # cursor toward the wrong goal every time.
    env.taskGame.nextTargetBucket = [target_name]

    obs, _ = env.reset()

    # ── Bug 3 fix: reset cursor to center ────────────────────────────────────
    # center_out_back sets resetCursorPos=False. The cursor only resets to [0,0]
    # when a real center trial times out — which never happens in this script.
    # Without this, every trial after the first starts with the cursor wherever
    # the previous trial left it, making success essentially impossible.
    env.taskGame.cursorPos = np.zeros(2)
    env.taskGame.secondCursorPos = np.zeros(2)
    env.taskGame.resetNextCursorPos = False  # clear any pending auto-reset

    # Defensive: confirm the env has the right target
    if env.taskGame.nextTarget != target_name:
        env.taskGame.nextTarget = target_name

    episode_start = True
    _states = None
    success = False

    for tick in range(64):
        # Inject real softmax before step reads it (Bug 2 fix: already normalized)
        real_softmax = softmax_seq[tick] if tick < len(softmax_seq) \
                       else np.zeros(N_STATE, dtype=np.float32)
        env.softmax = real_softmax

        action, _states = model.predict(obs, state=_states,
                                        deterministic=True,
                                        episode_start=episode_start)
        episode_start = False

        obs, reward, done, truncated, info = env.step(action)

        # Override whatever getSoftmax wrote — next step must see real data
        next_softmax = softmax_seq[tick + 1] if tick + 1 < len(softmax_seq) \
                       else np.zeros(N_STATE, dtype=np.float32)
        env.softmax = next_softmax

        # Check result
        result = info.get('result')
        if result is not None and len(result) > 4:
            gs = result[4]
            if gs == 'H':
                success = True
                break
            elif gs.isupper():
                break

        if done:
            break

    return success


def run_evaluation():
    print("=" * 60)
    print("Method 1 Copilot Evaluation on Real Lab Data")
    print("=" * 60)
    print(f"Model:  {MODEL_PATH}")
    print(f"Data:   {CSV_PATH}")
    print()

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded.")
    print()

    # ── load trials ───────────────────────────────────────────────────────────
    print("Loading CSV trials...")
    trials = load_trials(CSV_PATH)
    print(f"Loaded {len(trials)} trials from {len(set(t['subject_id'] for t in trials))} subjects.")
    print()

    # ── baseline: raw decoder accuracy from CSV ───────────────────────────────
    df = pd.read_csv(CSV_PATH)
    group_cols = ['subject_id','session_number','run_number',
                  'trial_number','inner_trial_number']
    last = df.groupby(group_cols).last().reset_index()
    raw_acc = (last['arm_prediction_label'] == last['target_label']).mean()
    print(f"Raw decoder baseline (final-tick accuracy): {raw_acc:.3f} ({raw_acc*100:.1f}%)")
    print()

    # ── create evaluation environment ─────────────────────────────────────────
    init_lab_surrogate(CSV_PATH)
    env = SJ4DirectionsEnv(
        isEval=True,
        softmax_type='normal_target',
        reward_type='baseLinDist.yaml',
        holdtime=0.5,
        center_out_back=True,
        velReplaceSoftmax=True,
        action=['chargeTargets'],           # ← was ['vx', 'vy', 'alpha']
        action_param=['temperature', '1'],  # ← was missing entirely
        historyDim=[5, 20, 'pos'],
        historyReset='last',
        extra_targets_yaml='lab_dir8.yaml',
        CSvalue=0.75,
        stillCS=0.0,
        obs=[],
    )

    # ── evaluate per subject ──────────────────────────────────────────────────
    results_by_subject = {}
    results_by_direction = {i: [0, 0] for i in range(8)}  # [successes, total]
    all_results = []

    subjects = sorted(set(t['subject_id'] for t in trials))
    for subj in subjects:
        subj_trials = [t for t in trials if t['subject_id'] == subj]
        successes = 0

        for trial in subj_trials:
            ok = simulate_trial(model, trial, env)
            successes += int(ok)
            all_results.append(ok)
            lbl = trial['target_label']
            results_by_direction[lbl][0] += int(ok)
            results_by_direction[lbl][1] += 1

        acc = successes / len(subj_trials)
        results_by_subject[subj] = (successes, len(subj_trials), acc)
        print(f"  {subj}: {successes}/{len(subj_trials)}  ({acc*100:.1f}%)")

    print()
    total_success = sum(r[0] for r in results_by_subject.values())
    total_trials  = sum(r[1] for r in results_by_subject.values())
    overall_acc   = total_success / total_trials

    print("=" * 60)
    print(f"OVERALL: {total_success}/{total_trials}  ({overall_acc*100:.1f}%)")
    print(f"Raw decoder baseline:   {raw_acc*100:.1f}%")
    print(f"Copilot (Method 1):     {overall_acc*100:.1f}%")
    delta = overall_acc - raw_acc
    sign  = '+' if delta >= 0 else ''
    print(f"Improvement:            {sign}{delta*100:.1f} pp")
    print("=" * 60)

    # ── per-direction breakdown ───────────────────────────────────────────────
    DIR_NAMES = {0:'NW', 1:'N', 2:'NE', 3:'E', 4:'SE', 5:'S', 6:'SW', 7:'W'}
    print()
    print("Per-direction success rate:")
    for lbl in range(8):
        s, n = results_by_direction[lbl]
        print(f"  {DIR_NAMES[lbl]:>2} (label {lbl}): {s}/{n}  ({s/n*100:.1f}%)")
    print()

    env.close()


if __name__ == '__main__':
    run_evaluation()
