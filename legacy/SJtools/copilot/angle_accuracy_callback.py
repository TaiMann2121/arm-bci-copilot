"""
angle_accuracy_callback.py
===========================
Place in: bci_raspy/SJtools/copilot/angle_accuracy_callback.py

A custom SB3 callback that:
  1. Every eval_freq steps, runs the copilot model on a fixed held-out set
     of real CSV trials using the angle-based success metric
     (exactly matching arm_prediction_label from online_arm_trajectories.csv)
  2. Logs angle-accuracy and per-direction breakdown to TensorBoard
  3. Saves best_model.zip when angle-accuracy improves

This replaces EvalCallback's noisy mean_reward criterion with the metric
that actually matters: does the copilot improve angular classification?

The held-out eval set is separate from the training data — it uses a fixed
random seed and covers all 8 directions equally (N_EVAL_PER_DIR trials each).

Usage in train_method2.py:
    from SJtools.copilot.angle_accuracy_callback import AngleAccuracyCallback

    angle_cb = AngleAccuracyCallback(
        model_save_path = myFiles.best_model_save_path,
        csv_path        = args.csv_path,
        eval_freq       = 8192,          # evaluate every 8k steps
        n_eval_per_dir  = 30,            # 30 trials per direction = 240 total
        verbose         = 1,
    )
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX   = 432.0
N_STATE     = 5
N_HIST      = 5
HIST_INTV   = 20
COPILOT_VEL = 0.02
# EPS, K, POWER, TEMPERATURE removed — no longer needed (Run6+ uses direct velocity)

DIR_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

LABEL_TO_DIR = np.array([
    [-0.707,  0.707],
    [ 0.000,  1.000],
    [ 0.707,  0.707],
    [ 1.000,  0.000],
    [ 0.707, -0.707],
    [ 0.000, -1.000],
    [-0.707, -0.707],
    [-1.000,  0.000],
], dtype=np.float32)

# TARGETS_SORTED removed — no longer needed (Run6+ uses direct velocity output)


# ── helper functions (self-contained, no env needed) ──────────────────────────

def _vel_to_softmax(vx, vy):
    speed = np.sqrt(vx * vx + vy * vy)
    if speed > 1e-6:
        vx /= speed; vy /= speed
    s = np.zeros(N_STATE, dtype=np.float32)
    s[1] = max(vx, 0.0); s[0] = max(-vx, 0.0)
    s[2] = max(vy, 0.0); s[3] = max(-vy, 0.0)
    return s

def _softmax_to_vel(s):
    return np.clip(np.array([s[1]-s[0], s[2]-s[3]], dtype=np.float32), -1, 1)

def _angle_pred(cursor):
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))

def _calc_copilot_vel(copilot_output, cursor_pos):
    # Run6+: direct (vx, vy) output — position-independent.
    # cursor_pos kept for API compatibility but unused.
    vx = float(np.clip(copilot_output[0], -1.0, 1.0))
    vy = float(np.clip(copilot_output[1], -1.0, 1.0))
    return np.array([vx, vy], dtype=np.float32) * COPILOT_VEL


class _HistQueue:
    def __init__(self):
        self.n    = (N_HIST - 1) * HIST_INTV + 1
        self.data = np.zeros((self.n, 2), dtype=np.float32)
        self.i    = 0

    def reset(self, last_pos=None):
        self.data[:] = last_pos if last_pos is not None else 0.0
        self.i = 0

    def add_get(self, pos):
        self.i = (self.i + 1) % self.n
        self.data[self.i] = pos
        return self.data[self.i - np.arange(N_HIST) * HIST_INTV]


def _build_obs(cursor_pos, vel, hist_queue):
    base = np.concatenate([cursor_pos, vel]).astype(np.float32)
    hist = hist_queue.add_get(cursor_pos)
    return np.concatenate([base, hist[1:].flatten()]).astype(np.float32)


# ── eval trial runner ─────────────────────────────────────────────────────────

def _run_eval_trials(model, trials):
    """
    Run angle-based evaluation on a list of trial dicts.
    Each trial: {'target_label': int, 'vel_seq': list of (vx, vy)}
    Returns (overall_acc, per_dir_successes, per_dir_totals)
    """
    hist_queue = _HistQueue()
    prev_cursor = np.zeros(2, dtype=np.float32)
    per_dir = {i: [0, 0] for i in range(8)}   # [successes, total]

    for trial in trials:
        lbl      = trial['target_label']
        vel_seq  = trial['vel_seq']

        hist_queue.reset(last_pos=prev_cursor)
        cursor        = np.zeros(2, dtype=np.float32)
        episode_start = True
        _states       = None

        for vx, vy in vel_seq:
            encoded_vel = _softmax_to_vel(_vel_to_softmax(vx, vy))
            obs         = _build_obs(cursor, encoded_vel, hist_queue)

            action, _states = model.predict(
                obs, state=_states, deterministic=True,
                episode_start=episode_start
            )
            episode_start = False

            real_vel   = np.array([vx, vy], dtype=np.float32)
            cop_vel    = _calc_copilot_vel(action, cursor)
            cursor     = np.clip(cursor + real_vel + cop_vel, -1.5, 1.5)

        prev_cursor = cursor.copy()
        pred    = _angle_pred(cursor)
        success = int(pred == lbl)
        per_dir[lbl][0] += success
        per_dir[lbl][1] += 1

    total_s = sum(v[0] for v in per_dir.values())
    total_n = sum(v[1] for v in per_dir.values())
    overall = total_s / total_n if total_n > 0 else 0.0
    return overall, per_dir


# ── the callback ──────────────────────────────────────────────────────────────

class AngleAccuracyCallback(BaseCallback):
    """
    Evaluates angle-based accuracy on a held-out real-data set and saves
    the model whenever accuracy improves.

    Parameters
    ----------
    model_save_path : str
        Directory where best_model.zip will be saved.
    csv_path : str
        Path to online_arm_trajectories.csv.
    eval_freq : int
        Evaluate every this many training steps.
    n_eval_per_dir : int
        Number of held-out trials per direction (8 directions total).
        Default 30 → 240 total eval trials.
    rng_seed : int
        Seed for reproducible held-out set selection.
    verbose : int
    """

    def __init__(self, model_save_path, csv_path,
                 eval_freq=8192, n_eval_per_dir=150,
                 rng_seed=0, verbose=1):
        super().__init__(verbose=verbose)
        self.model_save_path = model_save_path
        self.eval_freq       = eval_freq
        self.n_eval_per_dir  = n_eval_per_dir
        self._best_acc       = -np.inf
        self._last_eval_step = 0
        self._rng            = np.random.default_rng(rng_seed)

        # Pre-load the full per-label trial pool (all trials, not a fixed subset)
        # At each eval call we sample n_eval_per_dir fresh trials
        # This prevents the model from overfitting to a fixed 240-trial eval set
        self._trial_pool = self._build_trial_pool(csv_path)
        total = sum(len(v) for v in self._trial_pool.values())
        print(f"[AngleAccuracyCallback] Trial pool: "
              f"{total} total trials, sampling {n_eval_per_dir} per direction each eval")

    def _build_trial_pool(self, csv_path):
        """Load ALL trials into a per-label pool for random sampling each eval."""
        df  = pd.read_csv(csv_path)
        group_cols = ['subject_id', 'session_number', 'run_number',
                      'trial_number', 'inner_trial_number']
        pool = {i: [] for i in range(8)}
        for _, grp in df.groupby(group_cols):
            grp = grp.sort_values('timestamp_seconds')
            lbl = int(grp['target_label'].iloc[0])
            if lbl not in range(8):
                continue
            cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
            cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
            vx = np.diff(cx); vy = np.diff(cy)
            if len(vx) == 0:
                continue
            pool[lbl].append(list(zip(vx.tolist(), vy.tolist())))
        return pool

    def _sample_eval_trials(self):
        """Sample n_eval_per_dir fresh trials per direction (no fixed seed)."""
        trials = []
        for lbl in range(8):
            idxs = self._rng.choice(
                len(self._trial_pool[lbl]),
                size=self.n_eval_per_dir,
                replace=False,
            )
            for i in idxs:
                trials.append({'target_label': lbl,
                                'vel_seq': self._trial_pool[lbl][int(i)]})
        return trials

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        # Sample a fresh set of trials each eval — prevents overfit to fixed set
        eval_trials = self._sample_eval_trials()

        # Run evaluation
        acc, per_dir = _run_eval_trials(self.model, eval_trials)

        # Log to TensorBoard
        self.logger.record('angle_eval/accuracy', acc)
        for lbl in range(8):
            s, n = per_dir[lbl]
            self.logger.record(f'angle_eval/{DIR_NAMES[lbl]}',
                               s / n if n > 0 else 0.0)

        if self.verbose >= 1:
            print(f"\n[AngleAccuracyCallback] step={self.num_timesteps:,}  "
                  f"angle_accuracy={acc*100:.1f}%  "
                  f"(best={self._best_acc*100:.1f}%)")
            for lbl in range(8):
                s, n = per_dir[lbl]
                print(f"  {DIR_NAMES[lbl]}: {s}/{n} ({s/n*100:.0f}%)")

        # Save if improved (model_save_path is None when running with -dev flag)
        if acc > self._best_acc:
            self._best_acc = acc
            if self.model_save_path is not None:
                save_path = os.path.join(self.model_save_path, 'best_model')
                self.model.save(save_path)
                if self.verbose >= 1:
                    print(f"  ✓ New best! Saved to {save_path}.zip")
            elif self.verbose >= 1:
                print(f"  ✓ New best! (save disabled in dev mode)")

        return True
