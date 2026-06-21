"""
real_data_wrapper.py
====================
Place in: bci_raspy/SJtools/copilot/real_data_wrapper.py

A Gym wrapper that replaces the parametric surrogate with real CSV trajectory
data during training.  SB3 sees a normal env; model.learn() works unchanged.

HOW IT WORKS
------------
The env reads self.softmax at the start of every step() call (env.py line 632).
After step() it overwrites self.softmax with whatever getSoftmax() produces.

This wrapper intercepts step() and reset():

  reset():
    1. Calls inner env.reset() — env picks the next target from its queue.
    2. Reads which target was chosen (env.taskGame.nextTarget).
    3. Samples a real CSV trial whose target_label matches.
    4. Stores the velocity sequence and resets tick counter.
    5. Sets env.softmax to tick 0 of the real sequence.

  step(action):
    1. Saves inner.result[0] as prev_cursor (grounding point for counterfactual).
    2. Injects current tick's real velocity as env.softmax.
    3. Calls inner env.step(action) — chargeTargets Coulomb mechanism runs.
    4. Computes grounded counterfactual reward from inner.result[0].
    5. Advances tick counter; injects next tick's velocity.

The inner env must be created with:
  - action=['chargeTargets']    (Coulomb force, same as Run5 — keeps trial
                                 termination working correctly)
  - velReplaceSoftmax=True      (obs encodes velocity direction, not raw softmax)
  - softmax_type='normal_target' (getSoftmax fallback; overwritten each tick)
  - center_out_back=True

REWARD — Grounded counterfactual (Run7)
----------------------------------------
Per tick:
  cursor_with    = inner.result[0]          (real BCI+copilot cursor from env)
  cursor_without = prev_cursor + raw_vel    (BCI only, grounded each tick)
  reward = (cos_with - cos_without) * 3.0

At terminal:
  reward += cos_with * 0.1

WHY GROUNDED COUNTERFACTUAL over Run5's delta_cos:
  Run5 used delta_cos = cos_now - cos_prev, which credits the BCI decoder's
  accuracy rather than isolating the copilot's contribution. Directions where
  the BCI is already accurate (N: 60.2%) produced more total reward than harder
  directions (NE: 41.6%), biasing the policy toward N-favoring strategies.

  The grounded counterfactual measures only what the copilot contributed THIS
  tick: "compared to where the cursor would be with BCI alone, did the copilot
  help?" Since both cursor_with and cursor_without start from the same
  prev_cursor, the difference isolates the copilot's per-tick contribution.

WHY CHARGETAR GETS (not VXY):
  Run7 tried VXY (action=['vx','vy','alpha'], setAlpha=0.3) but it broke trial
  termination: the env's hit_target check uses the internal VXY cursor which
  moves at cursorVel=0.015/tick, never reaching radius 1.0 in 16 ticks.
  Episodes timed out at ep_len=480 instead of completing normally (~264 ticks).
  chargeTargets keeps the env's trial termination working correctly.

  The state-reward mismatch from Run6 is fixed here by reading prev_cursor FROM
  inner.result[0] at the start of each step, so observations and rewards are
  always grounded to the same cursor.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from collections import defaultdict


# ── constants (must match env and CSV) ───────────────────────────────────────
RADIUS_PX = 432.0
N_STATE   = 5

TARGET_NAME_TO_LABEL = {
    'nw': 0, 'n': 1, 'ne': 2, 'e': 3,
    'se': 4, 's': 5, 'sw': 6, 'w': 7,
}

_LABEL_TO_DIR = np.array([
    [-0.707,  0.707],  # 0 NW
    [ 0.000,  1.000],  # 1 N
    [ 0.707,  0.707],  # 2 NE
    [ 1.000,  0.000],  # 3 E
    [ 0.707, -0.707],  # 4 SE
    [ 0.000, -1.000],  # 5 S
    [-0.707, -0.707],  # 6 SW
    [-1.000,  0.000],  # 7 W
], dtype=np.float32)


def _vel_to_softmax(vx: float, vy: float) -> np.ndarray:
    speed = np.sqrt(vx * vx + vy * vy)
    if speed > 1e-6:
        vx /= speed
        vy /= speed
    s = np.zeros(N_STATE, dtype=np.float32)
    s[1] = max( vx, 0.0)
    s[0] = max(-vx, 0.0)
    s[2] = max( vy, 0.0)
    s[3] = max(-vy, 0.0)
    return s


class RealDataWrapper(gym.Wrapper):
    """
    Wraps SJ4DirectionsEnv and injects real CSV velocities as env.softmax
    before each step, replacing the parametric surrogate entirely.

    Parameters
    ----------
    env : SJ4DirectionsEnv
        Must be created with velReplaceSoftmax=True and action=['chargeTargets'].
    csv_path : str
        Path to online_arm_trajectories.csv.
    rng_seed : int | None
        For reproducibility.  None = random.
    """

    def __init__(self, env, csv_path: str, rng_seed=None):
        super().__init__(env)
        self._rng = np.random.default_rng(rng_seed)
        self._traj_library = self._load_csv(csv_path)

        # Runtime state
        self._vel_seq      = []
        self._raw_vel_seq  = np.zeros((0, 2), dtype=np.float32)
        self._tick         = 0
        self._zero_softmax = np.zeros(N_STATE, dtype=np.float32)
        self._prev_cursor  = np.zeros(2, dtype=np.float32)

        # Per-trial reward state
        self._target_dir   = None

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str) -> dict:
        df = pd.read_csv(csv_path)
        group_cols = ['subject_id', 'session_number', 'run_number',
                      'trial_number', 'inner_trial_number']
        library = defaultdict(list)
        for _, grp in df.groupby(group_cols):
            grp  = grp.sort_values('timestamp_seconds')
            lbl  = int(grp['target_label'].iloc[0])
            cx   = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
            cy   = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
            vx   = np.diff(cx)
            vy   = np.diff(cy)
            if len(vx) == 0:
                continue
            library[lbl].append(np.stack([vx, vy], axis=1))  # (T, 2)
        counts = {k: len(v) for k, v in library.items()}
        print(f"[RealDataWrapper] Trajectories per label: {counts}")
        print(f"[RealDataWrapper] Total: {sum(counts.values())}")
        return dict(library)

    # ── trial sampling ────────────────────────────────────────────────────────

    def _sample_trial(self, target_label: int):
        """Returns (softmax_seq, raw_vel_seq)."""
        trajs = self._traj_library.get(target_label)
        if not trajs:
            all_trajs = [t for ts in self._traj_library.values() for t in ts]
            trajs = all_trajs
        idx = self._rng.integers(len(trajs))
        seq = trajs[idx]  # (T, 2)
        softmax_seq = [_vel_to_softmax(seq[i, 0], seq[i, 1]) for i in range(len(seq))]
        return softmax_seq, seq

    def _get_raw_vel_for_tick(self, tick: int) -> np.ndarray:
        if tick < len(self._raw_vel_seq):
            return self._raw_vel_seq[tick].astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_softmax_for_tick(self, tick: int) -> np.ndarray:
        if tick < len(self._vel_seq):
            return self._vel_seq[tick]
        return self._zero_softmax.copy()

    # ── target label resolution ───────────────────────────────────────────────

    def _get_inner_env(self):
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def _get_current_target_label(self) -> int:
        inner = self._get_inner_env()
        target_name = getattr(inner.taskGame, 'nextTarget', None)
        if target_name is None:
            target_name = getattr(inner.taskGame, 'target', None)
        lbl = TARGET_NAME_TO_LABEL.get(target_name, None)
        if lbl is None:
            lbl = int(self._rng.integers(8))
        return lbl

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        lbl = self._get_current_target_label()

        if lbl in range(8):
            self._target_dir = _LABEL_TO_DIR[lbl]
        else:
            self._target_dir = None

        self._vel_seq, self._raw_vel_seq = self._sample_trial(lbl)
        self._tick        = 0
        self._prev_cursor = np.zeros(2, dtype=np.float32)
        self._inject(0)

        return obs, info

    def step(self, action):
        # Save the real BCI+copilot cursor BEFORE this step.
        # inner.result[0] is the cursor position from the PREVIOUS tick —
        # the grounding point for this tick's counterfactual reward.
        inner = self._get_inner_env()
        self._prev_cursor = np.array(inner.result[0], dtype=np.float32)

        # Get raw BCI velocity for this tick (for counterfactual computation)
        raw_vel = self._get_raw_vel_for_tick(self._tick)

        # Inject current tick velocity before env reads self.softmax
        self._inject(self._tick)

        # Run env step — chargeTargets Coulomb mechanism applies the action
        obs, reward, done, truncated, info = self.env.step(action)

        # Grounded counterfactual reward
        reward = self._counterfactual_reward(info, raw_vel, done or truncated)

        self._tick += 1
        self._inject(self._tick)

        if done or truncated:
            self._tick = 0
            self._prev_cursor = np.zeros(2, dtype=np.float32)

        return obs, reward, done, truncated, info

    # ── reward ────────────────────────────────────────────────────────────────

    def _cos_to_target(self, cursor: np.ndarray) -> float:
        norm = np.linalg.norm(cursor)
        if norm < 1e-6:
            return 0.0
        return float(np.dot(cursor / norm, self._target_dir))

    def _counterfactual_reward(self, info: dict,
                                raw_vel: np.ndarray,
                                is_terminal: bool) -> float:
        """
        Grounded counterfactual reward (Run7).

        Per tick:
            cursor_with    = inner.result[0]          (BCI+copilot, from env)
            cursor_without = prev_cursor + raw_vel    (BCI only, grounded)
            reward = (cos_with - cos_without) * 3.0

        At terminal:
            reward += cos_with * 0.1

        cursor_without is grounded to the real cursor at the START of this tick
        (prev_cursor = inner.result[0] from the previous step). This measures
        only what the copilot contributed THIS tick, keeping the difference
        small and stable rather than accumulating over the trial.

        No state-reward mismatch: both cursor_with and cursor_without derive
        from inner.result[0], which is also what the policy observes.

        Center trials (target_dir is None) return 0.0.
        """
        if self._target_dir is None:
            return 0.0

        result = info.get('result', None)
        if result is None:
            return 0.0

        # Real BCI+copilot cursor after this step
        cursor_with = np.array(result[0], dtype=np.float32)

        # Counterfactual: BCI only, grounded to real prev cursor
        cursor_without = self._prev_cursor + raw_vel

        cos_with    = self._cos_to_target(cursor_with)
        cos_without = self._cos_to_target(cursor_without)

        reward = (cos_with - cos_without) * 3.0

        if is_terminal:
            reward += cos_with * 0.1

        return reward

    def _inject(self, tick: int):
        """Write the real softmax into env.softmax."""
        softmax = self._get_softmax_for_tick(tick)
        self._get_inner_env().softmax = softmax
