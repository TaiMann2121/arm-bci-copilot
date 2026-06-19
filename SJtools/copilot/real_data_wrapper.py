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
    1. Records the cursor position BEFORE env.step() for counterfactual reward.
    2. Sets env.softmax to the current real tick velocity.
    3. Calls inner env.step(action).
    4. Computes counterfactual reward (see REWARD below).
    5. Advances tick counter; sets env.softmax to the NEXT tick velocity.

The inner env must be created with:
  - velReplaceSoftmax=True   (so obs encodes velocity, not raw softmax)
  - softmax_type='normal_target'  (getSoftmax fallback; overwritten each tick)
  - center_out_back=True
  - action=['chargeTargets']      (env init only; action space overridden below)
  - The wrapper overrides the gym action_space to Box(2,) so SB3 outputs (vx,vy).
  - The wrapper maintains its own cursor independently of the env's VXY cursor,
    applying: cursor += real_vel + clip(action, -1, 1) * COPILOT_VEL

REWARD — Counterfactual baseline (Run6)
---------------------------------------
reward = [cos(cursor_with_copilot, target) - cos(cursor_without_copilot, target)] * 3.0

The counterfactual cursor is computed as:
  cursor_without = prev_cursor + real_vel   (BCI only, no copilot)

This decorrelates the copilot reward from the BCI decoder's accuracy on each
direction. Previously the model received more total reward on N trials simply
because the BCI baseline accuracy is higher for N (60.2%) than NE (41.6%),
creating a learning bias. The counterfactual reward measures only what the
copilot contributed on top of the BCI signal, giving equal learning opportunity
across all 8 directions regardless of BCI accuracy.

ACTION — Direct velocity output (Run6)
---------------------------------------
The copilot outputs (vx, vy) in [-1, 1], scaled by COPILOT_VEL=0.02.
This replaces the chargeTargets Coulomb mechanism which was position-dependent:
the same action produced different physical forces depending on cursor position,
creating a feedback loop that amplified directional drift. Direct velocity output
is position-independent: the same action always produces the same cursor
displacement, giving the policy a clean gradient.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict


# ── constants (must match env and CSV) ───────────────────────────────────────
RADIUS_PX   = 432.0
N_STATE     = 5
COPILOT_VEL = 0.02  # copilot velocity scale (matches evaluate.py)

TARGET_NAME_TO_LABEL = {
    'nw': 0, 'n': 1, 'ne': 2, 'e': 3,
    'se': 4, 's': 5, 'sw': 6, 'w': 7,
}

# Unit direction vectors for each label — same convention as arm_prediction_label
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
    """
    Encode a (vx, vy) velocity as the 5-channel softmax the env expects.
    Normalises to unit magnitude first (matches Method 1 training scale).
    Zero-velocity ticks pass through as all-zeros.
    """
    speed = np.sqrt(vx * vx + vy * vy)
    if speed > 1e-6:
        vx /= speed
        vy /= speed
    s = np.zeros(N_STATE, dtype=np.float32)
    s[1] = max( vx, 0.0)   # right
    s[0] = max(-vx, 0.0)   # left
    s[2] = max( vy, 0.0)   # up
    s[3] = max(-vy, 0.0)   # down
    # s[4] (still) stays 0 — real data doesn't distinguish still explicitly
    return s


class RealDataWrapper(gym.Wrapper):
    """
    Wraps SJ4DirectionsEnv and injects real CSV velocities as env.softmax
    before each step, replacing the parametric surrogate entirely.

    Parameters
    ----------
    env : SJ4DirectionsEnv (unwrapped or Monitor-wrapped)
        Must be created with velReplaceSoftmax=True.
    csv_path : str
        Path to online_arm_trajectories.csv.
    rng_seed : int | None
        For reproducibility.  None = random.
    """

    def __init__(self, env, csv_path: str, rng_seed=None):
        super().__init__(env)
        self._rng = np.random.default_rng(rng_seed)
        self._traj_library = self._load_csv(csv_path)

        # Override action space: SB3 should output (vx, vy) shape (2,)
        # The env's chargeTargets action space is shape (9,) which we don't want.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Runtime state
        self._vel_seq      = []          # softmax sequences for env injection
        self._raw_vel_seq  = np.zeros((0, 2), dtype=np.float32)  # raw (vx,vy)
        self._tick         = 0
        self._zero_softmax = np.zeros(N_STATE, dtype=np.float32)
        self._prev_cursor  = np.zeros(2, dtype=np.float32)  # for counterfactual

        # Independent cursor tracked by wrapper (separate from env's VXY cursor)
        # Rewards are computed from this cursor, not inner.result[0].
        self._cursor         = np.zeros(2, dtype=np.float32)

        # Per-trial angular reward state
        self._prev_cos     = 0.0   # cos(angle) at previous tick
        self._target_dir   = None  # unit direction to current target (from origin)

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str) -> dict:
        """
        Returns dict: label (int) -> list of velocity sequences (np.ndarray, shape (T,2))
        """
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
            library[lbl].append(np.stack([vx, vy], axis=1))  # (T, 2) raw (vx,vy)

        counts = {k: len(v) for k, v in library.items()}
        print(f"[RealDataWrapper] Trajectories per label: {counts}")
        print(f"[RealDataWrapper] Total: {sum(counts.values())}")
        return dict(library)

    # ── trial sampling ────────────────────────────────────────────────────────

    def _sample_trial(self, target_label: int):
        """Sample a random real trajectory for the given label.

        Returns (softmax_seq, raw_vel_seq):
          softmax_seq : list of np.ndarray(5,) — encoded for env.softmax injection
          raw_vel_seq : np.ndarray (T, 2)      — raw (vx, vy) for counterfactual reward
        """
        trajs = self._traj_library.get(target_label)
        if not trajs:
            # Fallback: sample from any label (should never happen with lab_dir8)
            all_trajs = [t for ts in self._traj_library.values() for t in ts]
            trajs = all_trajs
        idx = self._rng.integers(len(trajs))
        seq = trajs[idx]   # (T, 2) raw (vx, vy)
        softmax_seq = [_vel_to_softmax(seq[i, 0], seq[i, 1]) for i in range(len(seq))]
        return softmax_seq, seq   # seq is already np.ndarray (T, 2)

    def _get_softmax_for_tick(self, tick: int) -> np.ndarray:
        if tick < len(self._vel_seq):
            return self._vel_seq[tick]
        return self._zero_softmax.copy()

    # ── target label resolution ───────────────────────────────────────────────

    def _get_inner_env(self):
        """Unwrap to the SJ4DirectionsEnv."""
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def _get_current_target_label(self) -> int:
        """
        Read the env's chosen target name and convert to CSV label (0-7).
        Falls back to random if target not in the 8-direction set
        (e.g. center target during center-out-back inward phase).
        """
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

        # Store target unit direction for angular reward computation
        if lbl in range(8):
            self._target_dir = _LABEL_TO_DIR[lbl]
        else:
            self._target_dir = None

        # Reset angular reward state
        self._prev_cos = 0.0

        # Sample matching real trajectory (returns softmax seq + raw vel seq)
        self._vel_seq, self._raw_vel_seq = self._sample_trial(lbl)
        self._tick        = 0
        self._prev_cursor = np.zeros(2, dtype=np.float32)
        self._cursor      = np.zeros(2, dtype=np.float32)
        self._inject(0)

        return obs, info

    def step(self, action):
        # Save wrapper cursor BEFORE update (for counterfactual: BCI-only position)
        self._prev_cursor = self._cursor.copy()

        # Get raw BCI velocity for this tick
        raw_vel = self._get_raw_vel_for_tick(self._tick)

        # Inject current tick before env reads self.softmax
        self._inject(self._tick)

        # Pass a dummy action to the env (env uses chargeTargets for init only;
        # its internal cursor is not used for rewards — we track our own).
        # We pass zeros shaped for chargeTargets (9,) so the env doesn't crash.
        dummy_action = np.zeros(9, dtype=np.float32)
        obs, _env_reward, done, truncated, info = self.env.step(dummy_action)

        # Update wrapper's independent cursor:
        #   cursor += real_vel + clip(cop_dir, -1, 1) * COPILOT_VEL
        cop_dir = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._cursor = self._cursor + raw_vel + cop_dir * COPILOT_VEL
        self._cursor = np.clip(self._cursor, -1.5, 1.5)

        # Counterfactual reward based on wrapper's cursor, not env's cursor
        reward = self._counterfactual_reward(raw_vel, done or truncated)

        self._tick += 1
        self._inject(self._tick)

        if done or truncated:
            self._tick = 0
            self._prev_cos = 0.0
            self._prev_cursor = np.zeros(2, dtype=np.float32)
            self._cursor      = np.zeros(2, dtype=np.float32)

        return obs, reward, done, truncated, info

    def _get_raw_vel_for_tick(self, tick: int) -> np.ndarray:
        """Return raw (vx, vy) for the given tick, or zeros if past end."""
        if tick < len(self._raw_vel_seq):
            return self._raw_vel_seq[tick].astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def _cos_to_target(self, cursor: np.ndarray) -> float:
        """cos(angle between cursor position vector and target direction)."""
        norm = np.linalg.norm(cursor)
        if norm < 1e-6:
            return 0.0
        return float(np.dot(cursor / norm, self._target_dir))

    def _counterfactual_reward(self, raw_vel: np.ndarray,
                                is_terminal: bool) -> float:
        """
        Counterfactual reward: measures only the copilot's angular contribution.

        Per tick:
            cursor_with    = self._cursor              (BCI + copilot, wrapper-tracked)
            cursor_without = self._prev_cursor + raw_vel  (BCI only, counterfactual)
            reward = (cos_with - cos_without) * 3.0

        At terminal:
            reward += cos_with * 0.1                  (small outcome anchor)

        Uses the wrapper's independently tracked cursor (self._cursor), NOT the
        env's internal cursor (inner.result[0]). The env uses chargeTargets
        internally for initialization, but its cursor position is not meaningful
        for our task. The wrapper applies: cursor += real_vel + cop_dir * 0.02.

        WHY COUNTERFACTUAL:
        Decorrelates the copilot reward from the BCI decoder's per-direction
        accuracy. Previously N trials generated more reward than NE simply
        because BCI baseline is higher for N (60.2%) vs NE (41.6%), biasing
        the policy. The counterfactual measures only what the copilot added.

        Center trials (target_dir is None) return 0.0 — no meaningful target.
        """
        if self._target_dir is None:
            return 0.0

        # Cursor with copilot (wrapper-tracked, already updated in step())
        cursor_with = self._cursor

        # Counterfactual: BCI only, no copilot
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
