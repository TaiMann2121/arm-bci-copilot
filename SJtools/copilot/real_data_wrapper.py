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
    1. Sets env.softmax to the current real tick velocity.
       (The env will read this immediately at line 632.)
    2. Calls inner env.step(action).
    3. Advances tick counter; sets env.softmax to the NEXT tick velocity
       so it's ready for the observation that SB3 will read.

The inner env must be created with:
  - velReplaceSoftmax=True   (so obs encodes velocity, not raw softmax)
  - softmax_type='normal_target'  (getSoftmax fallback; overwritten each tick)
  - center_out_back=True
  - action=['chargeTargets']

REWARD
------
Pass reward_type='baseLinAngle.yaml'.  This rewards angular progress toward
the target at each tick, which is exactly your lab's classification metric.
It works correctly regardless of whether the cursor reaches radius 1.0.

USAGE IN train.py
-----------------
  from SJtools.copilot.real_data_wrapper import RealDataWrapper

  env = SJ4DirectionsEnv(
      velReplaceSoftmax=True,
      softmax_type='normal_target',
      reward_type='baseLinAngle.yaml',
      action=['chargeTargets'],
      action_param=['temperature', '1'],
      historyDim=[5, 20, 'pos'],
      historyReset='last',
      extra_targets_yaml='lab_dir8.yaml',
      center_out_back=True,
      holdtime=0.5,
      obs=[],
      CSvalue=0.75,   # unused at runtime, needed for env init
      stillCS=0.0,
  )
  env = RealDataWrapper(env, csv_path='SJtools/copilot/Training Data/online_arm_trajectories.csv')
  env = Monitor(env)

Then train exactly as before:
  model = PPO('MlpPolicy', env, ...)
  model.learn(total_timesteps=1_200_000, ...)

TRAINING COMMAND (equivalent to Method 1 run, but with real data)
-----------------------------------------------------------------
  python train.py \\
      -model PPO \\
      -timesteps 1200000 \\
      -softmax_type normal_target \\
      -reward_type baseLinAngle.yaml \\
      -action chargeTargets \\
      -action_param temperature 1 \\
      -history 5 20 pos \\
      -historyReset last \\
      -extra_targets_yaml lab_dir8.yaml \\
      -center_out_back \\
      -velReplaceSoftmax \\
      -CS 0.75 \\
      -stillCS 0.0 \\
      -holdtime 0.5 \\
      -lr 0.0003 \\
      -n_steps 2048 \\
      -batch_size 64 \\
      -log_interval 4 \\
      -no_wandb \\
      -fileName LAB_realData_run1 \\
      -use_real_data    ← new flag added to train.py (see below)
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

        # Runtime state
        self._vel_seq      = []
        self._tick         = 0
        self._zero_softmax = np.zeros(N_STATE, dtype=np.float32)

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
            library[lbl].append(np.stack([vx, vy], axis=1))  # (T, 2)

        counts = {k: len(v) for k, v in library.items()}
        print(f"[RealDataWrapper] Trajectories per label: {counts}")
        print(f"[RealDataWrapper] Total: {sum(counts.values())}")
        return dict(library)

    # ── trial sampling ────────────────────────────────────────────────────────

    def _sample_trial(self, target_label: int):
        """Sample a random real trajectory for the given label."""
        trajs = self._traj_library.get(target_label)
        if not trajs:
            # Fallback: sample from any label (should never happen with lab_dir8)
            all_trajs = [t for ts in self._traj_library.values() for t in ts]
            trajs = all_trajs
        idx = self._rng.integers(len(trajs))
        seq = trajs[idx]   # (T, 2)
        return [_vel_to_softmax(seq[i, 0], seq[i, 1]) for i in range(len(seq))]

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

        # Sample matching real trajectory
        self._vel_seq = self._sample_trial(lbl)
        self._tick    = 0
        self._inject(0)

        return obs, info

    def step(self, action):
        # Inject current tick before env reads self.softmax
        self._inject(self._tick)

        obs, reward, done, truncated, info = self.env.step(action)

        # Override reward with position-based angular improvement.
        # This replaces the noisy movement-direction reward from baseLinAngle.yaml
        # with a signal directly aligned with the arm_prediction_label metric:
        #   reward_t = cos(cursor_angle_to_target)_t - cos(cursor_angle_to_target)_{t-1}
        # This is symmetric across all 8 directions and has much lower variance
        # than per-tick movement-direction rewards.
        reward = self._angular_reward(info, done or truncated)

        self._tick += 1
        self._inject(self._tick)

        if done or truncated:
            self._tick = 0
            self._prev_cos = 0.0

        return obs, reward, done, truncated, info

    def _angular_reward(self, info: dict, is_terminal: bool) -> float:
        """
        Compute position-based angular reward.

        Per tick:   delta_cos = cos(cursor_to_target_angle)_now - cos(...)_prev
        At terminal: additional bonus = final cos score (scaled modestly)

        Terminal scale of 2.0 (was 10.0) prevents the policy from learning
        extreme chargeTargets outputs just to maximise the large terminal bonus,
        which caused action std to blow up from 1.0 → 4.3 over 1.2M steps.

        The cos score is dot(cursor_unit, target_unit_direction).
        At origin (cursor = 0), cos is undefined -> reward = 0.
        """
        if self._target_dir is None:
            return 0.0

        result = info.get('result', None)
        if result is None:
            return 0.0

        cursor_pos = result[0]
        cursor_norm = np.linalg.norm(cursor_pos)

        if cursor_norm < 1e-6:
            self._prev_cos = 0.0
            return 0.0

        current_cos = float(np.dot(cursor_pos / cursor_norm, self._target_dir))
        delta = current_cos - self._prev_cos
        self._prev_cos = current_cos

        if is_terminal:
            # Terminal bonus scaled to 2× (not 10×) to keep action std near 1.0
            return delta + current_cos * 2.0

        return delta * 5.0

    def _inject(self, tick: int):
        """Write the real softmax into env.softmax."""
        softmax = self._get_softmax_for_tick(tick)
        self._get_inner_env().softmax = softmax
