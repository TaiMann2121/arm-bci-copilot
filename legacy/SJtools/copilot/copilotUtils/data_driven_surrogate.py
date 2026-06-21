"""
data_driven_surrogate.py
========================
bci_raspy/SJtools/copilot/copilotUtils/

Replaces the parametric surrogate policy in env.py with one built
directly from online_arm_trajectories.csv data.

Usage in env.py (see lab_env_patch.py for the full integration):

    from SJtools.copilot.copilotUtils.data_driven_surrogate import DataDrivenSurrogate

    surrogate = DataDrivenSurrogate('path/to/online_arm_trajectories.csv')
    env = SJ4DirectionsEnv(..., getVelocity=surrogate.make_get_velocity_fn(env_ref))

Architecture
------------
At each trial reset, the surrogate randomly draws one real trajectory from the
CSV that matches the current target direction (0-7).  It then replays the
normalised (vx, vy) velocity samples from that trajectory.  At the end of the
trajectory (trials are ~17 samples at 8 Hz), the surrogate loops or holds zero.

Coordinate convention
---------------------
The SJ_4_directions constructor operates in normalised [-1, 1] space.
CSV cursor_pos_{x,y} are in pixel space with radius = 432 px.
Normalisation: v_norm = v_pixels / 432.

The 8 arm targets map to normalised unit-circle positions:
    label 0 NW  (-0.707,  0.707)
    label 1 N   ( 0.000,  1.000)
    label 2 NE  ( 0.707,  0.707)
    label 3 E   ( 1.000,  0.000)
    label 4 SE  ( 0.707, -0.707)
    label 5 S   ( 0.000, -1.000)
    label 6 SW  (-0.707, -0.707)
    label 7 W   (-1.000,  0.000)

which correspond to the dir-8-close.yaml target names used in run.sh.

Calibrated parameters 
-------------------------------------------------
CS  (velocity correct-direction fraction)  : 0.75   (paper: 0.70)
Angular noise sigma                        : 0.761 rad  (paper: π/3 ≈ 1.047)
Additive Gaussian Sigma                    : 0.113·I    (paper: 0.03·I)
Onset delay (first directed movement)      : median 0.625 s  (paper: U[300,600] ms)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Target label → dir-8-close.yaml target name mapping
# The env cycles through these names when running center-out-back.
# ---------------------------------------------------------------------------
LABEL_TO_NAME = {
    0: 'nw',
    1: 'n',
    2: 'ne',
    3: 'e',
    4: 'se',
    5: 's',
    6: 'sw',
    7: 'w',
}

# Normalization constant
RADIUS_PX = 432.0


class DataDrivenSurrogate:
    """
    Builds a per-target library of real velocity trajectories 
    CSV and replays them during RL training.

    Parameters
    ----------
    csv_path : str | Path
        Path to online_arm_trajectories.csv.
    radius_px : float
        Pixel radius of the target circle (default 432).
    add_noise_sigma : float
        Optional scalar σ for extra i.i.d. Gaussian noise added on top of
        the replayed velocity at each step.  Set to 0 to replay raw data.
        Default 0.0 (no extra noise; the real data already contains noise).
    rng_seed : int | None
        For reproducibility.
    """

    def __init__(
    self,
    csv_path,
    radius_px: float = RADIUS_PX,
    add_noise_sigma: float = 0.0,
    rng_seed=None,
    ):
        self.radius = radius_px
        self.add_noise_sigma = add_noise_sigma
        self.rng = np.random.default_rng(rng_seed)

        # Build trajectory library
        self._trajs: dict[int, list[np.ndarray]] = {i: [] for i in range(8)}
        self._load(csv_path)

        # Runtime state (set by reset())
        self._current_seq = None  # shape (T, 2)
        self._step: int = 0
        self._current_label = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, csv_path) -> None:
        """Read CSV and build per-target velocity trajectory library."""
        df = pd.read_csv(csv_path)

        group_cols = [
            'subject_id', 'session_number', 'run_number',
            'trial_number', 'inner_trial_number',
        ]

        for keys, grp in df.groupby(group_cols):
            grp = grp.sort_values('timestamp_seconds')
            label = int(grp['target_label'].iloc[0])
            if label not in self._trajs:
                continue  # safety guard

            cx = grp['cursor_pos_x'].values.astype(np.float64) / self.radius
            cy = grp['cursor_pos_y'].values.astype(np.float64) / self.radius

            # NOTE: the env y-axis convention:
            #   In the constructor, up = positive y.
            #   In the CSV, cursor_pos_y appears to follow the same sign convention
            #   (target label 1 = N has target_pos_y = +432), so no flip needed.
            vx = np.diff(cx)
            vy = np.diff(cy)

            if len(vx) == 0:
                continue

            seq = np.stack([vx, vy], axis=1)  # shape (T, 2)
            self._trajs[label].append(seq)

        counts = {k: len(v) for k, v in self._trajs.items()}
        print(
            f"[DataDrivenSurrogate] Loaded trajectories per target: {counts}"
        )
        total = sum(counts.values())
        print(f"[DataDrivenSurrogate] Total trajectories: {total}")

    # ------------------------------------------------------------------
    # Runtime API used by env.py
    # ------------------------------------------------------------------

    def reset(self, target_label: int) -> None:
        """
        Call at the start of each new trial.

        Parameters
        ----------
        target_label : int
            The arm target label (0–7) for the upcoming trial.
        """
        self._current_label = target_label
        trajs = self._trajs.get(target_label, [])

        if not trajs:
            # Fallback: use all trajectories
            all_trajs = [t for ts in self._trajs.values() for t in ts]
            trajs = all_trajs

        idx = self.rng.integers(len(trajs))
        self._current_seq = trajs[idx]  # shape (T, 2)
        self._step = 0

    def get_velocity(self):
        if self._current_seq is None:
            all_trajs = [t for ts in self._trajs.values() for t in ts]
            self._current_seq = all_trajs[self.rng.integers(len(all_trajs))]
            self._step = 0

        if self._step >= len(self._current_seq):
            # End of trajectory: hold zero or loop
            return np.zeros(2)
        
        vel = self._current_seq[self._step].copy()
        self._step += 1

        if self.add_noise_sigma > 0:
            vel += self.rng.normal(0, self.add_noise_sigma, size=2)

        return vel

    # ------------------------------------------------------------------
    # Convenience: factory for getVelocity hook in env.py
    # ------------------------------------------------------------------

    def make_get_velocity_fn(self):
        """
        Returns a closure compatible with SJ4DirectionsEnv's getVelocity hook.

        The closure signature matches what env.py expects:
            getVelocity(env, arg) -> np.ndarray  (vx, vy)

        where arg = [cursorPos, targetPos, targetSize, still_state, game_state, detail]
        """
        surrogate = self

        def get_velocity_fn(env, arg):
            # arg[4] is game_state (string: lowercase = in-progress)
            # We rely on reset() being called at trial boundaries (see env patch).
            return surrogate.get_velocity()

        return get_velocity_fn

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Print summary statistics of the loaded trajectory library."""
        print("\n=== DataDrivenSurrogate Statistics ===")
        for label, trajs in self._trajs.items():
            if not trajs:
                print(f"  label {label}: NO DATA")
                continue
            lengths = [len(t) for t in trajs]
            speeds = [np.sqrt(t[:, 0]**2 + t[:, 1]**2).mean() for t in trajs]
            print(
                f"  label {label} ({LABEL_TO_NAME.get(label,'?'):>2}): "
                f"n={len(trajs):4d}  "
                f"len={np.mean(lengths):5.1f}±{np.std(lengths):4.1f}  "
                f"mean_speed={np.mean(speeds):.4f}"
            )
        print("======================================\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'online_arm_trajectories.csv'
    s = DataDrivenSurrogate(csv_path, rng_seed=42)
    s.print_stats()

    # Simulate a trial for target 3 (East)
    s.reset(3)
    print("Simulated velocities for target 3 (E):")
    for _ in range(17):
        v = s.get_velocity()
        print(f"  vx={v[0]:+.4f}  vy={v[1]:+.4f}")
