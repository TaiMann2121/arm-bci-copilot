"""
evaluate_eegk_transfer.py
==========================
Zero-shot cross-decoder transfer evaluation: tests the V3 supervised LSTM
copilot (trained on old decoder data) on EEGK decoder trajectories without
any retraining.

The EEGK dataset was collected using a newer, better-performing BCI decoder.
The copilot has never seen this data. This script answers the question:
does the copilot transfer, or does the velocity distribution shift hurt it?

TWO NORMALIZATION CONDITIONS
------------------------------
The copilot's 5th input feature (vel_mag_scaled) is z-scored using normalization
constants. We run the evaluation under two conditions:

  OLD NORM  — uses constants from old decoder training data
              (VEL_MAG_MEAN=0.0424, VEL_MAG_STD=0.0262)
              Reflects true zero-shot transfer with no adaptation

  EEGK NORM — uses constants recomputed from EEGK data
              (computed from non-calibration trials at script load time)
              Isolates the effect of normalization mismatch vs model mismatch

Comparing OLD NORM vs EEGK NORM tells you whether the velocity distribution
shift is the primary bottleneck for transfer.

DATA
-----
EEGK data lives in a per-subject/per-session folder hierarchy:
  data/OnlineArmTrajectoryEEGK/<subject>/<task>/<session>/online_arm_trajectories.csv
  data/OnlineArmTrajectoryEEGK/<subject>/<task>/<session>/typing_stats.npz

Run 1 in each session is the label-balanced calibration run and is EXCLUDED.
Non-calibration trials start at run_number >= 2.

SUBJECTS
---------
EEGK has 5 of the original 7 subjects: S01, S02, S04, S05, S07.
S03 and S06 are absent. The script labels each subject with their role
in the original train/test split for reference.

RUN
---
From arm-bci-copilot/:

    python evaluate_eegk_transfer.py
    python evaluate_eegk_transfer.py --model supervised_copilot/best_model.pt
    python evaluate_eegk_transfer.py --data data/OnlineArmTrajectoryEEGK
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── paths ─────────────────────────────────────────────────────────────────────
EEGK_DIR      = "data/OnlineArmTrajectoryEEGK"
DEFAULT_MODEL = "supervised_copilot/best_model.pt"

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX   = 432.0
COPILOT_VEL = 0.02
INPUT_SIZE  = 5
HIDDEN_SIZE = 64
N_LAYERS    = 2

# Old decoder normalization constants (from training data)
OLD_VEL_MAG_MEAN = 0.0424
OLD_VEL_MAG_STD  = 0.0262

LABEL_TO_DIR = np.array([
    [-0.707,  0.707],  # 0 NW
    [ 0.000,  1.000],  # 1 N
    [ 0.707,  0.707],  # 2 NE
    [ 1.000,  0.000],  # 3 E
    [ 0.707, -0.707],  # 4 SE
    [ 0.000, -1.000],  # 5 S
    [-0.707, -0.707],  # 6 SW
    [-1.000,  0.000],  # 7 W
], dtype=np.float32)
LABEL_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

# Original train/test split labels (for reference only — model was not
# trained on any EEGK data)
ORIG_TRAIN = {'S01', 'S02', 'S03', 'S04', 'S05'}
ORIG_TEST  = {'S06', 'S07'}


# ── model ─────────────────────────────────────────────────────────────────────

class LSTMCopilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm       = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS,
                                  batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(HIDDEN_SIZE, 8)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out)


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_vel(vx, vy, mean, std):
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - mean) / (std + 1e-8)
    if mag > 1e-6:
        return vx / mag, vy / mag, mag_scaled
    return 0.0, 0.0, (0.0 - mean) / (std + 1e-8)

def angle_pred(cursor):
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))

def angle_error_deg(cursor, target_label):
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return None
    cursor_unit = cursor / norm
    target_unit = LABEL_TO_DIR[target_label]
    cos_angle = np.clip(np.dot(cursor_unit, target_unit), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ── data loading ──────────────────────────────────────────────────────────────

def load_eegk_trials(eegk_dir):
    """
    Walks the EEGK folder hierarchy, loads all non-calibration trials
    (run_number >= 2), and returns:
      by_subj  — dict mapping subject_id -> list of trial dicts
      vel_mags — flat array of all velocity magnitudes (for norm recomputation)
    """
    csv_files = sorted(glob.glob(
        os.path.join(eegk_dir, '**', 'online_arm_trajectories.csv'),
        recursive=True
    ))
    if not csv_files:
        raise FileNotFoundError(
            f"No EEGK CSV files found under {eegk_dir}. "
            f"Check the --data path."
        )

    group_cols = ['subject_id', 'session_number', 'run_number',
                  'trial_number', 'inner_trial_number']

    by_subj  = {}
    all_vels = []

    for fpath in csv_files:
        df = pd.read_csv(fpath)
        # Exclude calibration run
        df = df[df['run_number'] >= 2]
        if df.empty:
            continue

        for _, grp in df.groupby(group_cols):
            grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
            lbl  = int(grp['target_label'].iloc[0])
            subj = grp['subject_id'].iloc[0]
            if lbl not in range(8):
                continue

            cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
            cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
            vx = np.diff(cx)
            vy = np.diff(cy)
            if len(vx) == 0:
                continue

            mags = np.sqrt(vx**2 + vy**2)
            all_vels.extend(mags.tolist())

            by_subj.setdefault(subj, []).append({
                'label': lbl, 'vx': vx, 'vy': vy,
            })

    return by_subj, np.array(all_vels, dtype=np.float32)


# ── trial simulation ──────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_trial(model, trial, vel_mean, vel_std):
    """
    Simulate one trial under a given normalization condition.
    Returns (cop_correct, bci_correct, cop_angle_err, bci_angle_err).
    """
    vx_seq, vy_seq = trial['vx'], trial['vy']
    lbl = trial['label']
    T   = len(vx_seq)

    cursor_cop = np.zeros(2, dtype=np.float32)
    cursor_bci = np.zeros(2, dtype=np.float32)
    h, c = None, None

    for t in range(T):
        vx_t, vy_t = float(vx_seq[t]), float(vy_seq[t])
        nvx, nvy, vmag = normalize_vel(vx_t, vy_t, vel_mean, vel_std)
        bci_vel = np.array([vx_t, vy_t], dtype=np.float32)

        x_t = torch.tensor(
            np.array([[[cursor_cop[0], cursor_cop[1], nvx, nvy, vmag]]]),
            dtype=torch.float32
        )
        if h is None:
            lstm_out, (h, c) = model.lstm(x_t)
        else:
            lstm_out, (h, c) = model.lstm(x_t, (h, c))

        logits_t = model.classifier(lstm_out[:, -1, :])
        conf     = float(torch.softmax(logits_t, dim=-1).max().item())
        pred_t   = int(logits_t.argmax().item())

        correction = LABEL_TO_DIR[pred_t] * COPILOT_VEL * conf
        cursor_cop = np.clip(cursor_cop + bci_vel + correction, -1.5, 1.5)
        cursor_bci = np.clip(cursor_bci + bci_vel, -1.5, 1.5)

    cop_err = angle_error_deg(cursor_cop, lbl)
    bci_err = angle_error_deg(cursor_bci, lbl)
    return (angle_pred(cursor_cop) == lbl,
            angle_pred(cursor_bci) == lbl,
            cop_err, bci_err)


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate_condition(model, by_subj, vel_mean, vel_std, label):
    """Run full evaluation under one normalization condition."""
    subjects = sorted(by_subj.keys())

    results_by_subj = {}
    results_by_dir  = {i: {'cop': 0, 'bci': 0, 'n': 0,
                            'cop_errs': [], 'bci_errs': []}
                       for i in range(8)}

    print(f"\n{'─' * 65}")
    print(f"  Condition: {label}")
    print(f"  vel_mag_mean={vel_mean:.4f}  vel_mag_std={vel_std:.4f}")
    print(f"{'─' * 65}")
    print(f"  {'Subject':<10} {'Orig split':<12} {'BCI':>7}  {'BCI+Cop':>8}  {'Delta':>7}  {'Trials':>7}")
    print("  " + "-" * 58)

    for subj in subjects:
        trials = by_subj[subj]
        cop_c = bci_c = 0
        cop_errs_s, bci_errs_s = [], []

        for trial in trials:
            cop_ok, bci_ok, cop_err, bci_err = simulate_trial(
                model, trial, vel_mean, vel_std)
            if cop_ok: cop_c += 1
            if bci_ok: bci_c += 1
            if cop_err is not None: cop_errs_s.append(cop_err)
            if bci_err is not None: bci_errs_s.append(bci_err)
            lbl = trial['label']
            results_by_dir[lbl]['cop'] += int(cop_ok)
            results_by_dir[lbl]['bci'] += int(bci_ok)
            results_by_dir[lbl]['n']   += 1
            if cop_err is not None: results_by_dir[lbl]['cop_errs'].append(cop_err)
            if bci_err is not None: results_by_dir[lbl]['bci_errs'].append(bci_err)

        n     = len(trials)
        split = ('train' if subj in ORIG_TRAIN else
                 'test'  if subj in ORIG_TEST  else 'n/a')
        d     = (cop_c - bci_c) / n * 100
        sign  = '+' if d >= 0 else ''
        results_by_subj[subj] = {
            'cop': cop_c, 'bci': bci_c, 'n': n, 'split': split,
            'cop_errs': cop_errs_s, 'bci_errs': bci_errs_s,
        }
        print(f"  {subj:<10} {split:<12} {bci_c/n*100:>6.1f}%  "
              f"{cop_c/n*100:>7.1f}%  {sign}{d:>5.1f}pp  {n:>7}")

    # ── overall summary ────────────────────────────────────────────────────
    all_cop  = sum(r['cop'] for r in results_by_subj.values())
    all_bci  = sum(r['bci'] for r in results_by_subj.values())
    all_n    = sum(r['n']   for r in results_by_subj.values())
    all_c_e  = [e for r in results_by_subj.values() for e in r['cop_errs']]
    all_b_e  = [e for r in results_by_subj.values() for e in r['bci_errs']]

    print()
    print(f"  OVERALL ({all_n} trials, 5 subjects):")
    print(f"    BCI baseline : {all_bci/all_n*100:.2f}%")
    print(f"    BCI + Copilot: {all_cop/all_n*100:.2f}%")
    print(f"    Improvement  : {(all_cop-all_bci)/all_n*100:+.2f}pp")
    print(f"    Angle error  : BCI {np.mean(all_b_e):.2f}°  →  "
          f"BCI+Cop {np.mean(all_c_e):.2f}°  "
          f"({np.mean(all_b_e)-np.mean(all_c_e):+.2f}°)")

    # ── per-direction breakdown ────────────────────────────────────────────
    print()
    print(f"  Per-direction breakdown:")
    print(f"  {'Dir':<4}  {'n':>5}  {'BCI':>7}  {'BCI+Cop':>8}  "
          f"{'Δacc':>7}  {'BCI err':>9}  {'Cop err':>9}  {'Δerr':>7}")
    print("  " + "-" * 72)
    for i in range(8):
        r    = results_by_dir[i]
        if r['n'] == 0:
            print(f"  {LABEL_NAMES[i]:<4}  {'—':>5}")
            continue
        bpct = r['bci'] / r['n'] * 100
        cpct = r['cop'] / r['n'] * 100
        d    = cpct - bpct
        b_a  = np.mean(r['bci_errs']) if r['bci_errs'] else float('nan')
        c_a  = np.mean(r['cop_errs']) if r['cop_errs'] else float('nan')
        a_d  = b_a - c_a
        print(f"  {LABEL_NAMES[i]:<4}  {r['n']:>5}  {bpct:>6.1f}%  "
              f"{cpct:>7.1f}%  {d:>+6.1f}pp  "
              f"{b_a:>8.2f}°  {c_a:>8.2f}°  {a_d:>+6.2f}°")

    return {
        'overall_bci': all_bci / all_n * 100,
        'overall_cop': all_cop / all_n * 100,
        'delta':       (all_cop - all_bci) / all_n * 100,
        'bci_ang':     np.mean(all_b_e),
        'cop_ang':     np.mean(all_c_e),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation(model_path, eegk_dir):
    print("=" * 65)
    print("Zero-Shot Cross-Decoder Transfer Evaluation")
    print("Model trained on: old decoder (online_arm_trajectories.csv)")
    print("Evaluated on   : EEGK decoder (OnlineArmTrajectoryEEGK)")
    print("=" * 65)
    print(f"Model : {model_path}")
    print(f"Data  : {eegk_dir}")

    # Load model
    print("\nLoading model...")
    model = LSTMCopilot()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load EEGK data
    print("\nLoading EEGK trials (excluding calibration run 1)...")
    by_subj, vel_mags = load_eegk_trials(eegk_dir)
    total = sum(len(v) for v in by_subj.values())
    print(f"  {total} trials across {len(by_subj)} subjects: "
          f"{sorted(by_subj.keys())}")
    print(f"  Subjects missing vs original split: "
          f"{sorted((ORIG_TRAIN | ORIG_TEST) - set(by_subj.keys()))}")

    # Compute EEGK normalization constants
    eegk_mean = float(np.mean(vel_mags))
    eegk_std  = float(np.std(vel_mags))
    print(f"\nVelocity magnitude statistics:")
    print(f"  Old decoder : mean={OLD_VEL_MAG_MEAN:.4f}  std={OLD_VEL_MAG_STD:.4f}")
    print(f"  EEGK decoder: mean={eegk_mean:.4f}  std={eegk_std:.4f}")

    # ── Condition 1: old normalization (true zero-shot) ────────────────────
    res_old = evaluate_condition(
        model, by_subj,
        vel_mean=OLD_VEL_MAG_MEAN,
        vel_std=OLD_VEL_MAG_STD,
        label="OLD NORM (true zero-shot — old decoder constants)"
    )

    # ── Condition 2: EEGK normalization (norm-adapted) ────────────────────
    res_new = evaluate_condition(
        model, by_subj,
        vel_mean=eegk_mean,
        vel_std=eegk_std,
        label="EEGK NORM (norm-adapted — recomputed from EEGK data)"
    )

    # ── Side-by-side summary ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("CROSS-DECODER TRANSFER SUMMARY")
    print("=" * 65)
    print(f"  {'Condition':<42} {'BCI':>7}  {'BCI+Cop':>8}  {'Delta':>7}")
    print("  " + "-" * 68)
    print(f"  {'Old decoder baseline (reference)':42}  "
          f"{'46.70%':>7}  {'47.77%':>8}  {'+1.07pp':>7}")
    print(f"  {'EEGK — old norm (true zero-shot)':42}  "
          f"{res_old['overall_bci']:>6.2f}%  "
          f"{res_old['overall_cop']:>7.2f}%  "
          f"{res_old['delta']:>+6.2f}pp")
    print(f"  {'EEGK — EEGK norm (norm-adapted)':42}  "
          f"{res_new['overall_bci']:>6.2f}%  "
          f"{res_new['overall_cop']:>7.2f}%  "
          f"{res_new['delta']:>+6.2f}pp")
    print()
    print(f"  Angle error improvement:")
    print(f"    Old norm  : {res_old['bci_ang']:.2f}° → {res_old['cop_ang']:.2f}°  "
          f"({res_old['bci_ang']-res_old['cop_ang']:+.2f}°)")
    print(f"    EEGK norm : {res_new['bci_ang']:.2f}° → {res_new['cop_ang']:.2f}°  "
          f"({res_new['bci_ang']-res_new['cop_ang']:+.2f}°)")
    norm_effect = res_new['delta'] - res_old['delta']
    print()
    print(f"  Effect of normalization fix: {norm_effect:+.2f}pp")
    print(f"  (difference between EEGK norm and old norm conditions)")
    print("=" * 65)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--data',  default=EEGK_DIR)
    args = parser.parse_args()
    run_evaluation(args.model, args.data)
