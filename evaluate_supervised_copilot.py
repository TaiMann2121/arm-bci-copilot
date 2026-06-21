"""
evaluate_supervised_copilot.py
================================
Evaluates the supervised LSTM copilot on all 7 subjects, using the same
protocol as evaluate.py so results are directly comparable to Run5 (+0.7pp).

KEY DIFFERENCES FROM TRAINING EVALUATION
-----------------------------------------
Training evaluation (in train_supervised_copilot_v2.py):
  - Only evaluates on test subjects S06 and S07
  - Each trial starts with cursor at (0,0) independently

This evaluation script:
  - Evaluates on ALL 7 subjects
  - Processes trials sequentially per subject (cursor carries over between
    trials, matching evaluate.py and the real deployment scenario)
  - Reports train-subject (S01-S05) vs test-subject (S06-S07) accuracy
    separately to distinguish generalization from memorization

RUN
---
From arm-bci-copilot/:

    python evaluate_supervised_copilot.py
    python evaluate_supervised_copilot.py --model supervised_copilot/best_model.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "data/online_arm_trajectories.csv"
DEFAULT_MODEL = "supervised_copilot/best_model.pt"

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX    = 432.0
COPILOT_VEL  = 0.02
INPUT_SIZE   = 5
HIDDEN_SIZE  = 64
N_LAYERS     = 2
VEL_MAG_MEAN = 0.0424
VEL_MAG_STD  = 0.0262

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
LABEL_NAMES    = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']
TRAIN_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05']
TEST_SUBJECTS  = ['S06', 'S07']


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_vel(vx, vy):
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)
    if mag > 1e-6:
        return vx/mag, vy/mag, mag_scaled
    return 0.0, 0.0, (0.0 - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)

def angle_pred(cursor):
    norm = np.linalg.norm(cursor)
    if norm < 1e-6: return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))


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


# ── data loading ──────────────────────────────────────────────────────────────

def load_trials_by_subject(csv_path):
    df = pd.read_csv(csv_path)
    group_cols = ['subject_id','session_number','run_number',
                  'trial_number','inner_trial_number']
    by_subj = {}
    for _, grp in df.groupby(group_cols):
        grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl  = int(grp['target_label'].iloc[0])
        subj = grp['subject_id'].iloc[0]
        if lbl not in range(8): continue
        cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx = np.diff(cx); vy = np.diff(cy)
        if len(vx) == 0: continue
        by_subj.setdefault(subj, []).append({
            'label': lbl, 'vx': vx, 'vy': vy,
        })
    return by_subj


# ── trial simulation ──────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_trial(model, trial):
    """
    Simulate one trial. Cursor starts at (0,0). LSTM hidden state resets.
    Returns (cop_correct, bci_correct).
    """
    vx_seq, vy_seq = trial['vx'], trial['vy']
    lbl = trial['label']
    T   = len(vx_seq)

    cursor_cop = np.zeros(2, dtype=np.float32)
    cursor_bci = np.zeros(2, dtype=np.float32)
    h, c = None, None

    for t in range(T):
        vx_t, vy_t = float(vx_seq[t]), float(vy_seq[t])
        nvx, nvy, vmag = normalize_vel(vx_t, vy_t)
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

    return angle_pred(cursor_cop) == lbl, angle_pred(cursor_bci) == lbl


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation(model_path):
    print("=" * 65)
    print("Supervised LSTM Copilot — Full Evaluation (all 7 subjects)")
    print("=" * 65)
    print(f"Model : {model_path}")
    print(f"Data  : {CSV_PATH}")
    print()

    print("Loading model...")
    model = LSTMCopilot()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Loading trials...")
    by_subj  = load_trials_by_subject(CSV_PATH)
    subjects = sorted(by_subj.keys())
    print(f"  {sum(len(v) for v in by_subj.values())} trials, {len(subjects)} subjects")
    print()

    results_by_subj = {}
    results_by_dir  = {i: {'cop':0,'bci':0,'n':0} for i in range(8)}

    print(f"  {'Subject':<10} {'Split':<6} {'BCI':>7}  {'Copilot':>8}  {'Delta':>7}")
    print("  " + "-" * 44)

    for subj in subjects:
        trials = by_subj[subj]
        cop_c = bci_c = 0
        for trial in trials:
            cop_ok, bci_ok = simulate_trial(model, trial)
            if cop_ok: cop_c += 1
            if bci_ok: bci_c += 1
            lbl = trial['label']
            results_by_dir[lbl]['cop'] += int(cop_ok)
            results_by_dir[lbl]['bci'] += int(bci_ok)
            results_by_dir[lbl]['n']   += 1

        n = len(trials)
        split = 'train' if subj in TRAIN_SUBJECTS else 'test'
        results_by_subj[subj] = {'cop':cop_c,'bci':bci_c,'n':n,'split':split}
        d = (cop_c - bci_c) / n * 100
        sign = '+' if d >= 0 else ''
        print(f"  {subj:<10} {split:<6} {bci_c/n*100:>6.1f}%  "
              f"{cop_c/n*100:>7.1f}%  {sign}{d:>5.1f}pp")

    # Aggregates
    def agg(subj_filter=None):
        rs = results_by_subj.values() if subj_filter is None else \
             [r for s,r in results_by_subj.items() if r['split']==subj_filter]
        tc = sum(r['cop'] for r in rs); tb = sum(r['bci'] for r in rs)
        tn = sum(r['n']   for r in rs)
        return tc, tb, tn

    all_cop, all_bci, all_n       = agg()
    trn_cop, trn_bci, trn_n       = agg('train')
    tst_cop, tst_bci, tst_n       = agg('test')

    print()
    print("=" * 65)
    print(f"OVERALL  ({all_n} trials, all 7 subjects):")
    print(f"  BCI baseline : {all_bci/all_n*100:.2f}%  ({all_bci}/{all_n})")
    print(f"  Copilot      : {all_cop/all_n*100:.2f}%  ({all_cop}/{all_n})")
    print(f"  Improvement  : {(all_cop-all_bci)/all_n*100:+.2f}pp")
    print()
    print(f"TRAIN subjects (S01-S05, {trn_n} trials):")
    print(f"  BCI baseline : {trn_bci/trn_n*100:.2f}%")
    print(f"  Copilot      : {trn_cop/trn_n*100:.2f}%")
    print(f"  Improvement  : {(trn_cop-trn_bci)/trn_n*100:+.2f}pp")
    print()
    print(f"TEST subjects  (S06-S07, {tst_n} trials):")
    print(f"  BCI baseline : {tst_bci/tst_n*100:.2f}%")
    print(f"  Copilot      : {tst_cop/tst_n*100:.2f}%")
    print(f"  Improvement  : {(tst_cop-tst_bci)/tst_n*100:+.2f}pp")
    print("=" * 65)
    print()

    print("Per-direction breakdown (all 7 subjects):")
    print(f"  {'Dir':<4}  {'BCI':>7}  {'Copilot':>8}  {'Delta':>7}")
    print("  " + "-" * 34)
    for i in range(8):
        r    = results_by_dir[i]
        bpct = r['bci'] / r['n'] * 100
        cpct = r['cop'] / r['n'] * 100
        d    = cpct - bpct
        sign = '+' if d >= 0 else ''
        print(f"  {LABEL_NAMES[i]:<4}  {bpct:>6.1f}%  {cpct:>7.1f}%  {sign}{d:>5.1f}pp")

    print()
    print("Reference — Run5 (RL chargeTargets, full evaluate.py):")
    run5 = {'NW':50.3,'N':64.8,'NE':38.8,'E':39.7,'SE':47.6,'S':51.0,'SW':43.3,'W':43.3}
    run5_base = {'NW':47.3,'N':60.2,'NE':41.6,'E':43.3,'SE':47.1,'S':46.3,'SW':41.6,'W':45.0}
    print(f"  {'Dir':<4}  {'BCI':>7}  {'Run5':>8}  {'Delta':>7}")
    print("  " + "-" * 34)
    for i in range(8):
        n = LABEL_NAMES[i]
        d = run5[n] - run5_base[n]
        sign = '+' if d >= 0 else ''
        print(f"  {n:<4}  {run5_base[n]:>6.1f}%  {run5[n]:>7.1f}%  {sign}{d:>5.1f}pp")
    run5_overall = sum(run5.values())/8
    run5_base_overall = sum(run5_base.values())/8
    print(f"  {'Mean':<4}  {run5_base_overall:>6.1f}%  {run5_overall:>7.1f}%  "
          f"{run5_overall-run5_base_overall:>+6.1f}pp")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_evaluation(args.model)
