"""
train_supervised_copilot_greedy.py
================================
Configurable supervised LSTM copilot trainer for Step 1 (data source search)
of the sequential greedy pipeline optimization.

CHANGES FROM V3
---------------
- Single CONFIG dict at top controls all experimental dimensions
- Data source: 'old_decoder', 'eegk_sim', 'eegk_real'
- Tick weighting: 'constant', 'linear', 'exponential'
- Subject split: 'cross_subject', 'all_subject', 'within_subject'
- Per-data-source normalization constants and MAX_TICKS auto-selected
- Output dir auto-named from config (no manual renaming between runs)
- Trial-length correction: Route 2 trials are 12 ticks (not 13 as previously noted)
- Everything else (model architecture, DAgger, evaluation) identical to V3

ARCHITECTURE (unchanged from V3)
---------------------------------
Stage 1 — LSTM target classifier:
  Input:  (cx, cy, vx_unit, vy_unit, vel_mag) per tick
  Output: softmax over 8 target directions

Stage 2 — Analytic corrective velocity:
  correction = LABEL_TO_DIR[predicted_label] * COPILOT_VEL * confidence

TRAINING (unchanged from V3)
------------------------------
Phase 1 (epochs 1-3):   raw BCI trajectories
Phase 2 (epochs 4-28):  DAgger augmented BCI+Copilot trajectories
                         (PHASE2_EPOCHS = 25, so total = 28 epochs)

NOTE ON V3 BASELINE
--------------------
V3 (Step 0 anchor, +1.07pp) was trained with LINEAR tick weighting, not constant.
The Step 0 row in the report reflects this. All greedy search runs (1A onward)
use EXPONENTIAL weighting as fixed, so 1A is not a straight rerun of V3 —
it is old decoder + exponential, which is the true Step 1 anchor.

RUN
---
From arm-bci-copilot/:
    python train_supervised_copilot_greedy.py

Edit CONFIG below to switch between experimental conditions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
import time

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG — edit this block to switch between runs
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data source ───────────────────────────────────────────────────────────
    # 'old_decoder'     : data/online_arm_trajectories.csv              (S01-S07, 17 ticks)
    # 'eegk_sim'        : data/online_arm_trajectories_EEGK_simulation.csv (S01-S07, 12 ticks)
    # 'eegk_real'       : data/OnlineArmTrajectoryEEGK/                 (5 subjects, variable ticks)
    # 'eegk_surrogate'  : NOT YET IMPLEMENTED — requires surrogate generation script
    'data_source': 'old_decoder',

    # ── Tick weighting in loss function ───────────────────────────────────────
    # 'constant'    : all ticks weighted equally          w = [1, 1, ..., 1]
    # 'linear'      : weight increases linearly           w = [1/T, 2/T, ..., 1]
    # 'exponential' : weight increases exponentially      w = exp(k * t/T), normalized
    # NOTE: V3 baseline (+1.07pp) used LINEAR weighting. All greedy runs fix EXPONENTIAL.
    'tick_weighting': 'exponential',

    # Exponent k for exponential weighting (only used when tick_weighting='exponential').
    # k=3: final tick ~20x weight of first tick.
    # k=5: final tick ~148x weight of first tick.
    # Has no effect when tick_weighting is 'constant' or 'linear'.
    'weight_exponent': 3.0,

    # ── Train/test subject split ───────────────────────────────────────────────
    # 'cross_subject'  : fixed train/test subject split per SUBJECT_SPLITS below
    # 'all_subject'    : train on all subjects, evaluate on all (no held-out subjects)
    # 'within_subject' : NOT YET IMPLEMENTED
    'split': 'cross_subject',
}

# ── Subject splits per data source ────────────────────────────────────────────
# For cross_subject split. Edit if you want a different held-out set.
SUBJECT_SPLITS = {
    'old_decoder': {
        'train': ['S01', 'S02', 'S03', 'S04', 'S05'],
        'test':  ['S06', 'S07'],
    },
    'eegk_sim': {
        'train': ['S01', 'S02', 'S03', 'S04', 'S05'],
        'test':  ['S06', 'S07'],
    },
    'eegk_real': {
        # 5 subjects in EEGK real: S01, S02, S04, S05, S07
        # Hold out S05 and S07 (largest trial counts) for test
        'train': ['S01', 'S02', 'S04'],
        'test':  ['S05', 'S07'],
    },
    'eegk_surrogate': {
        # Same subject pool as eegk_real (surrogate is calibrated to EEGK real profiles)
        # Populated once surrogate generation script is implemented
        'train': ['S01', 'S02', 'S04'],
        'test':  ['S05', 'S07'],
    },
}

# ── Per-data-source constants ──────────────────────────────────────────────────
DATA_SOURCE_PARAMS = {
    'old_decoder': {
        'csv_path':     'data/online_arm_trajectories.csv',
        'vel_mag_mean': 0.0424,
        'vel_mag_std':  0.0262,
        'max_ticks':    17,
    },
    'eegk_sim': {
        'csv_path':     'data/online_arm_trajectories_EEGK_simulation.csv',
        'vel_mag_mean': 0.0465,
        'vel_mag_std':  0.0197,
        'max_ticks':    12,
    },
    'eegk_real': {
        'csv_path':     'data/OnlineArmTrajectoryEEGK/',   # directory — loaded below
        'vel_mag_mean': 0.0367,
        'vel_mag_std':  0.0381,
        'max_ticks':    20,    # conservative upper bound; padded dynamically
    },
    'eegk_surrogate': {
        # Placeholder — velocity stats and csv_path to be filled once surrogate
        # generation script is implemented. Will raise NotImplementedError on load.
        'csv_path':     'data/online_arm_trajectories_EEGK_surrogate.csv',
        'vel_mag_mean': 0.0367,  # inherit from eegk_real until measured
        'vel_mag_std':  0.0381,
        'max_ticks':    20,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# FIXED HYPERPARAMETERS (identical to V3 — do not change between runs)
# ══════════════════════════════════════════════════════════════════════════════

RADIUS_PX    = 432.0
COPILOT_VEL  = 0.02
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 25
BATCH_SIZE   = 128
LR           = 1e-3
HIDDEN_SIZE  = 64
N_LAYERS     = 2
INPUT_SIZE   = 5    # (cx, cy, vx_unit, vy_unit, vel_mag)
DEVICE       = 'cpu'

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


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED CONFIG — resolved from CONFIG dict, do not edit
# ══════════════════════════════════════════════════════════════════════════════

def resolve_config(cfg: dict) -> dict:
    """Resolve CONFIG into all runtime parameters."""
    ds     = cfg['data_source']
    wt     = cfg['tick_weighting']
    split  = cfg['split']
    params = DATA_SOURCE_PARAMS[ds]

    # Auto-name output dir from config
    wt_tag = wt if wt != 'exponential' else f"exp{cfg['weight_exponent']:.0f}"
    out_dir = f"supervised_copilot_{ds}_{wt_tag}_{split}"

    return {
        **cfg,
        **params,
        'train_subjects': SUBJECT_SPLITS[ds]['train'] if split == 'cross_subject' else None,
        'test_subjects':  SUBJECT_SPLITS[ds]['test']  if split == 'cross_subject' else None,
        'output_dir':     Path(out_dir),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_vel(vx: float, vy: float, vel_mag_mean: float, vel_mag_std: float):
    """
    Returns (vx_unit, vy_unit, vel_mag_scaled).
    Direction is unit-normalized; magnitude is z-scored using data-source stats.
    """
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - vel_mag_mean) / (vel_mag_std + 1e-8)
    if mag > 1e-6:
        return vx / mag, vy / mag, mag_scaled
    return 0.0, 0.0, (0.0 - vel_mag_mean) / (vel_mag_std + 1e-8)


def angle_pred(cursor: np.ndarray) -> int:
    """Argmax dot product — matches arm_prediction_label."""
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))


def make_tick_weights(T: int, weighting: str, exponent: float,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Returns a (T,) tensor of per-tick loss weights, normalized to sum to T
    so total loss scale is comparable across weighting schemes.

    constant:    [1, 1, 1, ..., 1]
    linear:      [1/T, 2/T, ..., 1]
    exponential: exp(k * t/T) normalized
    """
    t = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    if weighting == 'constant':
        w = torch.ones(T, dtype=torch.float32, device=device)
    elif weighting == 'linear':
        w = t / T
    elif weighting == 'exponential':
        w = torch.exp(exponent * (t - 1) / (T - 1) if T > 1 else torch.zeros(T))
    else:
        raise ValueError(f"Unknown tick_weighting: {weighting}")
    # Normalize so weights sum to T (keeps loss scale consistent with constant)
    w = w * T / w.sum()
    return w


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_trials_csv(csv_path: str) -> dict:
    """
    Load trials from a single CSV file.
    Returns dict: subject_id -> list of trial dicts.
    Each trial: {'label': int, 'cx': array, 'cy': array, 'vx': array, 'vy': array}
    """
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    group_cols = ['subject_id', 'session_number', 'run_number',
                  'trial_number', 'inner_trial_number']
    by_subj = {}
    skipped = 0
    for _, grp in df.groupby(group_cols):
        grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl  = int(grp['target_label'].iloc[0])
        subj = grp['subject_id'].iloc[0]
        if lbl not in range(8):
            skipped += 1
            continue
        cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx = np.diff(cx)
        vy = np.diff(cy)
        if len(vx) == 0:
            skipped += 1
            continue
        by_subj.setdefault(subj, []).append({
            'label': lbl,
            'cx': cx, 'cy': cy,
            'vx': vx, 'vy': vy,
        })
    if skipped:
        print(f"  Skipped {skipped} invalid trials")
    return by_subj


def load_trials_eegk_dir(dir_path: str) -> dict:
    """
    Load EEGK real data from directory of CSVs (excluding calibration run 1).
    Merges all files, returns same format as load_trials_csv.
    """
    p = Path(dir_path)
    csv_files = sorted(p.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")
    print(f"  Found {len(csv_files)} CSV files in {dir_path}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Exclude calibration run 1
        if 'run_number' in df.columns:
            df = df[df['run_number'] != 1]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined)} rows")

    # Use same CSV loading logic
    group_cols = ['subject_id', 'session_number', 'run_number',
                  'trial_number', 'inner_trial_number']
    by_subj = {}
    skipped = 0
    for _, grp in combined.groupby(group_cols):
        grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl  = int(grp['target_label'].iloc[0])
        subj = grp['subject_id'].iloc[0]
        if lbl not in range(8):
            skipped += 1
            continue
        cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx = np.diff(cx)
        vy = np.diff(cy)
        if len(vx) == 0:
            skipped += 1
            continue
        by_subj.setdefault(subj, []).append({
            'label': lbl,
            'cx': cx, 'cy': cy,
            'vx': vx, 'vy': vy,
        })
    if skipped:
        print(f"  Skipped {skipped} invalid trials")
    return by_subj


def load_trials(rcfg: dict) -> dict:
    """Dispatch to correct loader based on data_source."""
    ds = rcfg['data_source']
    if ds == 'eegk_surrogate':
        raise NotImplementedError(
            "eegk_surrogate data source is not yet implemented. "
            "Run the surrogate generation script first, then update "
            "DATA_SOURCE_PARAMS['eegk_surrogate']['csv_path'] and velocity stats."
        )
    elif ds == 'eegk_real':
        return load_trials_eegk_dir(rcfg['csv_path'])
    else:
        return load_trials_csv(rcfg['csv_path'])


def split_trials(by_subj: dict, rcfg: dict) -> tuple:
    """
    Returns (train_trials, test_trials) lists based on split config.
    For all_subject: test_trials == train_trials (in-distribution eval).
    """
    split = rcfg['split']
    if split == 'cross_subject':
        train = [t for s in rcfg['train_subjects'] if s in by_subj
                 for t in by_subj[s]]
        test  = [t for s in rcfg['test_subjects']  if s in by_subj
                 for t in by_subj[s]]
        return train, test
    elif split == 'all_subject':
        all_trials = [t for s in by_subj for t in by_subj[s]]
        return all_trials, all_trials
    else:
        raise ValueError(f"split='{split}' not yet implemented. "
                         f"Use 'cross_subject' or 'all_subject'.")


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_sequence_bci(trial: dict, rcfg: dict) -> tuple:
    """
    Build input sequence from raw BCI trajectory (Phase 1).
    Returns (seq, label, T): seq is (T, INPUT_SIZE) float32.
    """
    vx, vy = trial['vx'], trial['vy']
    cx, cy = trial['cx'], trial['cy']
    T = len(vx)
    seq = np.zeros((T, INPUT_SIZE), dtype=np.float32)
    for t in range(T):
        nvx, nvy, vmag = normalize_vel(vx[t], vy[t],
                                       rcfg['vel_mag_mean'], rcfg['vel_mag_std'])
        seq[t] = [cx[t], cy[t], nvx, nvy, vmag]
    return seq, trial['label'], T


def build_sequence_augmented(trial: dict, model: nn.Module, rcfg: dict) -> tuple:
    """
    Build input sequence using running copilot (Phase 2 / DAgger).
    Generates sequences matching the distribution at inference time.
    Returns (seq, label, T).
    """
    model.eval()
    vx, vy = trial['vx'], trial['vy']
    T = len(vx)
    seq = np.zeros((T, INPUT_SIZE), dtype=np.float32)

    cursor = np.zeros(2, dtype=np.float32)
    h, c   = None, None

    with torch.no_grad():
        for t in range(T):
            nvx, nvy, vmag = normalize_vel(vx[t], vy[t],
                                           rcfg['vel_mag_mean'], rcfg['vel_mag_std'])
            seq[t] = [cursor[0], cursor[1], nvx, nvy, vmag]

            x_t = torch.tensor(np.array([[seq[t]]]), dtype=torch.float32)  # (1,1,5)
            if h is None:
                logits, (h, c) = model.lstm(x_t)
            else:
                logits, (h, c) = model.lstm(x_t, (h, c))

            cls_logits = model.classifier(logits[:, -1, :])
            conf       = float(torch.softmax(cls_logits, dim=-1).max().item())
            pred_label = int(cls_logits.argmax(dim=-1).item())

            correction = LABEL_TO_DIR[pred_label] * COPILOT_VEL * conf
            bci_vel    = np.array([vx[t], vy[t]], dtype=np.float32)
            cursor     = np.clip(cursor + bci_vel + correction, -1.5, 1.5)

    return seq, trial['label'], T


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class BCIDataset(Dataset):
    """
    Dataset of (padded_sequence, label, mask) tuples.
    Padded to max_ticks; mask marks valid ticks for loss computation.
    """
    def __init__(self, sequences: list, max_ticks: int):
        self.seqs   = []
        self.labels = []
        self.masks  = []
        for seq, label, T in sequences:
            padded = np.zeros((max_ticks, INPUT_SIZE), dtype=np.float32)
            T_clip = min(T, max_ticks)
            padded[:T_clip] = seq[:T_clip]
            mask = np.zeros(max_ticks, dtype=np.float32)
            mask[:T_clip] = 1.0
            self.seqs.append(padded)
            self.labels.append(label)
            self.masks.append(mask)
        self.seqs   = torch.tensor(np.stack(self.seqs),   dtype=torch.float32)
        self.labels = torch.tensor(self.labels,            dtype=torch.long)
        self.masks  = torch.tensor(np.stack(self.masks),  dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i], self.masks[i]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL (unchanged from V3)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMCopilot(nn.Module):
    """Two-layer LSTM + linear classifier. Predicts target at every tick."""
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 n_layers=N_LAYERS, n_classes=8):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, n_layers,
                                  batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """x: (B, T, 5) → logits: (B, T, 8)"""
        out, _ = self.lstm(x)
        return self.classifier(out)

    def predict_label(self, seq_np: np.ndarray) -> int:
        """seq_np: (T, 5) → int label"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(seq_np[None], dtype=torch.float32)
            logits = self.forward(x)
            return int(logits[0, -1, :].argmax().item())


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, rcfg: dict) -> float:
    """Train one epoch. Returns mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    total_loss = 0.0
    total_toks = 0.0

    for seqs, labels, masks in loader:
        seqs   = seqs.to(DEVICE)    # (B, T, 5)
        labels = labels.to(DEVICE)  # (B,)
        masks  = masks.to(DEVICE)   # (B, T)

        logits = model(seqs)         # (B, T, 8)
        B, T, C = logits.shape

        # Tick weights: shape (T,), normalized
        tick_w = make_tick_weights(T, rcfg['tick_weighting'],
                                   rcfg['weight_exponent'], DEVICE)  # (T,)
        tick_w = tick_w.unsqueeze(0).expand(B, T).reshape(B * T)     # (B*T,)

        logits_flat = logits.reshape(B * T, C)
        labels_flat = labels.unsqueeze(1).expand(B, T).reshape(B * T)
        masks_flat  = masks.reshape(B * T)

        combined_w = masks_flat * tick_w
        loss_flat  = criterion(logits_flat, labels_flat)              # (B*T,)
        loss = (loss_flat * combined_w).sum() / (combined_w.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * masks_flat.sum().item()
        total_toks += masks_flat.sum().item()

    return total_loss / (total_toks + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION (unchanged from V3)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_fast(model: nn.Module, trials: list, rcfg: dict) -> dict:
    """
    Fast evaluation: simulate full trial with per-tick copilot corrections.
    Returns accuracy, bci_baseline, per-direction breakdown.
    """
    model.eval()
    correct_by_dir  = {i: 0 for i in range(8)}
    total_by_dir    = {i: 0 for i in range(8)}
    bci_correct_dir = {i: 0 for i in range(8)}

    for trial in trials:
        lbl = trial['label']
        T   = len(trial['vx'])

        # Simulate augmented trajectory
        cursor = np.zeros(2, dtype=np.float32)
        h, c   = None, None

        for t in range(T):
            nvx, nvy, vmag = normalize_vel(trial['vx'][t], trial['vy'][t],
                                           rcfg['vel_mag_mean'], rcfg['vel_mag_std'])
            x_t = torch.tensor(
                np.array([[[cursor[0], cursor[1], nvx, nvy, vmag]]]),
                dtype=torch.float32)
            if h is None:
                lstm_out, (h, c) = model.lstm(x_t)
            else:
                lstm_out, (h, c) = model.lstm(x_t, (h, c))

            logits_t   = model.classifier(lstm_out[:, -1, :])
            conf       = float(torch.softmax(logits_t, dim=-1).max().item())
            pred_t     = int(logits_t.argmax().item())
            correction = LABEL_TO_DIR[pred_t] * COPILOT_VEL * conf
            bci_vel    = np.array([trial['vx'][t], trial['vy'][t]], dtype=np.float32)
            cursor     = np.clip(cursor + bci_vel + correction, -1.5, 1.5)

        if angle_pred(cursor) == lbl:
            correct_by_dir[lbl] += 1
        total_by_dir[lbl] += 1

        # BCI baseline (no copilot)
        bci_cursor = np.zeros(2, dtype=np.float32)
        for t in range(T):
            bci_cursor = np.clip(
                bci_cursor + np.array([trial['vx'][t], trial['vy'][t]], dtype=np.float32),
                -1.5, 1.5)
        if angle_pred(bci_cursor) == lbl:
            bci_correct_dir[lbl] += 1

    total   = sum(total_by_dir.values())
    correct = sum(correct_by_dir.values())
    bci_c   = sum(bci_correct_dir.values())
    return {
        'accuracy':     correct / total,
        'bci_baseline': bci_c   / total,
        'by_direction': {i: correct_by_dir[i]  / total_by_dir[i] for i in range(8)},
        'bci_by_dir':   {i: bci_correct_dir[i] / total_by_dir[i] for i in range(8)},
        'n_correct':    correct,
        'n_total':      total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rcfg = resolve_config(CONFIG)
    rcfg['output_dir'].mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Print run summary ─────────────────────────────────────────────────────
    print("=" * 65)
    print("Supervised LSTM Copilot Training - Greedy")
    print("=" * 65)
    print(f"  Data source    : {rcfg['data_source']}")
    print(f"  CSV/dir path   : {rcfg['csv_path']}")
    print(f"  Tick weighting : {rcfg['tick_weighting']}", end='')
    if rcfg['tick_weighting'] == 'exponential':
        print(f"  (k={rcfg['weight_exponent']})", end='')
    print()
    print(f"  Split          : {rcfg['split']}")
    if rcfg['split'] == 'cross_subject':
        print(f"  Train subjects : {rcfg['train_subjects']}")
        print(f"  Test subjects  : {rcfg['test_subjects']}")
    print(f"  Vel norm       : mean={rcfg['vel_mag_mean']}, std={rcfg['vel_mag_std']}")
    print(f"  MAX_TICKS      : {rcfg['max_ticks']}")
    print(f"  Output dir     : {rcfg['output_dir']}/")
    print(f"  Phase 1 epochs : {PHASE1_EPOCHS}")
    print(f"  Phase 2 epochs : {PHASE2_EPOCHS}")
    print(f"  LSTM           : hidden={HIDDEN_SIZE}, layers={N_LAYERS}, input={INPUT_SIZE}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    by_subj = load_trials(rcfg)
    total = sum(len(v) for v in by_subj.values())
    print(f"  Loaded {total} trials from {len(by_subj)} subjects")
    for s in sorted(by_subj):
        print(f"    {s}: {len(by_subj[s])} trials")

    train_trials, test_trials = split_trials(by_subj, rcfg)
    print(f"\nTrain: {len(train_trials)} trials | Test: {len(test_trials)} trials\n")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model     = LSTMCopilot(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                             n_layers=N_LAYERS, n_classes=8).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = rcfg['output_dir'] / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phase', 'train_loss',
                         'test_acc', 'bci_baseline', 'delta_pp', 'time_s',
                         'NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W'])

    best_acc     = 0.0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2
        t0    = time.time()

        print(f"Epoch {epoch}/{total_epochs} (Phase {phase}) — building sequences...",
              end=' ', flush=True)
        if phase == 1:
            seqs = [build_sequence_bci(t, rcfg) for t in train_trials]
        else:
            seqs = [build_sequence_augmented(t, model, rcfg) for t in train_trials]
        print(f"done ({len(seqs)} sequences)")

        dataset = BCIDataset(seqs, max_ticks=rcfg['max_ticks'])
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

        train_loss = train_epoch(model, loader, optimizer, rcfg)
        scheduler.step()

        print(f"  Evaluating on {len(test_trials)} test trials...", end=' ', flush=True)
        results = evaluate_fast(model, test_trials, rcfg)
        elapsed = time.time() - t0

        acc      = results['accuracy']
        bci_base = results['bci_baseline']
        delta    = (acc - bci_base) * 100
        sign     = '+' if delta >= 0 else ''

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), rcfg['output_dir'] / "best_model.pt")
            best_str = " ← best"
        else:
            best_str = ""

        print("done")
        print(f"  Loss: {train_loss:.4f} | "
              f"BCI: {bci_base*100:.2f}% | "
              f"Copilot: {acc*100:.2f}% | "
              f"Delta: {sign}{delta:.2f}pp{best_str} | "
              f"Time: {elapsed:.0f}s")

        # Per-direction table
        print(f"  {'Dir':<4}", end='')
        for n in LABEL_NAMES: print(f"  {n:>6}", end='')
        print()
        print(f"  {'BCI':<4}", end='')
        for i in range(8): print(f"  {results['bci_by_dir'][i]*100:>5.1f}%", end='')
        print()
        print(f"  {'Cop':<4}", end='')
        for i in range(8): print(f"  {results['by_direction'][i]*100:>5.1f}%", end='')
        print()
        print(f"  {'Δpp':<4}", end='')
        for i in range(8):
            d = (results['by_direction'][i] - results['bci_by_dir'][i]) * 100
            s = '+' if d >= 0 else ''
            print(f"  {s}{d:>4.1f}", end='')
        print('\n')

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, f"{train_loss:.6f}",
                f"{acc*100:.2f}", f"{bci_base*100:.2f}",
                f"{delta:.2f}", f"{elapsed:.1f}",
                *[f"{results['by_direction'][i]*100:.1f}" for i in range(8)]
            ])

    # ── Save final model ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), rcfg['output_dir'] / "model.pt")

    print("=" * 65)
    print("Training complete.")
    print(f"  Data source  : {rcfg['data_source']}")
    print(f"  Weighting    : {rcfg['tick_weighting']}", end='')
    if rcfg['tick_weighting'] == 'exponential':
        print(f" (k={rcfg['weight_exponent']})", end='')
    print()
    print(f"  Split        : {rcfg['split']}")
    print(f"  Best test acc: {best_acc*100:.2f}%")
    print(f"  BCI baseline : {results['bci_baseline']*100:.2f}%")
    print(f"  Best delta   : {(best_acc - results['bci_baseline'])*100:+.2f}pp")
    print(f"  Models saved : {rcfg['output_dir']}/")
    print(f"  Log saved    : {log_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
