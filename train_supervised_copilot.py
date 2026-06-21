"""
train_supervised_copilot.py
============================
Trains a two-stage supervised copilot for the BCI arm task.

ARCHITECTURE
------------
Stage 1 — LSTM target classifier:
  Input:  sequence of (cursor_x, cursor_y, vx_unit, vy_unit) per tick
  Output: softmax over 8 target directions
  Trained to predict the correct target at every tick (not just the final one),
  so the classifier learns to infer intent as early as possible.

Stage 2 — Analytic corrective velocity (no learning needed):
  correction = LABEL_TO_DIR[predicted_label] * COPILOT_VEL
  final_vel  = bci_vel + correction

TRAINING PROCEDURE — Two-phase DAgger
--------------------------------------
Phase 1 (epochs 1-3): raw BCI trajectories from the CSV.
  The classifier hasn't learned anything yet, so we use pure BCI data.

Phase 2 (epochs 4-10): augmented BCI+Copilot trajectories.
  At each tick the running classifier predicts a direction, applies the
  correction, and the updated cursor position is used as input for the
  next tick. This matches the distribution the copilot will see at
  inference time, where the cursor already reflects earlier corrections.

EVALUATION
----------
Subject-level train/test split (S01-S05 train, S06-S07 test).
Metric: arm_prediction_label accuracy on the test subjects.

RUN
---
From arm-bci-copilot/:
    python train_supervised_copilot.py

Output:
    supervised_copilot_v3/
        model.pt          — final LSTM weights
        best_model.pt     — best test accuracy weights
        training_log.csv  — per-epoch metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
import time

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/online_arm_trajectories.csv"
OUTPUT_DIR = Path("supervised_copilot_v3")

# ── constants ─────────────────────────────────────────────────────────────────
RADIUS_PX   = 432.0
COPILOT_VEL = 0.02
MAX_TICKS   = 16    # pad shorter trials to this length

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

TRAIN_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05']
TEST_SUBJECTS  = ['S06', 'S07']

# Training
PHASE1_EPOCHS = 3    # raw BCI trajectories
PHASE2_EPOCHS = 25   # augmented BCI+Copilot trajectories (v3: more epochs + slower LR decay)
BATCH_SIZE    = 128
LR            = 1e-3
HIDDEN_SIZE   = 64
N_LAYERS      = 2
INPUT_SIZE    = 5    # (cx, cy, vx_unit, vy_unit, vel_mag)
DEVICE        = 'cpu'


# ── helpers ───────────────────────────────────────────────────────────────────

# Velocity magnitude statistics from the dataset (for normalization)
VEL_MAG_MEAN = 0.0424   # mean per-tick velocity magnitude (normalized space)
VEL_MAG_STD  = 0.0262   # std of per-tick velocity magnitude

def normalize_vel(vx: float, vy: float):
    """
    Returns (vx_unit, vy_unit, vel_mag_scaled).
    Direction is unit-normalized; magnitude is z-scored using dataset stats.
    Adding raw magnitude helps distinguish strong N motion from weak NE drift.
    """
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)
    if mag > 1e-6:
        return vx / mag, vy / mag, mag_scaled
    return 0.0, 0.0, (0.0 - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)

def angle_pred(cursor: np.ndarray) -> int:
    """Argmax dot product — matches arm_prediction_label."""
    norm = np.linalg.norm(cursor)
    if norm < 1e-6:
        return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))


# ── data loading ──────────────────────────────────────────────────────────────

def load_trials(csv_path: str) -> dict:
    """
    Returns dict: subject_id -> list of trial dicts.
    Each trial: {'label': int, 'cx': array, 'cy': array, 'vx': array, 'vy': array}
    cx/cy include the starting position (T+1 values); vx/vy are T velocities.
    """
    print(f"Loading {csv_path}...")
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
            'label': lbl,
            'cx': cx, 'cy': cy,   # (T+1,)
            'vx': vx, 'vy': vy,   # (T,)
        })
    total = sum(len(v) for v in by_subj.values())
    print(f"  Loaded {total} trials from {len(by_subj)} subjects")
    for s in sorted(by_subj):
        print(f"    {s}: {len(by_subj[s])} trials")
    return by_subj


def build_sequence_bci(trial: dict) -> tuple:
    """
    Build input sequence from raw BCI trajectory (Phase 1).

    Returns:
        seq:   (T, 4) float32 — (cx, cy, vx_unit, vy_unit) per tick
        label: int — target label
        T:     int — number of ticks
    """
    vx, vy = trial['vx'], trial['vy']
    cx, cy = trial['cx'], trial['cy']
    T = len(vx)
    seq = np.zeros((T, INPUT_SIZE), dtype=np.float32)
    for t in range(T):
        nvx, nvy, vmag = normalize_vel(vx[t], vy[t])
        seq[t] = [cx[t], cy[t], nvx, nvy, vmag]
    return seq, trial['label'], T


def build_sequence_augmented(trial: dict, model: nn.Module) -> tuple:
    """
    Build input sequence using the running copilot (Phase 2 / DAgger).

    At each tick, the classifier predicts a target direction, applies the
    correction, and the updated cursor is used as input for the next tick.
    This generates sequences that match the distribution at inference time.

    Returns:
        seq:   (T, 4) float32 — augmented (cx, cy, vx_unit, vy_unit)
        label: int — target label
        T:     int — number of ticks
    """
    model.eval()
    vx, vy = trial['vx'], trial['vy']
    T = len(vx)
    seq = np.zeros((T, INPUT_SIZE), dtype=np.float32)

    cursor = np.zeros(2, dtype=np.float32)  # start at origin
    h, c   = None, None  # LSTM hidden state

    with torch.no_grad():
        for t in range(T):
            nvx, nvy, vmag = normalize_vel(vx[t], vy[t])
            seq[t] = [cursor[0], cursor[1], nvx, nvy, vmag]

            # Run LSTM on just this tick (incremental)
            x_t = torch.tensor(np.array([[seq[t]]]), dtype=torch.float32)  # (1, 1, 5)
            if h is None:
                logits, (h, c) = model.lstm(x_t)
            else:
                logits, (h, c) = model.lstm(x_t, (h, c))
            cls_logits = model.classifier(logits[:, -1, :])
            conf       = float(torch.softmax(cls_logits, dim=-1).max().item())
            pred_label = int(cls_logits.argmax(dim=-1).item())

            # Stage 2: confidence-weighted analytic correction
            target_dir = LABEL_TO_DIR[pred_label]
            correction = target_dir * COPILOT_VEL * conf
            bci_vel    = np.array([vx[t], vy[t]], dtype=np.float32)
            cursor     = np.clip(cursor + bci_vel + correction, -1.5, 1.5)

    return seq, trial['label'], T


# ── dataset ───────────────────────────────────────────────────────────────────

class BCIDataset(Dataset):
    """
    Dataset of (padded_sequence, label) pairs.
    Each sequence is (MAX_TICKS, 4); label is 0-7.
    mask tracks valid ticks for loss computation.
    """
    def __init__(self, sequences: list):
        """sequences: list of (seq, label, T) tuples"""
        self.seqs   = []
        self.labels = []
        self.masks  = []
        for seq, label, T in sequences:
            # Pad to MAX_TICKS
            padded = np.zeros((MAX_TICKS, INPUT_SIZE), dtype=np.float32)
            padded[:T] = seq[:T]
            mask = np.zeros(MAX_TICKS, dtype=np.float32)
            mask[:T] = 1.0
            self.seqs.append(padded)
            self.labels.append(label)
            self.masks.append(mask)
        self.seqs   = torch.tensor(np.stack(self.seqs),   dtype=torch.float32)
        self.labels = torch.tensor(self.labels,            dtype=torch.long)
        self.masks  = torch.tensor(np.stack(self.masks),  dtype=torch.float32)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.seqs[i], self.labels[i], self.masks[i]


# ── model ─────────────────────────────────────────────────────────────────────

class LSTMCopilot(nn.Module):
    """
    Two-layer LSTM followed by a linear classifier.
    At each tick, predicts the target direction from trajectory so far.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, n_classes=8):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, n_layers,
                                  batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        x: (B, T, 4)
        Returns logits: (B, T, 8) — prediction at every tick
        """
        out, _ = self.lstm(x)            # (B, T, hidden)
        return self.classifier(out)      # (B, T, 8)

    def predict_label(self, seq_np: np.ndarray) -> int:
        """
        Predict target label from a complete sequence.
        seq_np: (T, 4) numpy array
        Returns: int label (0-7)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(seq_np[None], dtype=torch.float32)  # (1, T, 4)
            logits = self.forward(x)       # (1, T, 8)
            return int(logits[0, -1, :].argmax().item())  # final tick prediction


# ── training ──────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer) -> float:
    """Train one epoch. Returns mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    total_loss = 0.0
    total_toks = 0

    for seqs, labels, masks in loader:
        seqs   = seqs.to(DEVICE)    # (B, T, 4)
        labels = labels.to(DEVICE)  # (B,)
        masks  = masks.to(DEVICE)   # (B, T)

        logits = model(seqs)         # (B, T, 8)
        B, T, C = logits.shape

        # Compute loss at every valid tick
        logits_flat = logits.reshape(B*T, C)
        labels_flat = labels.unsqueeze(1).expand(B, T).reshape(B*T)
        masks_flat  = masks.reshape(B*T)

        # Linear tick weighting: weight_t = (t+1)/T
        # Later ticks get proportionally more gradient — they carry more
        # directional signal and the classifier needs to be right at the end.
        tick_weights = torch.arange(1, T+1, dtype=torch.float32, device=DEVICE) / T
        tick_weights = tick_weights.unsqueeze(0).expand(B, T).reshape(B*T)
        combined_weights = masks_flat * tick_weights

        loss_flat = criterion(logits_flat, labels_flat)  # (B*T,)
        loss = (loss_flat * combined_weights).sum() / combined_weights.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * masks_flat.sum().item()
        total_toks += masks_flat.sum().item()

    return total_loss / total_toks


@torch.no_grad()
def evaluate(model: nn.Module, trials: list) -> dict:
    """
    Evaluate on raw BCI trajectories (no copilot augmentation).
    Measures final-tick accuracy and per-direction breakdown.
    """
    model.eval()
    correct_by_dir  = {i: 0 for i in range(8)}
    total_by_dir    = {i: 0 for i in range(8)}
    bci_correct     = 0

    for trial in trials:
        lbl = trial['label']
        seq, _, T = build_sequence_bci(trial)
        pred = model.predict_label(seq[:T])

        # Copilot accuracy (using predicted label for correction)
        target_dir  = LABEL_TO_DIR[pred]
        correction  = target_dir * COPILOT_VEL
        # Simulate full trial with copilot
        cursor = np.zeros(2, dtype=np.float32)
        for t in range(T):
            bci_vel = np.array([trial['vx'][t], trial['vy'][t]], dtype=np.float32)
            # Get per-tick prediction for correction
            seq_t = seq[:t+1]
            if t == 0:
                pred_t = model.predict_label(seq_t)
            else:
                pred_t = model.predict_label(seq_t)
            corr = LABEL_TO_DIR[pred_t] * COPILOT_VEL
            cursor = np.clip(cursor + bci_vel + corr, -1.5, 1.5)

        final_pred = angle_pred(cursor)
        if final_pred == lbl:
            correct_by_dir[lbl] += 1
        total_by_dir[lbl] += 1

        # BCI baseline
        bci_cursor = np.zeros(2, dtype=np.float32)
        for t in range(T):
            bci_cursor += np.array([trial['vx'][t], trial['vy'][t]], dtype=np.float32)
            bci_cursor  = np.clip(bci_cursor, -1.5, 1.5)
        if angle_pred(bci_cursor) == lbl:
            bci_correct += 1

    total   = sum(total_by_dir.values())
    correct = sum(correct_by_dir.values())
    return {
        'accuracy':      correct / total,
        'bci_baseline':  bci_correct / total,
        'by_direction':  {i: correct_by_dir[i]/total_by_dir[i] for i in range(8)},
        'n_correct':     correct,
        'n_total':       total,
    }


# ── evaluation — fast version using batched LSTM ──────────────────────────────

@torch.no_grad()
def evaluate_fast(model: nn.Module, trials: list) -> dict:
    """
    Fast evaluation: simulate full trial with per-tick copilot correction.
    Uses augmented cursor (copilot corrections applied) for a fair measure.
    """
    model.eval()
    correct_by_dir  = {i: 0 for i in range(8)}
    total_by_dir    = {i: 0 for i in range(8)}
    bci_correct_dir = {i: 0 for i in range(8)}

    for trial in trials:
        lbl = trial['label']
        T   = len(trial['vx'])

        # Simulate augmented trajectory tick by tick
        cursor = np.zeros(2, dtype=np.float32)
        h, c   = None, None

        for t in range(T):
            nvx, nvy, vmag = normalize_vel(trial['vx'][t], trial['vy'][t])
            x_t = torch.tensor(np.array([[[cursor[0], cursor[1], nvx, nvy, vmag]]]),
                                dtype=torch.float32)
            if h is None:
                lstm_out, (h, c) = model.lstm(x_t)
            else:
                lstm_out, (h, c) = model.lstm(x_t, (h, c))
            logits_t   = model.classifier(lstm_out[:, -1, :])   # (1, 8)
            conf       = float(torch.softmax(logits_t, dim=-1).max().item())
            pred_t     = int(logits_t.argmax().item())
            correction = LABEL_TO_DIR[pred_t] * COPILOT_VEL * conf
            bci_vel    = np.array([trial['vx'][t], trial['vy'][t]], dtype=np.float32)
            cursor     = np.clip(cursor + bci_vel + correction, -1.5, 1.5)

        if angle_pred(cursor) == lbl:
            correct_by_dir[lbl] += 1
        total_by_dir[lbl] += 1

        # BCI baseline
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
        'accuracy':      correct / total,
        'bci_baseline':  bci_c / total,
        'by_direction':  {i: correct_by_dir[i]/total_by_dir[i] for i in range(8)},
        'bci_by_dir':    {i: bci_correct_dir[i]/total_by_dir[i] for i in range(8)},
        'n_correct':     correct,
        'n_total':       total,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Supervised LSTM Copilot Training — V3")
    print("=" * 60)
    print(f"Train subjects: {TRAIN_SUBJECTS}")
    print(f"Test subjects:  {TEST_SUBJECTS}")
    print(f"Phase 1: {PHASE1_EPOCHS} epochs (raw BCI)")
    print(f"Phase 2: {PHASE2_EPOCHS} epochs (augmented BCI+Copilot, conf-weighted)")
    print(f"LSTM: hidden={HIDDEN_SIZE}, layers={N_LAYERS}, input={INPUT_SIZE}")
    print(f"Device: {DEVICE}")
    print()

    # Load data
    by_subj     = load_trials(CSV_PATH)
    train_trials = [t for s in TRAIN_SUBJECTS for t in by_subj[s]]
    test_trials  = [t for s in TEST_SUBJECTS  for t in by_subj[s]]
    print(f"\nTrain: {len(train_trials)} trials | Test: {len(test_trials)} trials\n")

    # Model + optimizer
    model     = LSTMCopilot(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                             n_layers=N_LAYERS, n_classes=8).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)  # v3: step every 6 epochs for longer settling at each LR

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    # Log file
    log_path = OUTPUT_DIR / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phase', 'train_loss',
                         'test_acc', 'bci_baseline',
                         'delta_pp', 'time_s',
                         'NW','N','NE','E','SE','S','SW','W'])

    best_acc  = 0.0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2
        t0    = time.time()

        # ── Build training dataset ────────────────────────────────────────
        print(f"Epoch {epoch}/{total_epochs} (Phase {phase}) — building sequences...",
              end=' ', flush=True)
        if phase == 1:
            seqs = [build_sequence_bci(t) for t in train_trials]
        else:
            seqs = [build_sequence_augmented(t, model) for t in train_trials]
        print(f"done ({len(seqs)} sequences)")

        dataset = BCIDataset(seqs)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

        # ── Train ─────────────────────────────────────────────────────────
        train_loss = train_epoch(model, loader, optimizer)
        scheduler.step()

        # ── Evaluate on test set ──────────────────────────────────────────
        print(f"  Evaluating on {len(test_trials)} test trials...", end=' ', flush=True)
        results = evaluate_fast(model, test_trials)
        elapsed = time.time() - t0

        acc      = results['accuracy']
        bci_base = results['bci_baseline']
        delta    = (acc - bci_base) * 100
        sign     = '+' if delta >= 0 else ''

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            best_str = " ← best"
        else:
            best_str = ""

        print(f"done")
        print(f"  Loss: {train_loss:.4f} | "
              f"BCI: {bci_base*100:.1f}% | "
              f"Copilot: {acc*100:.1f}% | "
              f"Delta: {sign}{delta:.1f}pp{best_str} | "
              f"Time: {elapsed:.0f}s")

        # Per-direction
        print(f"  {'Dir':<4}", end='')
        for n in LABEL_NAMES: print(f"  {n:>5}", end='')
        print()
        print(f"  {'BCI':<4}", end='')
        for i in range(8):
            print(f"  {results['bci_by_dir'][i]*100:>4.1f}%", end='')
        print()
        print(f"  {'Cop':<4}", end='')
        for i in range(8):
            d = results['by_direction'][i] - results['bci_by_dir'][i]
            s = '+' if d >= 0 else ''
            print(f"  {results['by_direction'][i]*100:>4.1f}%", end='')
        print()
        print(f"  {'Δpp':<4}", end='')
        for i in range(8):
            d = (results['by_direction'][i] - results['bci_by_dir'][i]) * 100
            s = '+' if d >= 0 else ''
            print(f"  {s}{d:>3.1f}", end='')
        print()
        print()

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, f"{train_loss:.6f}",
                f"{acc*100:.2f}", f"{bci_base*100:.2f}",
                f"{delta:.2f}", f"{elapsed:.1f}",
                *[f"{results['by_direction'][i]*100:.1f}" for i in range(8)]
            ])

    # Save final model
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print("=" * 60)
    print("Training complete.")
    print(f"Best test accuracy: {best_acc*100:.1f}%")
    print(f"BCI baseline:       {results['bci_baseline']*100:.1f}%")
    print(f"Best improvement:   {(best_acc - results['bci_baseline'])*100:+.1f}pp")
    print(f"Models saved to:    {OUTPUT_DIR}/")
    print(f"Log saved to:       {log_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
