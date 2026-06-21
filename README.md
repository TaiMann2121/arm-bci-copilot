# arm-bci-copilot

EEG-BCI robotic arm copilot — supervised LSTM approach for 8-direction arm motor imagery.

## Result

**+1.07pp** improvement over raw BCI decoder (46.70% → 47.77%) on all 7 subjects.  
Evaluated on the lab's `arm_prediction_label` angular classification metric.

| Subject | BCI Baseline | Copilot | Delta |
|---------|-------------|---------|-------|
| S01 (train) | 39.3% | 39.8% | +0.4pp |
| S02 (train) | 57.4% | 59.2% | +1.8pp |
| S03 (train) | 49.9% | 50.0% | +0.1pp |
| S04 (train) | 55.1% | 57.0% | +1.8pp |
| S05 (train) | 43.3% | 44.3% | +1.0pp |
| S06 (test)  | 34.2% | 35.2% | +1.1pp |
| S07 (test)  | 47.7% | 48.9% | +1.2pp |
| **Overall** | **46.7%** | **47.8%** | **+1.07pp** |

Train/test gap: 0.12pp — no meaningful overfitting across subjects.

## Architecture

Two-stage supervised copilot, trained on `online_arm_trajectories.csv`:

**Stage 1 — LSTM target classifier**  
Input per tick: `(cursor_x, cursor_y, vx_unit, vy_unit, vel_mag_scaled)` — 5 features  
Model: 2-layer LSTM, hidden size 64, ~52k parameters  
Output: probability distribution over 8 target directions

**Stage 2 — Confidence-weighted corrective velocity**  
```
correction = LABEL_TO_DIR[pred] × COPILOT_VEL × confidence
confidence = max(softmax(logits))  ∈ [0.125, 1.0]
final_cursor += bci_vel + correction        # COPILOT_VEL = 0.02
```

## Repository Structure

```
arm-bci-copilot/
├── data/
│   ├── online_arm_trajectories.csv          # 7 subjects, 11,760 trials, 8 directions
│   └── README_online_arm_trajectories.md
├── supervised_copilot/
│   └── best_model.pt                         # trained LSTM weights (V3)
├── train_supervised_copilot.py               # training script
├── evaluate_supervised_copilot.py            # full evaluation (all 7 subjects)
├── visualize_copilot.py                      # interactive HTML trajectory visualization
├── requirements.txt
├── README.md
└── legacy/                                   # Lee et al. RL copilot (Run1–Run7)
    ├── README_legacy.md
    ├── SJtools/copilot/                       # RL environment, wrapper, callbacks
    ├── modules/                               # game engine (kf_4_directions etc.)
    ├── asset/
    ├── train_rl_copilot.py                   # RL training (Run5 best: +0.74pp)
    └── evaluate_rl_copilot.py
```

## Usage

### Training
```bash
python train_supervised_copilot.py
# Output: supervised_copilot/best_model.pt
#         supervised_copilot/training_log.csv
```

### Evaluation (all 7 subjects)
```bash
python evaluate_supervised_copilot.py
# Optional: --model path/to/model.pt
```

### Visualization
```bash
python visualize_copilot.py
# Output: copilot_visualization.html  (open in browser)
```

## Training Details

- **Phase 1** (3 epochs): raw BCI-only trajectories from CSV
- **Phase 2** (25 epochs): DAgger-augmented BCI+Copilot trajectories
- **Loss**: cross-entropy with linear tick weighting — tick t gets weight (t+1)/T
- **Optimizer**: Adam, LR 1e-3, StepLR (step_size=6, γ=0.5)
- **Train/test split**: S01–S05 train, S06–S07 test (subject-level)

## Data

`online_arm_trajectories.csv` — 199,911 rows, 7 subjects, 8 directions, 8 Hz.  
See `data/README_online_arm_trajectories.md` for full schema.
