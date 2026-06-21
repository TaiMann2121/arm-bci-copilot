# Legacy: Lee et al. RL Copilot Experiments (Run1–Run7)

This folder contains the infrastructure from the initial RL-based copilot approach, adapted from Lee et al. (2023). It is preserved for reference but is not the current approach.

## Summary of RL Experiments

| Run | Architecture | Reward | Result | Notes |
|-----|-------------|--------|--------|-------|
| Run3 | chargeTargets | delta_cos×5 + terminal×2 | +0.0pp | std grew 1.0→3.36 |
| Run4 | chargeTargets | terminal-only ±1 | +0.1pp | value fn never learned (EV≈0) |
| Run5 | chargeTargets | delta_cos×3 + terminal×0.1 | **+0.74pp** | Best RL result; NE/E/W hurt |
| Run6 | Direct (vx,vy) | Counterfactual | −9.5pp | State-reward mismatch; cursor diverged |
| Run7 | chargeTargets | Grounded counterfactual | −23pp | Reward hacking; policy collapsed |

**Best RL result: Run5, +0.74pp** (BCI 46.7% → 47.4%)  
**Best supervised result: V3 LSTM, +1.07pp** (BCI 46.7% → 47.8%)

## Why RL Was Abandoned

Three persistent failure modes:

1. **Reward hacking**: Per-tick reward signals (both delta-cos and counterfactual) were consistently gamed by the chargeTargets policy. The Coulomb force mechanism has enough degrees of freedom to optimize the per-tick metric without improving final angular classification.

2. **Directional bias**: The chargeTargets Coulomb mechanism is position-dependent. The same policy output produces different physical forces depending on cursor position, creating a feedback loop that systematically pushed N/S at the expense of NE/E/W.

3. **Architectural mismatch**: Lee et al. designed the chargeTargets mechanism for a 4-direction task with a parametric surrogate. Our task has 8 directions and real recorded trajectories, making the RL loop structurally poorly suited.

## Model Path (Run5)

```
SJtools/copilot/runs/LAB_realData_run5/best_model.zip
```

## Running RL Training

```bash
python train_rl_copilot.py -timesteps 1200000 -no_wandb -fileName LAB_realData_run8
```

## Running RL Evaluation

```bash
python evaluate_rl_copilot.py
```
