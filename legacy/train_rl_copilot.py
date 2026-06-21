"""
train_method2.py
================
Method 2 training script — trains the copilot on real CSV trajectories
instead of the parametric surrogate.

Identical to the Method 1 training pipeline EXCEPT:
  - wraps both env and eval_env with RealDataWrapper
  - reward is terminal-only binary (+1/-1) matching angle_pred metric
  - softmax_type='normal_target' (used as a dummy; overwritten by wrapper)

Run from arm-bci-copilot/ :

    # Quick 40k-step diagnostic run
    python train_method2.py -timesteps 40000 -log_interval 1 -no_wandb -dev -fileName diag_realData_run7

    # Full training run (overnight)
    python train_method2.py -timesteps 1200000 -no_wandb -fileName LAB_realData_run7

All other train.py arguments work as normal.

Run history:
  Run1: wrong reward (movement-direction) → N/S overfitting
  Run2: terminal_scale=10 → extreme actions, std 1.0→4.3
  Run3: terminal_scale=2.0, ent_coef=0.01 → std grew monotonically 1.0→3.36,
        full eval 46.3% vs 46.7% baseline (no improvement)
  Run4: terminal-only binary reward (+1/-1), ent_coef=0.0
        → value fn never learned (EV≈0), callback mean 47.8%, full eval 46.8% (+0.1pp)
  Run5: position-based delta_cos*3.0 + terminal*0.1, ent_coef=0.0
        → EV rose to 0.35-0.48, full eval 47.4% (+0.7pp); NE/E/W hurt by Coulomb bias
  Run6: direct (vx,vy) output + counterfactual reward, ent_coef=0.0
        → self._cursor decoupled from obs (state-reward mismatch); full eval 37.2% (-9.5pp)
  Run7: chargeTargets + grounded counterfactual, ent_coef=0.0
        → reads prev_cursor from inner.result[0]; VXY avoided (breaks trial termination)
"""

import sys
import os
import numpy as np
import signal
import time
import argparse
import yaml
import torch
import gymnasium as gym

# ── argument parser (identical to train.py) ───────────────────────────────────
parser = argparse.ArgumentParser(description="Method 2: Real-data copilot training")
parser.add_argument("-model",          type=str,   default="PPO")
parser.add_argument("-timesteps",      type=int,   default=1_200_000)
parser.add_argument("-n_steps",        type=int,   default=2048)
parser.add_argument("-batch_size",     type=int,   default=64)
parser.add_argument("-log_interval",   type=int,   default=4)
parser.add_argument("-wandb",          default=False, action='store_true')
parser.add_argument("-no_wandb",       dest='wandb', action='store_false')
parser.add_argument("-lr",             default=3e-4, type=float)
parser.add_argument("-lr_min",         default=3e-10, type=float)
parser.add_argument("-lr_scheduler",   type=str,   default="constant")
parser.add_argument("-lr_patience",    default=10,  type=int)
parser.add_argument("-holdtime",       default=0.5, type=float)
parser.add_argument("-device",         default='cpu')
parser.add_argument("-save",           default=True, action='store_true')
parser.add_argument("-no_save",        dest='save', action='store_false')
parser.add_argument("-dev",            dest='save', action='store_false')
parser.add_argument("-devSave",        default=False, action='store_true')
parser.add_argument("-fileName",       type=str,   default='')
parser.add_argument("-filePath",       type=str,   default='')
parser.add_argument("-csv_path",       type=str,
                    default="SJtools/copilot/Training Data/online_arm_trajectories.csv")
parser.add_argument("-rng_seed",       type=int,   default=None)

# Fixed Method 2 hyperparameters (not exposed as flags — these are locked in)
# action=['chargeTargets'], history=[5,20,pos], reward=grounded_counterfactual

fullCommand = "python train_method2.py " + " ".join(sys.argv[1:])
args = parser.parse_args()

# ── imports ───────────────────────────────────────────────────────────────────
from SJtools.copilot.env import SJ4DirectionsEnv
from SJtools.copilot.real_data_wrapper import RealDataWrapper
from SJtools.copilot.callbacks import LearningRateCallback, TensorboardLoggerCallback
from SJtools.copilot.angle_accuracy_callback import AngleAccuracyCallback
from SJtools.copilot.trainUtil import fileOrganizer, getDevice
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ── wandb ─────────────────────────────────────────────────────────────────────
wandbUsed = args.wandb and args.save
run_id = ''
if wandbUsed:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    run = wandb.init(project="SJ-4-directions-copilot", entity="aaccjjt",
                     config=vars(args), sync_tensorboard=True)
    run_id = run.id
    def signal_handler(sig, frame):
        wandb.save(myFiles.bestModelPath)
        run.finish()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

# ── file organisation ─────────────────────────────────────────────────────────
myFiles = fileOrganizer(wandbUsed, run_id, no_save=not args.save,
                        fileName=args.fileName, currPath=args.filePath,
                        devSave=args.devSave)
print('\033[91m\033[1m' + fullCommand + '\033[0m')
print(f'\033[1mrun id: {myFiles.runId}\033[0m\n')
myFiles.log(fullCommand)

# ── env factory ───────────────────────────────────────────────────────────────
# Fixed config for Method 2 — do not change these without understanding why
ENV_KWARGS = dict(
    softmax_type      = 'normal_target',  # dummy; overwritten by wrapper each tick
    reward_type       = 'baseLinAngle.yaml',
    action            = ['chargeTargets'],  # Coulomb force — keeps trial termination correct
    action_param      = ['temperature', '1'],  # needed for env init
    historyDim        = [5, 20, 'pos'],
    historyReset      = 'last',
    extra_targets_yaml= 'lab_dir8.yaml',
    center_out_back   = True,
    velReplaceSoftmax = True,
    holdtime          = args.holdtime,
    obs               = [],
    CSvalue           = 0.75,   # unused at runtime; needed for env init
    stillCS           = 0.0,
)

N_EVAL_EPISODES = 16   # 2 trials (center-out-back) × 8 directions

def make_env(isEval=False):
    inner   = SJ4DirectionsEnv(isEval=isEval, **ENV_KWARGS)
    wrapped = RealDataWrapper(inner, csv_path=args.csv_path, rng_seed=args.rng_seed)
    monitored = Monitor(wrapped)
    # TensorboardLoggerCallback accesses self.env.env.taskGame and self.env.env.trialResults
    # Monitor.env → RealDataWrapper → .env → SJ4DirectionsEnv  ✓
    return monitored, wrapped   # wrapped.env is SJ4DirectionsEnv

env,      train_wrapper = make_env(isEval=False)
eval_env, eval_wrapper  = make_env(isEval=True)

# ── callbacks ─────────────────────────────────────────────────────────────────
eval_freq      = args.log_interval * args.n_steps
n_update_per_log = args.log_interval

lrSchedulerInfo = {
    "lr_scheduler": args.lr_scheduler,
    "lr":           args.lr,
    "lr_min":       args.lr_min,
    "lr_patience":  args.lr_patience,
}

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = None,        # disabled — AngleAccuracyCallback handles saving
    log_path             = myFiles.evalCallbackPath,
    eval_freq            = eval_freq,
    n_eval_episodes      = N_EVAL_EPISODES,
    deterministic        = True,
    render               = False,
)

# AngleAccuracyCallback: the real model-saving criterion
# Evaluates angle-based accuracy on 240 held-out trials every 8k steps
# Saves best_model.zip only when angle-accuracy genuinely improves
angle_callback = AngleAccuracyCallback(
    model_save_path = myFiles.best_model_save_path,
    csv_path        = args.csv_path,
    eval_freq       = 8192,
    n_eval_per_dir  = 150,   # 150 × 8 = 1200 trials, ±2.8pp CI — can detect 3pp improvement
    rng_seed        = 42,
    verbose         = 1,
)

lr_callback = LearningRateCallback(
    eval_callback, n_update_per_log, lrSchedulerInfo, args.timesteps
)

tb_callback = TensorboardLoggerCallback(
    env=train_wrapper,      # RealDataWrapper: .env → SJ4DirectionsEnv with taskGame/trialResults
    eval_env=eval_wrapper,  # RealDataWrapper: .env → SJ4DirectionsEnv with taskGame/trialResults
    n_update_per_log=n_update_per_log,
    n_eval_episodes=N_EVAL_EPISODES,
)

# ── model ─────────────────────────────────────────────────────────────────────
net_arch = [{'pi': [64, 64, 64], 'vf': [64, 64, 64]}]   # same as Method 1

model = PPO(
    'MlpPolicy', env,
    n_steps       = args.n_steps,
    learning_rate = args.lr,
    batch_size    = args.batch_size,
    verbose       = 1,
    tensorboard_log = myFiles.tensorboard_log,
    policy_kwargs = {'net_arch': net_arch},
    device        = args.device,
    ent_coef      = 0.0,   # no entropy bonus — keeps std stable (confirmed Run4/5)
)

# ── save model yaml ───────────────────────────────────────────────────────────
inner_train_env = train_wrapper.env   # SJ4DirectionsEnv
yamlcontent = {
    "copilot": {
        "model":    "PPO",
        "policy":   None,
        "net_arch": net_arch,
    },
    "targets": {
        "extra_targets":      {},
        "extra_targets_yaml": 'lab_dir8.yaml',
    },
    "method": "2_real_data",
    "csv_path": args.csv_path,
}
yamlcontent.update(inner_train_env.copilotYamlParam)
myFiles.saveYaml(yamlcontent)
myFiles.saveRewardYaml(inner_train_env.rewardClass.fullYamlPath)

# ── train ─────────────────────────────────────────────────────────────────────
callback = [eval_callback, lr_callback, tb_callback, angle_callback]
if wandbUsed:
    callback.append(WandbCallback())

model.learn(
    total_timesteps = args.timesteps,
    log_interval    = args.log_interval,
    callback        = callback,
)

# ── save ──────────────────────────────────────────────────────────────────────
if args.save:
    model.save(myFiles.lastModelPath)

print(fullCommand)
print(f'run id: {myFiles.runId}')
print("Done")
