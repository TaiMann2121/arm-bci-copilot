"""
debug_trial.py  —  run ONE trial with verbose tick-by-tick output
to diagnose why evaluate_method1_on_real_data gets 0% success.

Run from arm-bci-copilot/ :
    python3 debug_trial.py
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

CSV_PATH   = "SJtools/copilot/Training Data/online_arm_trajectories.csv"
MODEL_PATH = "SJtools/copilot/runs/lab_training/LAB_CS0.75_verify2/best_model.zip"

from SJtools.copilot.env import SJ4DirectionsEnv, init_lab_surrogate

RADIUS_PX = 432.0
N_STATE   = 5

LABEL_TO_POS = {
    0: np.array([-0.707,  0.707]), 1: np.array([ 0.000,  1.000]),
    2: np.array([ 0.707,  0.707]), 3: np.array([ 1.000,  0.000]),
    4: np.array([ 0.707, -0.707]), 5: np.array([ 0.000, -1.000]),
    6: np.array([-0.707, -0.707]), 7: np.array([-1.000,  0.000]),
}
LABEL_TO_NAME = {0:'nw',1:'n',2:'ne',3:'e',4:'se',5:'s',6:'sw',7:'w'}

def vel_to_softmax(vx, vy):
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 1e-6:
        vx, vy = vx/speed, vy/speed
    s = np.zeros(N_STATE, dtype=np.float32)
    if vx > 0: s[1] = vx
    else:       s[0] = -vx
    if vy > 0: s[2] = vy
    else:       s[3] = -vy
    return s

# ── load one trial (E direction, first subject) ───────────────────────────────
df = pd.read_csv(CSV_PATH)
group_cols = ['subject_id','session_number','run_number','trial_number','inner_trial_number']
target_trial = None
for keys, grp in df.groupby(group_cols):
    if int(grp['target_label'].iloc[0]) == 3:  # East
        target_trial = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        break

cx = target_trial['cursor_pos_x'].values / RADIUS_PX
cy = target_trial['cursor_pos_y'].values / RADIUS_PX
vx = np.diff(cx); vy = np.diff(cy)
softmax_seq = [vel_to_softmax(vx[i], vy[i]) for i in range(len(vx))]
print(f"Trial: {dict(zip(group_cols, target_trial[group_cols].iloc[0]))}")
print(f"Target: E (label 3), softmax_seq length: {len(softmax_seq)}")
print()

# ── build env ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = PPO.load(MODEL_PATH)
print("Loading env...")
init_lab_surrogate(CSV_PATH)
env = SJ4DirectionsEnv(
    isEval=True,
    softmax_type='normal_target',
    reward_type='baseLinDist.yaml',
    holdtime=0.5,
    center_out_back=True,
    velReplaceSoftmax=True,
    action=['chargeTargets'],
    action_param=['temperature', '1'],
    historyDim=[5, 20, 'pos'],
    historyReset='last',
    extra_targets_yaml='lab_dir8.yaml',
    CSvalue=0.75,
    stillCS=0.0,
    obs=[],
)
print(f"Action space: {env.action_space.shape}")
print(f"Obs space:    {env.observation_space.shape}")
print()

# ── run one trial ─────────────────────────────────────────────────────────────
env.taskGame.nextTargetBucket = ['e']
obs, _ = env.reset()
env.taskGame.cursorPos = np.zeros(2)
env.taskGame.secondCursorPos = np.zeros(2)
env.taskGame.resetNextCursorPos = False
if env.taskGame.nextTarget != 'e':
    env.taskGame.nextTarget = 'e'

print(f"After reset — nextTarget: {env.taskGame.nextTarget}  cursorPos: {env.taskGame.cursorPos}")
print(f"holdTimeThres: {env.taskGame.holdTimeThres}")
print()
print(f"{'tick':>4}  {'cursor_x':>9} {'cursor_y':>9} {'dist_to_tgt':>12} {'game_state':>12} {'done':>5} {'reward':>8}")
print("-" * 65)

episode_start = True
_states = None
target_pos = LABEL_TO_POS[3]  # East

for tick in range(64):
    real_softmax = softmax_seq[tick] if tick < len(softmax_seq) else np.zeros(N_STATE, dtype=np.float32)
    env.softmax = real_softmax

    action, _states = model.predict(obs, state=_states, deterministic=True, episode_start=episode_start)
    episode_start = False

    obs, reward, done, truncated, info = env.step(action)

    next_softmax = softmax_seq[tick+1] if tick+1 < len(softmax_seq) else np.zeros(N_STATE, dtype=np.float32)
    env.softmax = next_softmax

    result = info.get('result', [])
    cursor_pos = result[0] if len(result) > 0 else np.array([0,0])
    game_state = result[4] if len(result) > 4 else '?'
    dist = np.linalg.norm(cursor_pos - target_pos)

    print(f"{tick:>4}  {cursor_pos[0]:>9.4f} {cursor_pos[1]:>9.4f} {dist:>12.4f} {game_state:>12}  {str(done):>5} {reward:>8.4f}")

    if game_state == 'H':
        print("\n✓ SUCCESS")
        break
    elif isinstance(game_state, str) and game_state.isupper():
        print(f"\n✗ TERMINAL: game_state='{game_state}'")
        break
    if done:
        print(f"\n✗ DONE (no terminal game_state seen)")
        break

env.close()
