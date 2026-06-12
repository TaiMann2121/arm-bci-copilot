# ARM BCI Copilot

AI copilot for EEG-BCI controlled robotic arm, adapted from Lee et al. (2025) bci_raspy.

## Results
- Best model: 93% trial success rate on 8-direction centre-out task
- Trained with CS=0.75 (calibrated from lab data)

## Training
`python -m SJtools.copilot.train -model PPO -action chargeTargets -action_param temperature 1 -holdtime 2.0 -stillCS 0.0 -timesteps 1200000 -lr_scheduler constant -n_steps 2048 -batch_size 512 -softmax_type normal_target -CS 0.75 -velReplaceSoftmax -no_wandb -reward_type baseLinDist.yaml -center_out_back -history 5 20 pos -historyReset last -extra_targets_yaml lab_dir8.yaml -policy_param_p 64 64 64 -policy_param_v 64 64 64 -filePath ./SJtools/copilot/runs/lab_training/ -fileName LAB_CS0.75_run1`

## Best model
SJtools/copilot/runs/lab_training/LAB_CS0.75_verify2/best_model.zip
