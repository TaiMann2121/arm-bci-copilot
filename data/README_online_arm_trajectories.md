# Online Arm Trajectories CSV

Last updated on Jun. 22, 2026

Online cursor trajectories generated using the newer EEGK decoder are stored under `/OnlineArmTrajectoryEEGK`. Data from each session are saved separately, together with session-specific decoding statistics (`typing_stats.npz`) derived from its calibration run (run 1). Because decoder performance is expected to be similar across sessions from the same subject, you may either use these session-specific saved statistics or estimate them directly from the online trajectory data. Run 1 in each exported folder is label-balanced and is therefore a suitable source for estimating these statistics.

`online_arm_trajectories.csv` contains the exported online cursor trajectory for
each arm trial. 

## Row Structure

Each row is one saved trajectory sample. A single trajectory is identified by:

```text
subject_id, session_number, run_number, trial_number, inner_trial_number
```

Rows within a trajectory are ordered by `timestamp_seconds`. The original BCI samples are downsampled by a factor of `128`. For a source sampling rate of `1024 Hz`, the CSV therefore has one row every `0.125` seconds. The arm trial duration is `2` seconds.

## Columns

| Column | Description |
| --- | --- |
| `subject_id` | Subject identifier, such as `S01`. |
| `session_number` | Session folder index: `0` or `1`. |
| `run_number` | Global run number across sessions. Run number starts from `2` as run `1` served as calibration runs. |
| `trial_number` | Outer trial number (Block number) within the run. |
| `inner_trial_number` | Arm trial number within the outer trial. |
| `timestamp_seconds` | Time in seconds relative to the arm trial onset. |
| `cursor_pos_x` | Cursor x-coordinate. |
| `cursor_pos_y` | Cursor y-coordinate. |
| `target_label` | Arm target label. Expected labels are `0-7`. |
| `target_pos_x` | Target x-coordinate on the eight-direction target circle. |
| `target_pos_y` | Target y-coordinate on the eight-direction target circle. |
| `arm_prediction_label` | Online arm prediction label at this timestamp. |

## Target Positions

Targets lie on a circle with radius `432`:

| Target label | Direction | Position |
| --- | --- | --- |
| `0` | northwest | `(-305.47, 305.47)` |
| `1` | north | `(0, 432)` |
| `2` | northeast | `(305.47, 305.47)` |
| `3` | east | `(432, 0)` |
| `4` | southeast | `(305.47, -305.47)` |
| `5` | south | `(0, -432)` |
| `6` | southwest | `(-305.47, -305.47)` |
| `7` | west | `(-432, 0)` |


## `typing_stats.npz` Structure (available only for the EEGK results)

| Array | Shape | Description |
| --- | --- | --- |
| `arm_confusion` | `(8, 8)` | Row-normalized arm confusion matrix. Rows are true arm targets (`0-7`) and columns are predicted arm labels (`0-7`), so element `[target, prediction]` estimates `P(prediction \| target)`. |
| `angle_sigma_rad` | scalar | Robust scale of the final cursor-angle errors, expressed in radians. It is used as the standard-deviation-like parameter of the Gaussian arm-angle likelihood during EEG decoding. |
| `arm_pred_stats` | `(n_trials,)` | Predicted arm label for each calibration trial. Values are expected to be `0-7`. |
| `arm_target_stats` | `(n_trials,)` | True arm target label for each calibration trial. Values are expected to be `0-7`. |
| `arm_angle_error_deg` | `(n_trials,)` | Absolute wrapped angular error between the final cursor direction and target direction, in degrees. Values are nonnegative and normally lie in `[0, 180]`. |
| `arm_angle_diff_deg` | `(n_trials,)` | Signed wrapped angular difference between the final cursor direction and target direction, in degrees, normally in `[-180, 180)`. |
| `finger_confusion` | `(3, 3)` | Row-normalized finger confusion matrix. Rows are true finger targets (`0-2`) and columns are predicted finger labels (`0-2`), so element `[target, prediction]` estimates `P(prediction \| target)`. |
| `finger_pred_stats` | `(n_trials,)` | Predicted finger label for each calibration trial. Values are expected to be `0-2`. |
| `finger_target_stats` | `(n_trials,)` | True finger target label for each calibration trial. Values are expected to be `0-2`. |
| `n_trials` | scalar | Number of saved arm calibration trials |

