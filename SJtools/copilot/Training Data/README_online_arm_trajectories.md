# Online Arm Trajectories CSV

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
