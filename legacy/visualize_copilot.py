"""
visualize_copilot.py
====================
Generates an interactive HTML visualization comparing BCI-only vs
BCI+Copilot cursor trajectories for all 8 target directions.

For each direction panel:
  - Grey lines:   20 randomly sampled individual BCI-only trajectories

NOTE ON ACCURACY NUMBERS
------------------------
The per-direction accuracy shown in annotations is computed with each trial
starting fresh (prev_cursor reset to zeros). This differs slightly from
evaluate.py which carries cursor history across trials within a subject.
The visualization numbers are cleaner for per-direction comparison; the
evaluate.py numbers (overall +0.7pp) are the official result.
  - Blue line:    Mean BCI-only trajectory (computed over all 1,470 trials)
  - Orange line:  Mean BCI+Copilot trajectory (Run5 model, all 1,470 trials)
  - Blue dot:     Mean BCI-only final cursor position
  - Orange dot:   Mean BCI+Copilot final cursor position
  - Wedge shading: 45° angular classification zone for that direction
  - Accuracy annotations: baseline % and copilot % for that direction

Run from arm-bci-copilot/:
    python visualize_copilot.py

Output: copilot_visualization.html  (open in any browser)
"""

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "SJtools/copilot/Training Data/online_arm_trajectories.csv"
MODEL_PATH = "SJtools/copilot/runs/LAB_realData_run5/best_model.zip"
OUTPUT_PATH = "copilot_visualization.html"

N_SAMPLE_TRACES = 20   # individual BCI traces to show per panel
RANDOM_SEED     = 42
RADIUS_PX       = 432.0

# ── copilot constants (Run5 — chargeTargets) ──────────────────────────────────
COPILOT_VEL = 0.02
N_HIST      = 5
HIST_INTV   = 20
TEMPERATURE = 1.0
EPS         = 0.01
K           = 1.0
POWER       = 2.0

# ── geometry ──────────────────────────────────────────────────────────────────
LABEL_TO_DIR = np.array([
    [-0.707,  0.707],  # 0  NW
    [ 0.000,  1.000],  # 1  N
    [ 0.707,  0.707],  # 2  NE
    [ 1.000,  0.000],  # 3  E
    [ 0.707, -0.707],  # 4  SE
    [ 0.000, -1.000],  # 5  S
    [-0.707, -0.707],  # 6  SW
    [-1.000,  0.000],  # 7  W
], dtype=np.float32)

LABEL_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

_all_targets   = LABEL_TO_DIR.tolist() + [[0.0, 0.0]]
TARGETS_SORTED = np.array(
    sorted(_all_targets, key=lambda p: (p[0], p[1])), dtype=np.float32
)

# Run5 full-eval per-direction results (from evaluate.py output)
COPILOT_ACC  = [50.3, 64.8, 38.8, 39.7, 47.6, 51.0, 43.3, 43.3]
BASELINE_ACC = [47.3, 60.2, 41.6, 43.3, 47.1, 46.3, 41.6, 45.0]


# ── simulation helpers ────────────────────────────────────────────────────────

def vel_to_softmax(vx, vy):
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 1e-6:
        vx /= speed; vy /= speed
    s = np.zeros(5, dtype=np.float32)
    s[1] = max(vx, 0.0); s[0] = max(-vx, 0.0)
    s[2] = max(vy, 0.0); s[3] = max(-vy, 0.0)
    return s

def softmax_to_vel(s):
    return np.clip(np.array([s[1]-s[0], s[2]-s[3]], dtype=np.float32), -1, 1)

def calc_copilot_vel(copilot_output, cursor_pos):
    charges = (copilot_output + 1.0) / 2.0
    charges = torch.softmax(
        torch.tensor(charges / TEMPERATURE, dtype=torch.float32), dim=0
    ).numpy()
    diffs = TARGETS_SORTED - cursor_pos
    dists = np.linalg.norm(diffs, axis=1)
    mag   = K * charges / (dists**POWER + EPS)
    return (mag @ diffs) * COPILOT_VEL

class HistoryQueue:
    def __init__(self):
        self.n = (N_HIST - 1) * HIST_INTV + 1
        self.data = np.zeros((self.n, 2), dtype=np.float32)
        self.i = 0
    def reset(self, last_pos=None):
        self.data[:] = last_pos if last_pos is not None else 0.0
        self.i = 0
    def add_get(self, pos):
        self.i = (self.i + 1) % self.n
        self.data[self.i] = pos
        return self.data[self.i - np.arange(N_HIST) * HIST_INTV]

def build_obs(cursor_pos, vel, hq):
    base = np.concatenate([cursor_pos, vel]).astype(np.float32)
    hist = hq.add_get(cursor_pos)
    return np.concatenate([base, hist[1:].flatten()]).astype(np.float32)

def angle_pred(cursor):
    norm = np.linalg.norm(cursor)
    if norm < 1e-6: return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor / norm)))


# ── data loading ──────────────────────────────────────────────────────────────

def load_trials(csv_path):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    group_cols = ['subject_id','session_number','run_number',
                  'trial_number','inner_trial_number']
    trials_by_dir = {i: [] for i in range(8)}
    for _, grp in df.groupby(group_cols):
        grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl  = int(grp['target_label'].iloc[0])
        if lbl not in range(8): continue
        cx   = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy   = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx   = np.diff(cx); vy = np.diff(cy)
        if len(vx) == 0: continue
        trials_by_dir[lbl].append({
            'vel_seq':    list(zip(vx.tolist(), vy.tolist())),
            'cursor_seq': list(zip(cx.tolist(), cy.tolist())),
        })
    for lbl in range(8):
        print(f"  Dir {LABEL_NAMES[lbl]}: {len(trials_by_dir[lbl])} trials")
    return trials_by_dir


# ── trajectory simulation ─────────────────────────────────────────────────────

def simulate_trial_bci_only(trial):
    """Replay BCI trajectory starting at (0,0). Returns array (T+1, 2)."""
    cursor = np.zeros(2, dtype=np.float32)
    path   = [cursor.copy()]
    for vx, vy in trial['vel_seq']:
        cursor = cursor + np.array([vx, vy], dtype=np.float32)
        cursor = np.clip(cursor, -1.5, 1.5)
        path.append(cursor.copy())
    return np.array(path)

def simulate_trial_with_copilot(model, trial, hq, prev_cursor):
    """Run copilot simulation. Returns array (T+1, 2)."""
    hq.reset(last_pos=prev_cursor)
    cursor       = np.zeros(2, dtype=np.float32)
    path         = [cursor.copy()]
    ep_start     = True
    _states      = None
    for vx, vy in trial['vel_seq']:
        enc_vel = softmax_to_vel(vel_to_softmax(vx, vy))
        obs     = build_obs(cursor, enc_vel, hq)
        action, _states = model.predict(
            obs, state=_states, deterministic=True, episode_start=ep_start
        )
        ep_start  = False
        real_vel  = np.array([vx, vy], dtype=np.float32)
        cop_vel   = calc_copilot_vel(action, cursor)
        cursor    = np.clip(cursor + real_vel + cop_vel, -1.5, 1.5)
        path.append(cursor.copy())
    return np.array(path)

def compute_trajectories(model, trials_by_dir):
    """
    For each direction:
      - Compute mean BCI trajectory over all trials
      - Compute mean copilot trajectory over all trials
      - Sample N_SAMPLE_TRACES individual BCI trajectories
    """
    rng = random.Random(RANDOM_SEED)
    hq  = HistoryQueue()
    results = {}

    for lbl in range(8):
        name   = LABEL_NAMES[lbl]
        trials = trials_by_dir[lbl]
        print(f"  Simulating {name} ({len(trials)} trials)...", end=' ', flush=True)

        # All BCI trajectories (for mean)
        bci_paths = [simulate_trial_bci_only(t) for t in trials]

        # All copilot trajectories (for mean)
        # Reset prev_cursor each trial for clean simulation (no cross-trial history)
        cop_paths = []
        for t in trials:
            path = simulate_trial_with_copilot(model, t, hq, np.zeros(2, dtype=np.float32))
            cop_paths.append(path)

        # Pad all paths to same length
        max_len = max(len(p) for p in bci_paths)
        def pad(paths):
            out = []
            for p in paths:
                if len(p) < max_len:
                    p = np.concatenate([p, np.repeat(p[-1:], max_len-len(p), axis=0)])
                out.append(p)
            return np.stack(out)   # (N, T, 2)

        bci_arr = pad(bci_paths)
        cop_arr = pad(cop_paths)

        mean_bci = bci_arr.mean(axis=0)  # (T, 2)
        mean_cop = cop_arr.mean(axis=0)  # (T, 2)

        # Sample individual BCI traces for display
        sample_idx  = rng.sample(range(len(trials)), min(N_SAMPLE_TRACES, len(trials)))
        sample_bci  = [bci_paths[i] for i in sample_idx]
        sample_info = []
        for i in sample_idx:
            final = bci_paths[i][-1]
            pred  = angle_pred(final)
            sample_info.append({
                'path':    bci_paths[i],
                'correct': pred == lbl,
                'pred':    LABEL_NAMES[pred] if pred >= 0 else '?',
            })

        # Accuracy stats
        bci_correct = sum(angle_pred(p[-1]) == lbl for p in bci_paths)
        cop_correct = sum(angle_pred(p[-1]) == lbl for p in cop_paths)

        results[lbl] = {
            'mean_bci':    mean_bci,
            'mean_cop':    mean_cop,
            'sample_info': sample_info,
            'bci_acc':     bci_correct / len(trials),
            'cop_acc':     cop_correct / len(trials),
            'n_trials':    len(trials),
        }
        print(f"BCI={bci_correct/len(trials)*100:.1f}% → Copilot={cop_correct/len(trials)*100:.1f}%")

    return results


# ── wedge geometry ────────────────────────────────────────────────────────────

def wedge_polygon(center_angle_deg, half_width_deg=22.5, r=1.0, n_pts=30):
    """Return (x, y) arrays for a wedge polygon."""
    angles = np.linspace(
        np.radians(center_angle_deg - half_width_deg),
        np.radians(center_angle_deg + half_width_deg),
        n_pts
    )
    xs = np.concatenate([[0], r * np.cos(angles), [0]])
    ys = np.concatenate([[0], r * np.sin(angles), [0]])
    return xs, ys

def dir_to_angle_deg(dir_vec):
    """Convert (x, y) unit direction to degrees (standard math convention)."""
    return np.degrees(np.arctan2(dir_vec[1], dir_vec[0]))


# ── HTML visualization ────────────────────────────────────────────────────────

def build_html(results):
    """Build interactive Plotly figure with 8 direction panels."""

    # Layout: 2 rows × 4 columns
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{LABEL_NAMES[lbl]}" for lbl in range(8)],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    # Color palette
    COLOR_BCI_SAMPLE  = 'rgba(150, 170, 200, 0.35)'   # muted blue-grey, transparent
    COLOR_BCI_MEAN    = '#2563EB'                        # strong blue
    COLOR_COP_MEAN    = '#EA580C'                        # orange
    COLOR_WEDGE       = 'rgba(250, 250, 210, 0.55)'     # soft yellow
    COLOR_WEDGE_LINE  = 'rgba(200, 200, 150, 0.8)'

    panel_order = [0, 1, 2, 3, 4, 5, 6, 7]  # NW N NE E SE S SW W
    shown_legend = set()

    for panel_idx, lbl in enumerate(panel_order):
        row = panel_idx // 4 + 1
        col = panel_idx % 4 + 1
        res = results[lbl]
        dir_vec = LABEL_TO_DIR[lbl]
        center_deg = dir_to_angle_deg(dir_vec)
        name = LABEL_NAMES[lbl]

        # ── Wedge background ───────────────────────────────────────────────
        wx, wy = wedge_polygon(center_deg, half_width_deg=22.5, r=0.95)
        fig.add_trace(go.Scatter(
            x=wx, y=wy, fill='toself',
            fillcolor=COLOR_WEDGE,
            line=dict(color=COLOR_WEDGE_LINE, width=1),
            mode='lines', hoverinfo='skip',
            showlegend=False, name='_wedge',
        ), row=row, col=col)

        # ── Target direction arrow ─────────────────────────────────────────
        fig.add_annotation(
            x=dir_vec[0]*0.85, y=dir_vec[1]*0.85,
            ax=0, ay=0, xref=f'x{panel_idx+1 if panel_idx>0 else ""}',
            yref=f'y{panel_idx+1 if panel_idx>0 else ""}',
            axref=f'x{panel_idx+1 if panel_idx>0 else ""}',
            ayref=f'y{panel_idx+1 if panel_idx>0 else ""}',
            arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor='rgba(100,120,160,0.6)',
            showarrow=True, text='',
        )

        # ── Individual BCI sample traces ───────────────────────────────────
        for i, info in enumerate(res['sample_info']):
            path   = info['path']
            show_l = ('BCI samples' not in shown_legend)
            if show_l: shown_legend.add('BCI samples')
            fig.add_trace(go.Scatter(
                x=path[:, 0], y=path[:, 1],
                mode='lines',
                line=dict(color=COLOR_BCI_SAMPLE, width=1),
                showlegend=show_l,
                legendgroup='bci_samples',
                name='BCI samples',
                hovertemplate=(
                    f"<b>BCI sample</b><br>"
                    f"Predicted: {info['pred']}<br>"
                    f"{'✓ Correct' if info['correct'] else '✗ Incorrect'}"
                    "<extra></extra>"
                ),
            ), row=row, col=col)

        # ── Mean BCI trajectory ────────────────────────────────────────────
        bci = res['mean_bci']
        show_l = ('Mean BCI' not in shown_legend)
        if show_l: shown_legend.add('Mean BCI')
        fig.add_trace(go.Scatter(
            x=bci[:, 0], y=bci[:, 1],
            mode='lines+markers',
            line=dict(color=COLOR_BCI_MEAN, width=2.5),
            marker=dict(size=[3]*len(bci) + [0], color=COLOR_BCI_MEAN),
            showlegend=show_l,
            legendgroup='mean_bci',
            name='Mean BCI',
            hovertemplate="<b>Mean BCI</b><br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        ), row=row, col=col)

        # ── Mean BCI endpoint ──────────────────────────────────────────────
        bci_end_pred = angle_pred(bci[-1])
        bci_end_correct = bci_end_pred == lbl
        fig.add_trace(go.Scatter(
            x=[bci[-1, 0]], y=[bci[-1, 1]],
            mode='markers',
            marker=dict(
                size=12, color=COLOR_BCI_MEAN,
                symbol='circle',
                line=dict(color='white', width=1.5)
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>BCI endpoint</b><br>"
                f"x={bci[-1,0]:.3f}, y={bci[-1,1]:.3f}<br>"
                f"Predicted: {LABEL_NAMES[bci_end_pred] if bci_end_pred>=0 else '?'}<br>"
                f"{'✓ Correct' if bci_end_correct else '✗ Incorrect'}<br>"
                f"Accuracy: {res['bci_acc']*100:.1f}%"
                "<extra></extra>"
            ),
        ), row=row, col=col)

        # ── Mean Copilot trajectory ────────────────────────────────────────
        cop = res['mean_cop']
        show_l = ('Mean BCI+Copilot' not in shown_legend)
        if show_l: shown_legend.add('Mean BCI+Copilot')
        fig.add_trace(go.Scatter(
            x=cop[:, 0], y=cop[:, 1],
            mode='lines+markers',
            line=dict(color=COLOR_COP_MEAN, width=2.5),
            marker=dict(size=[3]*len(cop) + [0], color=COLOR_COP_MEAN),
            showlegend=show_l,
            legendgroup='mean_cop',
            name='Mean BCI+Copilot',
            hovertemplate="<b>Mean BCI+Copilot</b><br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        ), row=row, col=col)

        # ── Mean Copilot endpoint ──────────────────────────────────────────
        cop_end_pred    = angle_pred(cop[-1])
        cop_end_correct = cop_end_pred == lbl
        fig.add_trace(go.Scatter(
            x=[cop[-1, 0]], y=[cop[-1, 1]],
            mode='markers',
            marker=dict(
                size=12, color=COLOR_COP_MEAN,
                symbol='diamond',
                line=dict(color='white', width=1.5)
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>Copilot endpoint</b><br>"
                f"x={cop[-1,0]:.3f}, y={cop[-1,1]:.3f}<br>"
                f"Predicted: {LABEL_NAMES[cop_end_pred] if cop_end_pred>=0 else '?'}<br>"
                f"{'✓ Correct' if cop_end_correct else '✗ Incorrect'}<br>"
                f"Accuracy: {res['cop_acc']*100:.1f}%"
                "<extra></extra>"
            ),
        ), row=row, col=col)

        # ── Origin marker ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=7, color='#6B7280', symbol='cross'),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        # ── Accuracy annotation box ────────────────────────────────────────
        delta     = res['cop_acc'] - res['bci_acc']
        sign      = '+' if delta >= 0 else ''
        color_ann = '#16a34a' if delta >= 0 else '#dc2626'
        ann_text  = (
            f"BCI: {res['bci_acc']*100:.1f}%  "
            f"Copilot: {res['cop_acc']*100:.1f}%  "
            f"<span style='color:{color_ann}'>{sign}{delta*100:.1f}pp</span>"
        )

    # ── Global layout ──────────────────────────────────────────────────────
    axis_range = [-0.80, 0.80]
    axis_cfg   = dict(
        range=axis_range, showgrid=True, gridcolor='rgba(200,200,200,0.4)',
        zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1,
        showticklabels=False, scaleanchor='y', scaleratio=1,
    )

    layout_updates = {}
    for i in range(1, 9):
        suffix = str(i) if i > 1 else ''
        layout_updates[f'xaxis{suffix}'] = dict(**axis_cfg, title='')
        layout_updates[f'yaxis{suffix}'] = dict(**axis_cfg, title='')

    # Per-panel accuracy annotations (added as layout annotations)
    acc_annotations = []
    for panel_idx, lbl in enumerate(panel_order):
        row    = panel_idx // 4 + 1
        col    = panel_idx % 4 + 1
        res    = results[lbl]
        delta  = res['cop_acc'] - res['bci_acc']
        sign   = '+' if delta >= 0 else ''
        color  = '#16a34a' if delta >= 0 else '#dc2626'
        # Position in paper coords (approximate)
        x_frac = (col - 1) / 4 + 0.5 / 4
        y_frac = 1.0 - (row - 1) / 2 - 0.03

        acc_annotations.append(dict(
            text=(
                f"BCI {res['bci_acc']*100:.1f}%  →  "
                f"Copilot {res['cop_acc']*100:.1f}%  "
                f"<b><span style='color:{color}'>{sign}{delta*100:.1f}pp</span></b>"
            ),
            xref='paper', yref='paper',
            x=x_frac, y=y_frac,
            showarrow=False,
            font=dict(size=10, family='monospace'),
            align='center',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(200,200,200,0.6)',
            borderwidth=1,
            borderpad=3,
        ))

    fig.update_layout(
        **layout_updates,
        title=dict(
            text=(
                '<b>BCI Copilot — Cursor Trajectory Visualization</b><br>'
                '<sup>Run5 model (chargeTargets, ent_coef=0.0) · '
                'Overall: BCI 46.7% → Copilot 47.4% (+0.7pp) · '
                '1,470 trials per direction</sup>'
            ),
            x=0.5, xanchor='center',
            font=dict(size=16, family='Arial'),
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.06,
            xanchor='center', x=0.5,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(200,200,200,0.5)',
            borderwidth=1,
        ),
        annotations=acc_annotations,
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        height=780,
        width=1400,
        margin=dict(t=80, b=80, l=20, r=20),
        hovermode='closest',
    )

    # Ensure equal axes on all subplots
    for i in range(1, 9):
        suffix = str(i) if i > 1 else ''
        fig.update_layout(**{
            f'xaxis{suffix}': dict(range=axis_range, constrain='domain'),
            f'yaxis{suffix}': dict(range=axis_range, scaleanchor=f'x{suffix}', scaleratio=1),
        })

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BCI Copilot Visualization")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {CSV_PATH}")
    print()

    # Load model
    print("Loading Run5 model...")
    model = PPO.load(MODEL_PATH)
    print("Done.\n")

    # Load data
    trials_by_dir = load_trials(CSV_PATH)
    print()

    # Compute trajectories
    print("Computing trajectories (this takes ~2 minutes)...")
    results = compute_trajectories(model, trials_by_dir)
    print()

    # Build figure
    print("Building interactive figure...")
    fig = build_html(results)

    # Save
    # Use inline JS so the file is fully self-contained (works offline, no CDN needed)
    # File will be ~5MB but opens instantly in any browser by double-clicking.
    html_str = fig.to_html(
        include_plotlyjs='inline',
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'copilot_trajectories',
                'height': 900,
                'width': 1600,
                'scale': 2,
            },
        },
    )
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_str)
    print(f"\n✓ Saved to: {OUTPUT_PATH}")
    print("  → Double-click the file to open in your browser (works offline)")
    print("  → Use the camera icon (top-right toolbar) to export as PNG for slides")
    print()

    # Print summary
    print("=" * 60)
    print("Per-direction summary:")
    print(f"  {'Dir':<4}  {'BCI':>6}  {'Copilot':>8}  {'Delta':>7}")
    print("  " + "-" * 32)
    total_bci = total_cop = 0
    for lbl in range(8):
        res   = results[lbl]
        delta = res['cop_acc'] - res['bci_acc']
        sign  = '+' if delta >= 0 else ''
        print(f"  {LABEL_NAMES[lbl]:<4}  {res['bci_acc']*100:>5.1f}%  "
              f"{res['cop_acc']*100:>7.1f}%  {sign}{delta*100:>5.1f}pp")
        total_bci += res['bci_acc']
        total_cop += res['cop_acc']
    print("  " + "-" * 32)
    overall_delta = (total_cop - total_bci) / 8
    sign = '+' if overall_delta >= 0 else ''
    print(f"  {'Mean':<4}  {total_bci/8*100:>5.1f}%  "
          f"{total_cop/8*100:>7.1f}%  {sign}{overall_delta*100:>5.1f}pp")
    print("=" * 60)


if __name__ == '__main__':
    main()
