"""
visualize_copilot_v3.py
=======================
V3: Updated to use the supervised LSTM copilot (best_model.pt from
supervised_copilot_v3/) instead of the RL chargeTargets model (Run5).

The LSTM copilot uses:
  - Input per tick: (cursor_x, cursor_y, vx_unit, vy_unit, vel_mag_scaled)
  - Hidden: 64 units, 2 layers
  - Confidence-weighted Stage 2:
      correction = LABEL_TO_DIR[pred] * COPILOT_VEL * confidence

Overall result: BCI 46.70% → Copilot 47.77% (+1.07pp, all 7 subjects)

Run from arm-bci-copilot/:
    python visualize_copilot_v3.py

Output: copilot_visualization.html  (~8MB, self-contained, works offline)
"""

import numpy as np
import pandas as pd
import torch
import random
import json
import torch.nn as nn
import plotly.graph_objects as go

# ── paths ──────────────────────────────────────────────────────────────────────
CSV_PATH    = "data/online_arm_trajectories.csv"
MODEL_PATH  = "supervised_copilot/best_model.pt"
OUTPUT_PATH = "copilot_visualization.html"

# ── constants ──────────────────────────────────────────────────────────────────
RADIUS_PX    = 432.0
COPILOT_VEL  = 0.02
INPUT_SIZE   = 5
HIDDEN_SIZE  = 64
N_LAYERS     = 2
VEL_MAG_MEAN = 0.0424
VEL_MAG_STD  = 0.0262
N_SAMPLE     = 20
RANDOM_SEED  = 42

LABEL_TO_DIR = np.array([
    [-0.707,  0.707], [ 0.000,  1.000], [ 0.707,  0.707],
    [ 1.000,  0.000], [ 0.707, -0.707], [ 0.000, -1.000],
    [-0.707, -0.707], [-1.000,  0.000],
], dtype=np.float32)
LABEL_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

# Keyboard clusters for each direction (from the lab typing paradigm)
KEYBOARD = {
    0: 'W/Q · E · R',    # NW
    1: 'T · Y · U',       # N
    2: 'I · O · P',       # NE
    3: 'F · G · H/J',     # E
    4: 'M · K · L',       # SE
    5: 'B · N',           # S
    6: 'Z/X · C · V',     # SW
    7: 'A · S · D',       # W
}

# Compass grid positions (row, col) in a 3×3 grid with center empty
GRID_POS = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,2), 4:(2,2), 5:(2,1), 6:(2,0), 7:(1,0)}


# ── simulation helpers ─────────────────────────────────────────────────────────

def normalize_vel(vx, vy):
    """Returns (vx_unit, vy_unit, vel_mag_scaled) — matches training."""
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)
    if mag > 1e-6:
        return vx/mag, vy/mag, mag_scaled
    return 0.0, 0.0, (0.0 - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)

def angle_pred(cursor):
    n = np.linalg.norm(cursor)
    if n < 1e-6: return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor/n)))


class LSTMCopilot(nn.Module):
    """Two-layer LSTM target classifier — matches train_supervised_copilot_v3.py."""
    def __init__(self):
        super().__init__()
        self.lstm       = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS,
                                  batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(HIDDEN_SIZE, 8)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out)


# ── data loading ───────────────────────────────────────────────────────────────

def load_trials(csv_path):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    gcols = ['subject_id','session_number','run_number','trial_number','inner_trial_number']
    by_dir  = {i:[] for i in range(8)}
    by_subj = {}
    for key, grp in df.groupby(gcols):
        grp = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl = int(grp['target_label'].iloc[0])
        if lbl not in range(8): continue
        subj = grp['subject_id'].iloc[0]
        cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx = np.diff(cx); vy = np.diff(cy)
        if len(vx) == 0: continue
        t = {
            'subject_id': subj, 'target_label': lbl,
            'vel_seq':    list(zip(vx.tolist(), vy.tolist())),
            'cursor_seq': list(zip(cx.tolist(), cy.tolist())),
        }
        by_dir[lbl].append(t)
        by_subj.setdefault(subj, []).append(t)
    for lbl in range(8):
        print(f"  {LABEL_NAMES[lbl]}: {len(by_dir[lbl])} trials")
    return by_dir, by_subj


# ── trajectory computation ─────────────────────────────────────────────────────

def sim_bci(trial):
    """BCI-only trajectory starting from (0,0). Returns (T+1,2) normalized."""
    cursor = np.zeros(2, dtype=np.float32)
    path   = [cursor.copy()]
    for vx, vy in trial['vel_seq']:
        cursor = np.clip(cursor + np.array([vx, vy], dtype=np.float32), -1.5, 1.5)
        path.append(cursor.copy())
    return np.array(path)

@torch.no_grad()
def sim_copilot(model, trial, hq_unused, prev_unused):
    """
    BCI+Copilot trajectory using the supervised LSTM copilot.
    hq_unused and prev_unused kept for API compatibility with compute_all().
    Returns (T+1,2) normalized.
    """
    model.eval()
    cursor = np.zeros(2, dtype=np.float32)
    path   = [cursor.copy()]
    h, c   = None, None

    for vx, vy in trial['vel_seq']:
        nvx, nvy, vmag = normalize_vel(vx, vy)
        x_t = torch.tensor(
            np.array([[[cursor[0], cursor[1], nvx, nvy, vmag]]]),
            dtype=torch.float32
        )
        if h is None:
            lstm_out, (h, c) = model.lstm(x_t)
        else:
            lstm_out, (h, c) = model.lstm(x_t, (h, c))

        logits_t   = model.classifier(lstm_out[:, -1, :])
        conf       = float(torch.softmax(logits_t, dim=-1).max().item())
        pred_t     = int(logits_t.argmax().item())
        correction = LABEL_TO_DIR[pred_t] * COPILOT_VEL * conf
        bci_vel    = np.array([vx, vy], dtype=np.float32)
        cursor     = np.clip(cursor + bci_vel + correction, -1.5, 1.5)
        path.append(cursor.copy())

    return np.array(path)

def compute_all(model, by_dir, by_subj):
    """
    Compute mean BCI and Copilot trajectories per direction.
    Copilot evaluation matches evaluate.py: cross-trial history per subject.
    Returns dict with trajectories and trial-level outcome data.
    """
    hq = None  # unused by LSTM copilot (kept for API compatibility)

    # Run copilot on all trials in subject order (matches evaluate_supervised_copilot.py)
    cop_outcomes = {}  # trial_id -> (bci_correct, cop_correct, bci_path, cop_path, lbl)
    print("Simulating copilot (subject order, matching evaluate.py)...")
    for subj in sorted(by_subj.keys()):
        prev = np.zeros(2, dtype=np.float32)
        for trial in by_subj[subj]:
            lbl  = trial['target_label']
            bci_path = sim_bci(trial)
            cop_path = sim_copilot(model, trial, hq, prev)
            prev = cop_path[-1].copy()
            bci_ok = angle_pred(bci_path[-1]) == lbl
            cop_ok = angle_pred(cop_path[-1]) == lbl
            tid = id(trial)
            cop_outcomes[tid] = (bci_ok, cop_ok, bci_path, cop_path, lbl)
        print(f"  {subj} done")

    # Aggregate per direction
    rng = random.Random(RANDOM_SEED)
    results = {}
    for lbl in range(8):
        trials = by_dir[lbl]
        # BCI paths (independent, no history needed)
        bci_paths = [sim_bci(t) for t in trials]
        cop_paths = [cop_outcomes[id(t)][3] for t in trials]
        bci_oks   = [cop_outcomes[id(t)][0] for t in trials]
        cop_oks   = [cop_outcomes[id(t)][1] for t in trials]

        # Pad to same length
        max_len = max(len(p) for p in bci_paths)
        def pad(paths):
            out = []
            for p in paths:
                if len(p) < max_len:
                    p = np.concatenate([p, np.repeat(p[-1:], max_len-len(p), axis=0)])
                out.append(p)
            return np.stack(out)

        bci_arr = pad(bci_paths); cop_arr = pad(cop_paths)
        mean_bci = bci_arr.mean(0); mean_cop = cop_arr.mean(0)

        # Sample individual BCI traces
        idx = rng.sample(range(len(trials)), min(N_SAMPLE, len(trials)))
        samples = [{'path': bci_paths[i],
                    'correct': bci_oks[i],
                    'pred': LABEL_NAMES[angle_pred(bci_paths[i][-1])]} for i in idx]

        # Find best correction example: BCI wrong → Copilot right
        corrections = [(i, bci_paths[i], cop_paths[i])
                       for i in range(len(trials)) if not bci_oks[i] and cop_oks[i]]
        # Find best failure example: BCI right → Copilot wrong
        failures    = [(i, bci_paths[i], cop_paths[i])
                       for i in range(len(trials)) if bci_oks[i] and not cop_oks[i]]

        # Pick a correction/failure example near the median trajectory length
        example = None
        if corrections:
            chosen = rng.choice(corrections[:20])
            example = {'type': 'correction', 'bci': chosen[1], 'cop': chosen[2]}
        elif failures:
            chosen = rng.choice(failures[:20])
            example = {'type': 'failure', 'bci': chosen[1], 'cop': chosen[2]}

        bci_acc = sum(bci_oks)/len(bci_oks)
        cop_acc = sum(cop_oks)/len(cop_oks)
        print(f"  {LABEL_NAMES[lbl]}: BCI={bci_acc*100:.1f}% → Copilot={cop_acc*100:.1f}%  "
              f"({len(corrections)} corrections, {len(failures)} failures)")

        results[lbl] = {
            'mean_bci': mean_bci, 'mean_cop': mean_cop,
            'samples': samples, 'example': example,
            'bci_acc': bci_acc, 'cop_acc': cop_acc,
        }
    return results


# ── wedge geometry ─────────────────────────────────────────────────────────────

def wedge_xy(center_deg, half_deg=22.5, r=RADIUS_PX, n=40):
    angs = np.linspace(np.radians(center_deg-half_deg),
                       np.radians(center_deg+half_deg), n)
    xs = np.concatenate([[0], r*np.cos(angs), [0]])
    ys = np.concatenate([[0], r*np.sin(angs), [0]])
    return xs*1.0, ys*1.0

def dir_deg(v): return float(np.degrees(np.arctan2(v[1], v[0])))


# ── build figure ───────────────────────────────────────────────────────────────

def build_figure(results):
    """
    Build a single go.Figure with 8 subplots in compass arrangement
    using manual axis domains.
    """
    # Panel domains: 3×3 compass grid, center empty
    PW = 0.285; PH = 0.265; GX = 0.03; GY = 0.07
    TITLE_H = 0.07  # space reserved for title at top

    def domain(lbl):
        row, col = GRID_POS[lbl]
        x0 = col*(PW+GX)
        x1 = x0 + PW
        y1 = 1.0 - TITLE_H - row*(PH+GY)
        y0 = y1 - PH
        # Clamp to [0,1] — floating point can push bottom row slightly negative
        x0 = max(0.0, min(1.0, x0)); x1 = max(0.0, min(1.0, x1))
        y0 = max(0.0, min(1.0, y0)); y1 = max(0.0, min(1.0, y1))
        return {'x':[x0,x1], 'y':[y0,y1]}

    domains = {lbl: domain(lbl) for lbl in range(8)}

    # Colors
    C_SAMPLE = 'rgba(148,163,184,0.3)'
    C_BCI    = '#2563EB'
    C_COP    = '#EA580C'
    C_WEDGE  = 'rgba(254,252,232,0.6)'
    C_WEDGE_L= 'rgba(202,198,150,0.7)'
    C_TARGET = 'rgba(100,100,100,0.5)'
    C_CORR   = '#16a34a'   # green for corrections
    C_FAIL   = '#dc2626'   # red for failures

    fig = go.Figure()
    shown = set()
    annotations = []
    axis_idx = {}  # lbl -> axis number

    AXIS_RANGE = [-RADIUS_PX*0.92, RADIUS_PX*0.92]

    for panel, lbl in enumerate(range(8)):
        ax_n  = panel + 1
        ax_s  = str(ax_n) if ax_n > 1 else ''
        axis_idx[lbl] = ax_s
        dom   = domains[lbl]
        xref  = f'x{ax_s}'; yref = f'y{ax_s}'
        dir_v = LABEL_TO_DIR[lbl]
        res   = results[lbl]

        # ── Wedge ──────────────────────────────────────────────────────────
        wx, wy = wedge_xy(dir_deg(dir_v))
        fig.add_trace(go.Scatter(
            x=wx, y=wy, fill='toself', fillcolor=C_WEDGE,
            line=dict(color=C_WEDGE_L, width=1),
            mode='lines', hoverinfo='skip', showlegend=False,
            xaxis=xref, yaxis=yref,
        ))

        # ── All 8 target markers + keyboard labels ──────────────────────
        for t_lbl in range(8):
            tx = LABEL_TO_DIR[t_lbl][0] * RADIUS_PX * 0.82
            ty = LABEL_TO_DIR[t_lbl][1] * RADIUS_PX * 0.82
            is_target = (t_lbl == lbl)
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='markers+text',
                marker=dict(
                    size=10 if is_target else 6,
                    color=C_COP if is_target else C_TARGET,
                    symbol='diamond' if is_target else 'circle',
                    line=dict(color='white', width=1),
                ),
                text=[f'<b>{LABEL_NAMES[t_lbl]}</b>' if is_target else LABEL_NAMES[t_lbl]],
                textposition='top center',
                textfont=dict(
                    size=9 if is_target else 7,
                    color=C_COP if is_target else '#9CA3AF',
                ),
                showlegend=False, hoverinfo='skip',
                xaxis=xref, yaxis=yref,
            ))

        # ── Keyboard label annotation for target direction ──────────────
        kx = dir_v[0] * RADIUS_PX * 0.60
        ky = dir_v[1] * RADIUS_PX * 0.60
        annotations.append(dict(
            x=kx, y=ky, xref=xref, yref=yref,
            text=f'<i>{KEYBOARD[lbl]}</i>',
            showarrow=False,
            font=dict(size=7, color='#6B7280'),
            bgcolor='rgba(255,255,255,0.7)',
            borderpad=1,
        ))

        # ── BCI sample traces ──────────────────────────────────────────
        for s in res['samples']:
            path = s['path'] * RADIUS_PX
            show_l = 'BCI samples' not in shown
            if show_l: shown.add('BCI samples')
            fig.add_trace(go.Scatter(
                x=path[:,0], y=path[:,1],
                mode='lines', line=dict(color=C_SAMPLE, width=1),
                showlegend=show_l, legendgroup='samples',
                name='BCI samples',
                hovertemplate=(
                    f"<b>BCI sample</b> — {LABEL_NAMES[lbl]} trial<br>"
                    f"Predicted: {s['pred']}<br>"
                    f"{'✓ Correct' if s['correct'] else '✗ Incorrect'}"
                    "<extra></extra>"
                ),
                xaxis=xref, yaxis=yref,
            ))

        # ── Mean BCI ────────────────────────────────────────────────────
        bci = res['mean_bci'] * RADIUS_PX
        show_l = 'Mean BCI' not in shown
        if show_l: shown.add('Mean BCI')
        fig.add_trace(go.Scatter(
            x=bci[:,0], y=bci[:,1], mode='lines',
            line=dict(color=C_BCI, width=2.5),
            showlegend=show_l, legendgroup='mean_bci', name='Mean BCI',
            hovertemplate="<b>Mean BCI</b><br>x=%{x:.0f} px<br>y=%{y:.0f} px<extra></extra>",
            xaxis=xref, yaxis=yref,
        ))
        fig.add_trace(go.Scatter(
            x=[bci[-1,0]], y=[bci[-1,1]], mode='markers',
            marker=dict(size=11, color=C_BCI, symbol='circle',
                        line=dict(color='white', width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>BCI mean endpoint</b><br>"
                f"x=%{{x:.0f}} px, y=%{{y:.0f}} px<br>"
                f"Accuracy: {res['bci_acc']*100:.1f}%"
                "<extra></extra>"
            ),
            xaxis=xref, yaxis=yref,
        ))

        # ── Mean BCI+Copilot ────────────────────────────────────────────
        cop = res['mean_cop'] * RADIUS_PX
        show_l = 'Mean BCI+Copilot' not in shown
        if show_l: shown.add('Mean BCI+Copilot')
        fig.add_trace(go.Scatter(
            x=cop[:,0], y=cop[:,1], mode='lines',
            line=dict(color=C_COP, width=2.5),
            showlegend=show_l, legendgroup='mean_cop', name='Mean BCI+Copilot',
            hovertemplate="<b>Mean BCI+Copilot</b><br>x=%{x:.0f} px<br>y=%{y:.0f} px<extra></extra>",
            xaxis=xref, yaxis=yref,
        ))
        fig.add_trace(go.Scatter(
            x=[cop[-1,0]], y=[cop[-1,1]], mode='markers',
            marker=dict(size=11, color=C_COP, symbol='diamond',
                        line=dict(color='white', width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>Copilot mean endpoint</b><br>"
                f"x=%{{x:.0f}} px, y=%{{y:.0f}} px<br>"
                f"Accuracy: {res['cop_acc']*100:.1f}%"
                "<extra></extra>"
            ),
            xaxis=xref, yaxis=yref,
        ))

        # ── Origin ──────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=7, color='#6B7280', symbol='cross'),
            showlegend=False, hoverinfo='skip',
            xaxis=xref, yaxis=yref,
        ))

        # ── Accuracy annotation (single, no duplicate) ──────────────────
        delta = res['cop_acc'] - res['bci_acc']
        sign  = '+' if delta >= 0 else ''
        col   = C_CORR if delta >= 0 else C_FAIL
        # Place at top of panel in paper coords
        cx_paper = (dom['x'][0] + dom['x'][1]) / 2
        cy_paper = dom['y'][1] + 0.005
        annotations.append(dict(
            xref='paper', yref='paper',
            x=cx_paper, y=cy_paper,
            text=(f"<b>{LABEL_NAMES[lbl]}</b>  "
                  f"BCI {res['bci_acc']*100:.1f}% → "
                  f"Copilot {res['cop_acc']*100:.1f}%  "
                  f"<span style='color:{col}'><b>{sign}{delta*100:.1f}pp</b></span>"),
            showarrow=False,
            font=dict(size=9, family='monospace'),
            align='center',
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='rgba(180,180,180,0.5)',
            borderwidth=1, borderpad=3,
        ))

        # ── Panel title (direction name) ─────────────────────────────
        annotations.append(dict(
            xref='paper', yref='paper',
            x=cx_paper, y=dom['y'][1] + 0.028,
            text=f'<b>{LABEL_NAMES[lbl]}</b>',
            showarrow=False,
            font=dict(size=12, family='Arial'),
        ))

    # ── Axis layout ────────────────────────────────────────────────────────
    axis_updates = {}
    for lbl in range(8):
        ax_s = axis_idx[lbl]
        dom  = domains[lbl]
        base_cfg = dict(
            range=AXIS_RANGE,
            showgrid=True, gridcolor='rgba(200,200,200,0.4)',
            zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1,
            showticklabels=False,
            constrain='domain',
            title='',
        )
        x_key = f'xaxis{axis_idx[lbl]}'
        y_key = f'yaxis{axis_idx[lbl]}'
        axis_updates[x_key] = dict(**base_cfg, domain=domains[lbl]['x'])
        axis_updates[y_key] = dict(**base_cfg, domain=domains[lbl]['y'],
                                   scaleanchor=f'x{axis_idx[lbl]}', scaleratio=1)

    # ── Global title ───────────────────────────────────────────────────────
    annotations.append(dict(
        xref='paper', yref='paper', x=0.5, y=1.0,
        text='<b>BCI Copilot — Cursor Trajectory Visualization</b>',
        showarrow=False, font=dict(size=17, family='Arial'),
        align='center',
    ))
    annotations.append(dict(
        xref='paper', yref='paper', x=0.5, y=0.974,
        text=(f'Supervised LSTM Copilot V3 · '
              f'Data: online_arm_trajectories.csv · '
              f'Overall: BCI 46.7% → Copilot 47.8% (+1.07pp) · '
              f'1,470 trials per direction'),
        showarrow=False, font=dict(size=10, color='#6B7280'),
        align='center',
    ))

    fig.update_layout(
        **axis_updates,
        annotations=annotations,
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.04,
            xanchor='center', x=0.5,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(200,200,200,0.5)', borderwidth=1,
        ),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
        height=900, width=1300,
        margin=dict(t=60, b=60, l=20, r=20),
        hovermode='closest',
    )

    return fig


# ── correction animation HTML ──────────────────────────────────────────────────

def build_correction_section(results):
    """
    Build a separate section with 8 animated panels (one per direction)
    showing a real correction example: BCI wrong → Copilot right.
    Returns an HTML string to embed after the main figure.
    """
    sections = []
    for lbl in range(8):
        res = results[lbl]
        ex  = res.get('example')
        name = LABEL_NAMES[lbl]
        keys = KEYBOARD[lbl]

        if ex is None:
            sections.append(f'<div class="panel"><h3>{name}</h3><p>No example found.</p></div>')
            continue

        bci_path = ex['bci'] * RADIUS_PX   # (T, 2)
        cop_path = ex['cop'] * RADIUS_PX
        ex_type  = ex['type']

        bci_final_pred = LABEL_NAMES[angle_pred(ex['bci'][-1])]
        cop_final_pred = LABEL_NAMES[angle_pred(ex['cop'][-1])]
        bci_ok = angle_pred(ex['bci'][-1]) == lbl
        cop_ok = angle_pred(ex['cop'][-1]) == lbl

        if ex_type == 'correction':
            headline = f'✓ Copilot corrects a wrong trial'
            subline  = f'BCI predicted {bci_final_pred} (✗) → Copilot predicted {cop_final_pred} (✓)'
            h_color  = '#16a34a'
        else:
            headline = f'✗ Copilot fails a correct trial'
            subline  = f'BCI predicted {bci_final_pred} (✓) → Copilot predicted {cop_final_pred} (✗)'
            h_color  = '#dc2626'

        T = len(bci_path)
        wx, wy = wedge_xy(dir_deg(LABEL_TO_DIR[lbl]))

        # Build Plotly animation figure for this direction
        fig_anim = go.Figure()

        # Static: wedge
        fig_anim.add_trace(go.Scatter(
            x=wx.tolist(), y=wy.tolist(), fill='toself',
            fillcolor='rgba(254,252,232,0.6)',
            line=dict(color='rgba(202,198,150,0.7)', width=1),
            mode='lines', hoverinfo='skip', showlegend=False,
        ))

        # Static: target markers for all 8 directions
        for t_lbl in range(8):
            tx = LABEL_TO_DIR[t_lbl][0] * RADIUS_PX * 0.82
            ty = LABEL_TO_DIR[t_lbl][1] * RADIUS_PX * 0.82
            is_tgt = t_lbl == lbl
            fig_anim.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='markers+text',
                marker=dict(size=9 if is_tgt else 5,
                            color='#EA580C' if is_tgt else 'rgba(100,100,100,0.4)',
                            symbol='diamond' if is_tgt else 'circle'),
                text=[f'<b>{LABEL_NAMES[t_lbl]}</b>' if is_tgt else LABEL_NAMES[t_lbl]],
                textposition='top center',
                textfont=dict(size=8 if is_tgt else 6,
                              color='#EA580C' if is_tgt else '#9CA3AF'),
                showlegend=False, hoverinfo='skip',
            ))

        # Origin
        fig_anim.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=6, color='#6B7280', symbol='cross'),
            showlegend=False, hoverinfo='skip',
        ))

        N_STATIC = len(fig_anim.data)

        # Initial animated traces (tick 0)
        fig_anim.add_trace(go.Scatter(
            x=[0], y=[0], mode='lines+markers',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(size=8, color='#2563EB'),
            name='BCI decoder', showlegend=True,
        ))
        fig_anim.add_trace(go.Scatter(
            x=[0], y=[0], mode='lines+markers',
            line=dict(color='#EA580C', width=2.5, dash='dot'),
            marker=dict(size=8, color='#EA580C', symbol='diamond'),
            name='BCI+Copilot', showlegend=True,
        ))

        # Animation frames (one per tick)
        frames = []
        for t in range(1, T+1):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=bci_path[:t,0].tolist(), y=bci_path[:t,1].tolist(),
                               mode='lines+markers',
                               marker=dict(size=[4]*(t-1)+[10], color='#2563EB'),
                               line=dict(color='#2563EB', width=2.5)),
                    go.Scatter(x=cop_path[:t,0].tolist(), y=cop_path[:t,1].tolist(),
                               mode='lines+markers',
                               marker=dict(size=[4]*(t-1)+[10], color='#EA580C',
                                           symbol='diamond'),
                               line=dict(color='#EA580C', width=2.5, dash='dot')),
                ],
                traces=[N_STATIC, N_STATIC+1],
                name=str(t),
            ))
        fig_anim.frames = frames

        AXIS_R = [-RADIUS_PX*0.92, RADIUS_PX*0.92]
        fig_anim.update_layout(
            xaxis=dict(range=AXIS_R, showticklabels=False, showgrid=True,
                       gridcolor='rgba(200,200,200,0.4)', zeroline=True,
                       zerolinecolor='rgba(150,150,150,0.5)',
                       constrain='domain'),
            yaxis=dict(range=AXIS_R, showticklabels=False, showgrid=True,
                       gridcolor='rgba(200,200,200,0.4)', zeroline=True,
                       zerolinecolor='rgba(150,150,150,0.5)',
                       scaleanchor='x', scaleratio=1),
            plot_bgcolor='#F8FAFC', paper_bgcolor='white',
            height=380, width=380,
            margin=dict(t=10, b=50, l=10, r=10),
            legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center',
                        font=dict(size=10)),
            updatemenus=[dict(
                type='buttons', showactive=False,
                y=-0.18, x=0.5, xanchor='center',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate')]),
                    dict(label='↺ Reset',
                         method='animate',
                         args=[['1'], dict(frame=dict(duration=0, redraw=True),
                                           mode='immediate')]),
                ]
            )],
            sliders=[dict(
                steps=[dict(args=[[f.name],
                                  dict(frame=dict(duration=0, redraw=True),
                                       mode='immediate')],
                            label=f.name, method='animate')
                       for f in fig_anim.frames],
                x=0.05, len=0.9, y=-0.05,
                currentvalue=dict(prefix='Tick: ', font=dict(size=10)),
                pad=dict(t=10),
            )],
        )

        anim_html = fig_anim.to_html(
            include_plotlyjs=False, full_html=False,
            config={'displayModeBar': False},
        )

        sections.append(f"""
        <div class="ex-panel">
            <div class="ex-header">
                <span class="dir-badge">{name}</span>
                <span class="keys">{keys}</span>
                <span class="headline" style="color:{h_color}">{headline}</span>
            </div>
            <div class="subline">{subline}</div>
            {anim_html}
        </div>
        """)

    return '\n'.join(sections)


# ── assemble final HTML ────────────────────────────────────────────────────────

def save_html(main_fig, results):
    main_html = main_fig.to_html(
        include_plotlyjs='inline', full_html=False,
        config={'displayModeBar': True, 'scrollZoom': True,
                'toImageButtonOptions': {'format':'png','filename':'bci_copilot_overview',
                                         'height':900,'width':1300,'scale':2}},
    )
    correction_html = build_correction_section(results)

    full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BCI Copilot Visualization V2</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F8FAFC;
          margin: 0; padding: 20px; color: #1e293b; }}
  h1   {{ text-align: center; font-size: 22px; margin: 20px 0 4px; color: #0f172a; }}
  h2   {{ text-align: center; font-size: 16px; color: #475569; margin: 0 0 16px;
          font-weight: normal; }}
  .section-title {{ font-size: 18px; font-weight: bold; margin: 32px 0 12px;
                    padding-left: 8px; border-left: 4px solid #2563EB; }}
  .section-sub   {{ font-size: 12px; color: #64748b; margin: -8px 0 16px 12px; }}
  .ex-grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
  .ex-panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px;
               padding: 12px; width: 400px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  .ex-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
  .dir-badge {{ background: #1e40af; color: white; font-weight: bold; font-size: 13px;
                padding: 2px 8px; border-radius: 4px; }}
  .keys      {{ font-size: 11px; color: #64748b; font-family: monospace; }}
  .headline  {{ font-size: 12px; font-weight: 600; }}
  .subline   {{ font-size: 11px; color: #64748b; margin-bottom: 8px; }}
  .divider   {{ border: none; border-top: 1px solid #e2e8f0; margin: 28px 0; }}
</style>
</head>
<body>
<h1>BCI Copilot — Cursor Trajectory Visualization</h1>
<h2>Supervised LSTM Copilot V3 &nbsp;·&nbsp;
    Data: online_arm_trajectories.csv &nbsp;·&nbsp;
    Overall: BCI 46.7% → Copilot 47.8% (+1.07pp)</h2>

<div class="section-title">Overview: all 8 directions</div>
<div class="section-sub">
  Compass layout · 20 individual BCI samples (grey) · Mean BCI (blue) · Mean BCI+Copilot (orange)
  · Coordinates in pixels (radius = 432)
</div>
{main_html}

<hr class="divider">

<div class="section-title">Individual trajectory replay</div>
<div class="section-sub">
  Each panel shows a real trial where the BCI decoder alone was wrong (or right)
  and the copilot changed the outcome. Press ▶ Play to animate tick-by-tick.
  Use the slider to step through individual ticks.
</div>
<div class="ex-grid">
{correction_html}
</div>

</body>
</html>"""

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    print("  Double-click to open in browser (self-contained, works offline)")
    print("  Use the camera icon in the overview toolbar to export PNG for slides")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BCI Copilot Visualization V3 — Supervised LSTM")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {CSV_PATH}")
    print()

    print("Loading supervised LSTM copilot (V3)...")
    model = LSTMCopilot()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("Done.\n")

    by_dir, by_subj = load_trials(CSV_PATH)
    print()

    results = compute_all(model, by_dir, by_subj)
    print()

    print("Building overview figure...")
    main_fig = build_figure(results)

    print("Building correction animations and assembling HTML...")
    save_html(main_fig, results)

    print()
    print("=" * 60)
    print("Per-direction summary (matches evaluate.py cross-trial history):")
    print(f"  {'Dir':<4}  {'BCI':>6}  {'Copilot':>8}  {'Delta':>8}")
    print("  " + "-" * 34)
    for lbl in range(8):
        r = results[lbl]
        d = r['cop_acc'] - r['bci_acc']
        s = '+' if d >= 0 else ''
        print(f"  {LABEL_NAMES[lbl]:<4}  {r['bci_acc']*100:>5.1f}%"
              f"  {r['cop_acc']*100:>7.1f}%  {s}{d*100:>5.1f}pp")
    print("=" * 60)


if __name__ == '__main__':
    main()
