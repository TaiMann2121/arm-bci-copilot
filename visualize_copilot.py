"""
visualize_copilot.py
=======================
V4: Redesigned visualization — removes mean trajectory overview,
replaces with compact accuracy summary table and a richer trajectory
explorer with 4 outcome categories per direction.

The LSTM copilot uses:
  - Input per tick: (cursor_x, cursor_y, vx_unit, vy_unit, vel_mag_scaled)
  - Hidden: 64 units, 2 layers
  - Confidence-weighted Stage 2:
      correction = LABEL_TO_DIR[pred] * COPILOT_VEL * confidence

Overall result: BCI 46.70% → Copilot 47.77% (+1.07pp, all 7 subjects)

Run from arm-bci-copilot/:
    python visualize_copilot.py

Output: copilot_visualization.html  (self-contained, works offline)
"""

import numpy as np
import pandas as pd
import torch
import random
import json
import torch.nn as nn

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
N_SAMPLE     = 20   # trials per category per direction embedded in HTML
RANDOM_SEED  = 42

LABEL_TO_DIR = np.array([
    [-0.707,  0.707], [ 0.000,  1.000], [ 0.707,  0.707],
    [ 1.000,  0.000], [ 0.707, -0.707], [ 0.000, -1.000],
    [-0.707, -0.707], [-1.000,  0.000],
], dtype=np.float32)
LABEL_NAMES = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']

# Primary keyboard cluster for each direction
KEYBOARD = {
    0: 'W/Q · E · R',
    1: 'T · Y · U',
    2: 'I · O · P',
    3: 'F · G · H/J',
    4: 'M · K · L',
    5: 'B · N',
    6: 'Z/X · C · V',
    7: 'A · S · D',
}

# 4 outcome categories
CATEGORIES = {
    'correction': {'label': '✓ Copilot corrects',  'color': '#16a34a',
                   'desc': 'BCI wrong → Copilot correct'},
    'failure':    {'label': '✗ Copilot fails',      'color': '#dc2626',
                   'desc': 'BCI correct → Copilot wrong'},
    'both_fail':  {'label': '✗ Both wrong',          'color': '#9333ea',
                   'desc': 'BCI wrong → Copilot wrong'},
    'both_ok':    {'label': '✓ Both correct',        'color': '#2563EB',
                   'desc': 'BCI correct → Copilot correct'},
}
CAT_ORDER = ['correction', 'failure', 'both_fail', 'both_ok']


# ── model ──────────────────────────────────────────────────────────────────────

class LSTMCopilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm       = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS,
                                  batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(HIDDEN_SIZE, 8)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out)


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_vel(vx, vy):
    mag = np.sqrt(vx**2 + vy**2)
    mag_scaled = (mag - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)
    if mag > 1e-6:
        return vx/mag, vy/mag, mag_scaled
    return 0.0, 0.0, (0.0 - VEL_MAG_MEAN) / (VEL_MAG_STD + 1e-8)

def angle_pred(cursor):
    n = np.linalg.norm(cursor)
    if n < 1e-6: return -1
    return int(np.argmax(LABEL_TO_DIR @ (cursor/n)))

def dir_deg(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))

def wedge_points(center_deg, half_deg=22.5, r=RADIUS_PX, n=40):
    angs = np.linspace(np.radians(center_deg-half_deg),
                       np.radians(center_deg+half_deg), n)
    xs = np.concatenate([[0], r*np.cos(angs), [0]])
    ys = np.concatenate([[0], r*np.sin(angs), [0]])
    return xs.tolist(), ys.tolist()


# ── data loading ───────────────────────────────────────────────────────────────

def load_trials(csv_path):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    gcols = ['subject_id','session_number','run_number',
             'trial_number','inner_trial_number']
    by_dir  = {i:[] for i in range(8)}
    by_subj = {}
    for _, grp in df.groupby(gcols):
        grp  = grp.sort_values('timestamp_seconds').reset_index(drop=True)
        lbl  = int(grp['target_label'].iloc[0])
        if lbl not in range(8): continue
        subj = grp['subject_id'].iloc[0]
        cx = grp['cursor_pos_x'].values.astype(np.float32) / RADIUS_PX
        cy = grp['cursor_pos_y'].values.astype(np.float32) / RADIUS_PX
        vx = np.diff(cx); vy = np.diff(cy)
        if len(vx) == 0: continue
        t = {'subject_id': subj, 'target_label': lbl,
             'vel_seq': list(zip(vx.tolist(), vy.tolist())),
             'cursor_seq': list(zip(cx.tolist(), cy.tolist()))}
        by_dir[lbl].append(t)
        by_subj.setdefault(subj, []).append(t)
    for lbl in range(8):
        print(f"  {LABEL_NAMES[lbl]}: {len(by_dir[lbl])} trials")
    return by_dir, by_subj


# ── trajectory simulation ──────────────────────────────────────────────────────

def sim_bci(trial):
    cursor = np.zeros(2, dtype=np.float32)
    path   = [cursor.copy()]
    for vx, vy in trial['vel_seq']:
        cursor = np.clip(cursor + np.array([vx,vy], dtype=np.float32), -1.5, 1.5)
        path.append(cursor.copy())
    return np.array(path)

@torch.no_grad()
def sim_copilot(model, trial):
    model.eval()
    cursor = np.zeros(2, dtype=np.float32)
    path   = [cursor.copy()]
    h, c   = None, None
    for vx, vy in trial['vel_seq']:
        nvx, nvy, vmag = normalize_vel(vx, vy)
        x_t = torch.tensor(
            np.array([[[cursor[0], cursor[1], nvx, nvy, vmag]]]),
            dtype=torch.float32)
        if h is None:
            lstm_out, (h, c) = model.lstm(x_t)
        else:
            lstm_out, (h, c) = model.lstm(x_t, (h, c))
        logits_t   = model.classifier(lstm_out[:, -1, :])
        conf       = float(torch.softmax(logits_t, dim=-1).max().item())
        pred_t     = int(logits_t.argmax().item())
        correction = LABEL_TO_DIR[pred_t] * COPILOT_VEL * conf
        cursor     = np.clip(cursor + np.array([vx,vy],dtype=np.float32)
                             + correction, -1.5, 1.5)
        path.append(cursor.copy())
    return np.array(path)


# ── main computation ───────────────────────────────────────────────────────────

def compute_all(model, by_dir, by_subj):
    """
    Simulate copilot on all trials (subject order for cross-trial consistency).
    Returns per-direction accuracy stats and 4-category trial samples.
    """
    rng = random.Random(RANDOM_SEED)

    # Step 1: simulate all trials in subject order
    outcomes = {}  # id(trial) -> (bci_ok, cop_ok, bci_path, cop_path)
    print("Simulating copilot (subject order)...")
    for subj in sorted(by_subj.keys()):
        for trial in by_subj[subj]:
            bci_path = sim_bci(trial)
            cop_path = sim_copilot(model, trial)
            bci_ok   = angle_pred(bci_path[-1]) == trial['target_label']
            cop_ok   = angle_pred(cop_path[-1]) == trial['target_label']
            outcomes[id(trial)] = (bci_ok, cop_ok, bci_path, cop_path)
        print(f"  {subj} done")

    # Step 2: per-direction stats and category buckets
    results = {}
    for lbl in range(8):
        trials   = by_dir[lbl]
        bci_oks  = [outcomes[id(t)][0] for t in trials]
        cop_oks  = [outcomes[id(t)][1] for t in trials]
        bci_acc  = sum(bci_oks) / len(bci_oks)
        cop_acc  = sum(cop_oks) / len(cop_oks)

        # Bucket all trials into 4 categories
        buckets = {'correction':[], 'failure':[], 'both_fail':[], 'both_ok':[]}
        for i, t in enumerate(trials):
            bci_ok, cop_ok, bci_path, cop_path = outcomes[id(t)]
            entry = {
                'bci': (bci_path * RADIUS_PX).tolist(),
                'cop': (cop_path * RADIUS_PX).tolist(),
                'bci_pred': LABEL_NAMES[angle_pred(bci_path[-1])],
                'cop_pred': LABEL_NAMES[angle_pred(cop_path[-1])],
            }
            if not bci_ok and cop_ok:
                buckets['correction'].append(entry)
            elif bci_ok and not cop_ok:
                buckets['failure'].append(entry)
            elif not bci_ok and not cop_ok:
                buckets['both_fail'].append(entry)
            else:
                buckets['both_ok'].append(entry)

        # Sample N_SAMPLE from each bucket (shuffle for variety)
        sampled = {}
        for cat, entries in buckets.items():
            rng.shuffle(entries)
            sampled[cat] = entries[:N_SAMPLE]

        n_corr = len(buckets['correction'])
        n_fail = len(buckets['failure'])
        print(f"  {LABEL_NAMES[lbl]}: BCI={bci_acc*100:.1f}% → "
              f"Copilot={cop_acc*100:.1f}%  "
              f"({n_corr} corrections, {n_fail} failures)")

        results[lbl] = {
            'bci_acc': bci_acc, 'cop_acc': cop_acc,
            'buckets': sampled,
            'counts':  {cat: len(buckets[cat]) for cat in CAT_ORDER},
        }

    return results


# ── accuracy summary table ─────────────────────────────────────────────────────

def build_accuracy_table(results):
    """Compact HTML table showing per-direction accuracy metrics."""
    rows = []
    total_bci = sum(r['bci_acc'] for r in results.values()) / 8
    total_cop = sum(r['cop_acc'] for r in results.values()) / 8

    for lbl in range(8):
        r     = results[lbl]
        delta = r['cop_acc'] - r['bci_acc']
        sign  = '+' if delta >= 0 else ''
        dcol  = '#16a34a' if delta >= 0 else '#dc2626'
        counts = r['counts']
        rows.append(f"""
        <tr>
          <td class="dir-cell">
            <span class="dir-badge-sm">{LABEL_NAMES[lbl]}</span>
            <span class="key-sm">{KEYBOARD[lbl]}</span>
          </td>
          <td class="num">{r['bci_acc']*100:.1f}%</td>
          <td class="num">{r['cop_acc']*100:.1f}%</td>
          <td class="num delta" style="color:{dcol}">{sign}{delta*100:.1f}pp</td>
          <td class="num cat-count corr">{counts['correction']}</td>
          <td class="num cat-count fail">{counts['failure']}</td>
          <td class="num cat-count bfail">{counts['both_fail']}</td>
          <td class="num cat-count bok">{counts['both_ok']}</td>
        </tr>""")

    # Overall row
    d_all  = total_cop - total_bci
    s_all  = '+' if d_all >= 0 else ''
    dc_all = '#16a34a' if d_all >= 0 else '#dc2626'
    rows.append(f"""
        <tr class="total-row">
          <td class="dir-cell"><b>Overall</b></td>
          <td class="num"><b>{total_bci*100:.1f}%</b></td>
          <td class="num"><b>{total_cop*100:.1f}%</b></td>
          <td class="num delta" style="color:{dc_all}"><b>{s_all}{d_all*100:.2f}pp</b></td>
          <td class="num" colspan="4" style="color:#94a3b8;font-size:11px;">
            46.70% → 47.77% (+1.07pp, all 7 subjects)
          </td>
        </tr>""")

    return f"""
<table class="acc-table">
  <thead>
    <tr>
      <th>Direction</th>
      <th>BCI</th>
      <th>Copilot</th>
      <th>Delta</th>
      <th title="BCI wrong → Copilot correct" class="cat-head corr">✓ Corrected</th>
      <th title="BCI correct → Copilot wrong" class="cat-head fail">✗ Failed</th>
      <th title="BCI wrong → Copilot wrong"   class="cat-head bfail">✗ Both fail</th>
      <th title="BCI correct → Copilot correct" class="cat-head bok">✓ Both ok</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>"""


# ── trajectory explorer ────────────────────────────────────────────────────────

def build_explorer(results):
    """
    Interactive trajectory explorer: user picks direction + category,
    then navigates through up to N_SAMPLE pre-embedded trials.
    Returns HTML string with embedded JS.
    """
    # Embed all sampled trial data as JSON
    data_by_dir = {}
    for lbl in range(8):
        data_by_dir[lbl] = results[lbl]['buckets']

    # Wedge data for each direction
    wedges = {}
    for lbl in range(8):
        wx, wy = wedge_points(dir_deg(LABEL_TO_DIR[lbl]))
        wedges[lbl] = {'x': wx, 'y': wy}

    # Target marker positions
    target_markers = []
    for t in range(8):
        target_markers.append({
            'x': float(LABEL_TO_DIR[t][0] * RADIUS_PX * 0.82),
            'y': float(LABEL_TO_DIR[t][1] * RADIUS_PX * 0.82),
            'name': LABEL_NAMES[t],
        })

    # Category labels / colors for JS
    cat_meta = {k: {'label': v['label'], 'color': v['color'], 'desc': v['desc']}
                for k, v in CATEGORIES.items()}
    counts_by_dir = {lbl: results[lbl]['counts'] for lbl in range(8)}
    acc_by_dir    = {lbl: {'bci': results[lbl]['bci_acc'],
                            'cop': results[lbl]['cop_acc']}
                     for lbl in range(8)}

    data_json    = json.dumps(data_by_dir)
    wedges_json  = json.dumps(wedges)
    markers_json = json.dumps(target_markers)
    catmeta_json = json.dumps(cat_meta)
    counts_json  = json.dumps(counts_by_dir)
    acc_json     = json.dumps(acc_by_dir)
    labels_json  = json.dumps(LABEL_NAMES)
    keys_json    = json.dumps(KEYBOARD)
    catorder_json= json.dumps(CAT_ORDER)
    R            = RADIUS_PX

    return f"""
<div id="explorer">

  <!-- Controls row -->
  <div class="ctrl-row">
    <div class="ctrl-group">
      <label class="ctrl-label">Direction</label>
      <div class="btn-group" id="dir-btns">
        <!-- filled by JS -->
      </div>
    </div>
    <div class="ctrl-group">
      <label class="ctrl-label">Outcome category</label>
      <div class="btn-group" id="cat-btns">
        <!-- filled by JS -->
      </div>
    </div>
    <div class="ctrl-group nav-group">
      <label class="ctrl-label">Trial</label>
      <div class="nav-row">
        <button class="nav-btn" id="btn-prev" onclick="navTrial(-1)">&#8592;</button>
        <span id="trial-counter" class="trial-counter">1 / 20</span>
        <button class="nav-btn" id="btn-next" onclick="navTrial(+1)">&#8594;</button>
        <button class="nav-btn" id="btn-rand" onclick="randTrial()" title="Random trial">&#x21BB;</button>
      </div>
    </div>
  </div>

  <!-- Main plot + info panel -->
  <div class="explorer-body">
    <div id="traj-plot" style="width:520px;height:520px;flex-shrink:0;"></div>
    <div class="info-panel">
      <div id="cat-header" class="cat-header-box"></div>
      <div id="outcome-info" class="outcome-info"></div>
      <div class="playback-box">
        <div class="tick-row">
          <span class="tick-label">Tick:</span>
          <input type="range" id="tick-slider" min="0" max="16" value="16"
                 oninput="onSlider(this.value)" style="flex:1;">
          <span id="tick-val" class="tick-val">16</span>
        </div>
        <div class="play-row">
          <button class="play-btn" id="btn-play"  onclick="playAnim()">&#9654; Play</button>
          <button class="play-btn" id="btn-pause" onclick="pauseAnim()" disabled>&#9646;&#9646; Pause</button>
          <button class="play-btn" id="btn-reset" onclick="resetAnim()">&#8635; Reset</button>
        </div>
      </div>
      <div class="legend-box">
        <div class="leg-item"><span class="leg-line bci-line"></span>BCI decoder</div>
        <div class="leg-item"><span class="leg-line cop-line"></span>BCI + Copilot</div>
      </div>
      <div id="count-note" class="count-note"></div>
    </div>
  </div>

</div>

<script>
(function() {{
  // ── embedded data ──────────────────────────────────────────────────────────
  const DATA     = {data_json};
  const WEDGES   = {wedges_json};
  const MARKERS  = {markers_json};
  const CATMETA  = {catmeta_json};
  const COUNTS   = {counts_json};
  const ACC      = {acc_json};
  const LABELS   = {labels_json};
  const KEYS     = {keys_json};
  const CATORDER = {catorder_json};
  const R        = {R};

  // ── state ──────────────────────────────────────────────────────────────────
  let curDir  = 0;
  let curCat  = 'correction';
  let curIdx  = 0;
  let curTick = 16;
  let animTimer = null;

  // ── init Plotly ────────────────────────────────────────────────────────────
  const AXIS_R  = [-R*0.92, R*0.92];
  const axCfg   = {{
    range: AXIS_R, showticklabels: false, showgrid: true,
    gridcolor: 'rgba(200,200,200,0.4)', zeroline: true,
    zerolinecolor: 'rgba(150,150,150,0.5)', constrain: 'domain',
    fixedrange: true,
  }};
  Plotly.newPlot('traj-plot', [], {{
    xaxis: axCfg, yaxis: {{...axCfg, scaleanchor:'x', scaleratio:1}},
    plot_bgcolor:'#F8FAFC', paper_bgcolor:'white',
    margin: {{t:10,b:10,l:10,r:10}},
    showlegend: false,
    height: 520, width: 520,
  }}, {{displayModeBar:false, responsive:false}});

  // ── build control buttons ──────────────────────────────────────────────────
  function buildButtons() {{
    // Direction buttons
    const db = document.getElementById('dir-btns');
    db.innerHTML = '';
    LABELS.forEach((name, lbl) => {{
      const b = document.createElement('button');
      b.className = 'sel-btn' + (lbl === curDir ? ' active' : '');
      b.textContent = name;
      b.title = KEYS[lbl];
      b.onclick = () => {{ curDir = lbl; curIdx = 0; curTick = 16; rebuild(); }};
      db.appendChild(b);
    }});

    // Category buttons
    const cb = document.getElementById('cat-btns');
    cb.innerHTML = '';
    CATORDER.forEach(cat => {{
      const n = (DATA[curDir][cat] || []).length;
      const b = document.createElement('button');
      b.className = 'sel-btn cat-btn' + (cat === curCat ? ' active' : '');
      b.style.setProperty('--cat-color', CATMETA[cat].color);
      b.innerHTML = `${{CATMETA[cat].label}} <span class="cnt">(${{n}})</span>`;
      b.disabled  = (n === 0);
      b.onclick   = () => {{
        if (n === 0) return;
        curCat = cat; curIdx = 0; curTick = 16; rebuild();
      }};
      cb.appendChild(b);
    }});
  }}

  // ── update plot ────────────────────────────────────────────────────────────
  function updatePlot() {{
    const trials = DATA[curDir][curCat] || [];
    if (trials.length === 0) return;
    const trial  = trials[curIdx];
    const T      = trial.bci.length - 1;  // number of ticks
    const tick   = Math.min(curTick, T);
    const bci_s  = trial.bci.slice(0, tick+1);
    const cop_s  = trial.cop.slice(0, tick+1);

    const wx = WEDGES[curDir].x;
    const wy = WEDGES[curDir].y;
    const catColor = CATMETA[curCat].color;

    const traces = [
      // Wedge
      {{x: wx, y: wy, fill:'toself', fillcolor:'rgba(254,252,232,0.7)',
        line:{{color:'rgba(202,198,150,0.8)',width:1}},
        mode:'lines', hoverinfo:'skip', type:'scatter'}},
      // Target markers
      ...MARKERS.map((m,i) => ({{
        x:[m.x], y:[m.y], mode:'markers+text',
        marker:{{size: i===curDir?11:6,
                 color: i===curDir?catColor:'rgba(100,100,100,0.4)',
                 symbol: i===curDir?'diamond':'circle',
                 line:{{color:'white',width:1}}}},
        text:[`<b>${{m.name}}</b>`],
        textposition:'top center',
        textfont:{{size: i===curDir?10:7,
                   color: i===curDir?catColor:'#9CA3AF'}},
        hoverinfo:'skip', type:'scatter'
      }})),
      // Origin
      {{x:[0],y:[0],mode:'markers',
        marker:{{size:7,color:'#6B7280',symbol:'cross'}},
        hoverinfo:'skip',type:'scatter'}},
      // BCI path
      {{x: bci_s.map(p=>p[0]), y: bci_s.map(p=>p[1]),
        mode: 'lines+markers',
        line:{{color:'#2563EB',width:2.5}},
        marker:{{size: bci_s.map((_,i)=>i===bci_s.length-1?10:4),
                 color:'#2563EB'}},
        hovertemplate:'BCI<br>x: %{{x:.0f}}px<br>y: %{{y:.0f}}px<extra></extra>',
        type:'scatter'}},
      // Copilot path
      {{x: cop_s.map(p=>p[0]), y: cop_s.map(p=>p[1]),
        mode: 'lines+markers',
        line:{{color:'#EA580C',width:2.5,dash:'dot'}},
        marker:{{size: cop_s.map((_,i)=>i===cop_s.length-1?10:4),
                 color:'#EA580C', symbol:'diamond'}},
        hovertemplate:'Copilot<br>x: %{{x:.0f}}px<br>y: %{{y:.0f}}px<extra></extra>',
        type:'scatter'}},
    ];

    Plotly.react('traj-plot', traces, {{
      xaxis: axCfg, yaxis: {{...axCfg, scaleanchor:'x', scaleratio:1}},
      plot_bgcolor:'#F8FAFC', paper_bgcolor:'white',
      margin:{{t:10,b:10,l:10,r:10}}, showlegend:false,
      height:520, width:520,
    }}, {{displayModeBar:false, responsive:false}});

    // Slider
    const slider = document.getElementById('tick-slider');
    slider.max   = T;
    slider.value = tick;
    document.getElementById('tick-val').textContent = tick;

    // Outcome info
    const bciOk = (LABELS.indexOf(trial.bci_pred) === curDir);
    const copOk = (LABELS.indexOf(trial.cop_pred) === curDir);
    document.getElementById('outcome-info').innerHTML = `
      <div class="pred-row">
        <span class="pred-label">BCI predicted:</span>
        <span class="pred-val ${{bciOk?'ok':'wrong'}}">${{trial.bci_pred}}
          ${{bciOk?'✓':'✗'}}</span>
      </div>
      <div class="pred-row">
        <span class="pred-label">Copilot predicted:</span>
        <span class="pred-val ${{copOk?'ok':'wrong'}}">${{trial.cop_pred}}
          ${{copOk?'✓':'✗'}}</span>
      </div>
      <div class="pred-row">
        <span class="pred-label">Target:</span>
        <span class="pred-val target">${{LABELS[curDir]}}
          &nbsp;<i style="font-size:10px;color:#94a3b8;">${{KEYS[curDir]}}</i></span>
      </div>`;
  }}

  // ── rebuild everything ─────────────────────────────────────────────────────
  function rebuild() {{
    pauseAnim();
    buildButtons();
    updateCountNote();
    updateCatHeader();
    updateTrialCounter();
    updatePlot();
    resetSlider();
  }}

  function updateCatHeader() {{
    const meta = CATMETA[curCat];
    document.getElementById('cat-header').innerHTML =
      `<span style="color:${{meta.color}};font-weight:700;">${{meta.label}}</span>
       <span class="cat-desc">&nbsp;—&nbsp;${{meta.desc}}</span>`;
  }}

  function updateTrialCounter() {{
    const n = (DATA[curDir][curCat] || []).length;
    document.getElementById('trial-counter').textContent =
      n > 0 ? `${{curIdx+1}} / ${{n}}` : '0 / 0';
    document.getElementById('btn-prev').disabled = (n === 0 || curIdx === 0);
    document.getElementById('btn-next').disabled = (n === 0 || curIdx >= n-1);
  }}

  function updateCountNote() {{
    const c  = COUNTS[curDir];
    const a  = ACC[curDir];
    const d  = a.cop - a.bci;
    const ds = (d>=0?'+':'')+( d*100).toFixed(1)+'pp';
    const dc = d>=0?'#16a34a':'#dc2626';
    document.getElementById('count-note').innerHTML = `
      <b>${{LABELS[curDir]}}</b> &nbsp;
      BCI ${{(a.bci*100).toFixed(1)}}% → Copilot ${{(a.cop*100).toFixed(1)}}%
      <span style="color:${{dc}};font-weight:700;">${{ds}}</span><br>
      <span style="color:#94a3b8;font-size:10px;">
        ✓ Corrected: ${{c.correction}} &nbsp;|&nbsp;
        ✗ Failed: ${{c.failure}} &nbsp;|&nbsp;
        ✗ Both fail: ${{c.both_fail}} &nbsp;|&nbsp;
        ✓ Both ok: ${{c.both_ok}}
      </span>`;
  }}

  function resetSlider() {{
    const trials = DATA[curDir][curCat] || [];
    const T = trials.length > 0 ? trials[0].bci.length - 1 : 16;
    curTick = T;
    const sl = document.getElementById('tick-slider');
    sl.max = T; sl.value = T;
    document.getElementById('tick-val').textContent = T;
  }}

  // ── navigation ─────────────────────────────────────────────────────────────
  window.navTrial = function(d) {{
    const n = (DATA[curDir][curCat] || []).length;
    curIdx  = Math.max(0, Math.min(n-1, curIdx+d));
    curTick = (DATA[curDir][curCat][curIdx]?.bci.length ?? 17) - 1;
    pauseAnim();
    updateTrialCounter();
    updatePlot();
    resetSlider();
  }};

  window.randTrial = function() {{
    const n = (DATA[curDir][curCat] || []).length;
    if (n === 0) return;
    curIdx  = Math.floor(Math.random() * n);
    curTick = (DATA[curDir][curCat][curIdx]?.bci.length ?? 17) - 1;
    pauseAnim();
    updateTrialCounter();
    updatePlot();
    resetSlider();
  }};

  // ── slider ──────────────────────────────────────────────────────────────────
  window.onSlider = function(v) {{
    pauseAnim();
    curTick = parseInt(v);
    document.getElementById('tick-val').textContent = v;
    updatePlot();
  }};

  // ── animation ──────────────────────────────────────────────────────────────
  window.playAnim = function() {{
    pauseAnim();
    const trials = DATA[curDir][curCat] || [];
    if (trials.length === 0) return;
    const T = trials[curIdx].bci.length - 1;
    if (curTick >= T) curTick = 0;
    document.getElementById('btn-play').disabled  = true;
    document.getElementById('btn-pause').disabled = false;
    animTimer = setInterval(() => {{
      curTick++;
      document.getElementById('tick-slider').value = curTick;
      document.getElementById('tick-val').textContent = curTick;
      updatePlot();
      if (curTick >= T) pauseAnim();
    }}, 180);
  }};

  window.pauseAnim = function() {{
    if (animTimer) {{ clearInterval(animTimer); animTimer = null; }}
    document.getElementById('btn-play').disabled  = false;
    document.getElementById('btn-pause').disabled = true;
  }};

  window.resetAnim = function() {{
    pauseAnim();
    curTick = 0;
    document.getElementById('tick-slider').value = 0;
    document.getElementById('tick-val').textContent = 0;
    updatePlot();
  }};

  // ── boot ───────────────────────────────────────────────────────────────────
  rebuild();
}})();
</script>"""


# ── assemble HTML ──────────────────────────────────────────────────────────────

def save_html(results):
    acc_table  = build_accuracy_table(results)
    explorer   = build_explorer(results)

    # Plotly CDN for the explorer traces (just the JS lib, no figure)
    full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BCI Copilot Visualization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #F8FAFC; margin: 0; padding: 20px 28px;
    color: #1e293b; max-width: 100%;
  }}
  h1 {{ text-align:center; font-size:22px; margin:16px 0 4px; color:#0f172a; }}
  h2 {{ text-align:center; font-size:13px; color:#64748b; margin:0 0 24px;
        font-weight:normal; }}

  /* ── section headers ── */
  .section-title {{
    font-size:17px; font-weight:700; margin:28px 0 6px;
    padding-left:10px; border-left:4px solid #2563EB; color:#0f172a;
  }}
  .section-sub {{
    font-size:12px; color:#64748b; margin:0 0 14px 14px;
  }}
  .divider {{ border:none; border-top:1px solid #e2e8f0; margin:32px 0; }}

  /* ── accuracy table ── */
  .acc-table {{
    width:100%; border-collapse:collapse; font-size:13px;
    background:white; border-radius:8px;
    box-shadow:0 1px 4px rgba(0,0,0,0.07); overflow:hidden;
  }}
  .acc-table th {{
    background:#1e40af; color:white; padding:8px 12px;
    text-align:center; font-weight:600; font-size:12px;
  }}
  .acc-table td {{ padding:7px 12px; border-bottom:1px solid #f1f5f9; }}
  .acc-table tr:last-child td {{ border-bottom:none; }}
  .acc-table tr:hover {{ background:#f8fafc; }}
  .acc-table .total-row td {{ background:#eff6ff; border-top:2px solid #bfdbfe; }}
  .dir-cell {{ white-space:nowrap; }}
  .dir-badge-sm {{
    background:#1e40af; color:white; font-weight:700; font-size:11px;
    padding:2px 7px; border-radius:4px; margin-right:6px;
  }}
  .key-sm {{ font-size:11px; color:#64748b; font-family:monospace; }}
  .num {{ text-align:center; font-variant-numeric:tabular-nums; }}
  .delta {{ font-weight:700; }}
  .cat-head {{ font-size:11px; text-align:center; padding:8px 6px; }}
  .cat-head.corr {{ color:#16a34a; }}
  .cat-head.fail {{ color:#dc2626; }}
  .cat-head.bfail {{ color:#9333ea; }}
  .cat-head.bok   {{ color:#2563EB; }}
  .cat-count {{ font-size:12px; }}
  .cat-count.corr  {{ color:#16a34a; font-weight:600; }}
  .cat-count.fail  {{ color:#dc2626; font-weight:600; }}
  .cat-count.bfail {{ color:#9333ea; font-weight:600; }}
  .cat-count.bok   {{ color:#2563EB; font-weight:600; }}

  /* ── explorer ── */
  #explorer {{ width:100%; }}
  .ctrl-row {{
    display:flex; flex-wrap:wrap; gap:20px; align-items:flex-start;
    margin-bottom:18px;
  }}
  .ctrl-group {{ display:flex; flex-direction:column; gap:6px; }}
  .ctrl-label {{ font-size:11px; font-weight:600; color:#64748b;
                  text-transform:uppercase; letter-spacing:0.05em; }}
  .btn-group {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .sel-btn {{
    padding:5px 11px; border:1.5px solid #cbd5e1; border-radius:6px;
    background:white; cursor:pointer; font-size:12px; font-weight:500;
    color:#475569; transition:all 0.12s;
  }}
  .sel-btn:hover:not(:disabled) {{ border-color:#2563EB; color:#2563EB; background:#eff6ff; }}
  .sel-btn.active {{ background:#1e40af; color:white; border-color:#1e40af; }}
  .sel-btn:disabled {{ opacity:0.35; cursor:not-allowed; }}
  .cat-btn.active {{ background: var(--cat-color); border-color: var(--cat-color); }}
  .cnt {{ font-size:10px; opacity:0.8; }}
  .nav-group .ctrl-label {{ margin-bottom:2px; }}
  .nav-row {{ display:flex; align-items:center; gap:6px; }}
  .nav-btn {{
    padding:5px 10px; border:1.5px solid #cbd5e1; border-radius:6px;
    background:white; cursor:pointer; font-size:14px; color:#475569;
  }}
  .nav-btn:hover:not(:disabled) {{ border-color:#2563EB; color:#2563EB; }}
  .nav-btn:disabled {{ opacity:0.3; cursor:not-allowed; }}
  .trial-counter {{ font-size:13px; font-weight:600; color:#0f172a;
                    min-width:50px; text-align:center; }}

  .explorer-body {{
    display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;
  }}
  .info-panel {{
    flex:1; min-width:260px; display:flex; flex-direction:column; gap:12px;
  }}
  .cat-header-box {{
    font-size:14px; padding:8px 12px; background:white;
    border-radius:6px; border:1px solid #e2e8f0;
  }}
  .cat-desc {{ font-size:12px; color:#64748b; }}
  .outcome-info {{
    background:white; border:1px solid #e2e8f0; border-radius:6px;
    padding:10px 14px; display:flex; flex-direction:column; gap:6px;
  }}
  .pred-row {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
  .pred-label {{ color:#64748b; min-width:130px; }}
  .pred-val {{ font-weight:700; font-size:14px; }}
  .pred-val.ok    {{ color:#16a34a; }}
  .pred-val.wrong {{ color:#dc2626; }}
  .pred-val.target {{ color:#1e40af; }}

  .playback-box {{
    background:white; border:1px solid #e2e8f0; border-radius:6px;
    padding:10px 14px; display:flex; flex-direction:column; gap:10px;
  }}
  .tick-row {{ display:flex; align-items:center; gap:8px; }}
  .tick-label {{ font-size:12px; color:#64748b; white-space:nowrap; }}
  .tick-val {{ font-size:12px; font-weight:600; min-width:20px; text-align:right; }}
  .play-row {{ display:flex; gap:8px; }}
  .play-btn {{
    padding:5px 12px; border:1.5px solid #cbd5e1; border-radius:6px;
    background:white; cursor:pointer; font-size:12px; color:#475569;
    font-weight:500;
  }}
  .play-btn:hover:not(:disabled) {{ border-color:#2563EB; color:#2563EB; }}
  .play-btn:disabled {{ opacity:0.35; cursor:not-allowed; }}

  .legend-box {{
    display:flex; gap:16px; font-size:12px; color:#475569;
    padding:6px 0; align-items:center;
  }}
  .leg-item {{ display:flex; align-items:center; gap:6px; }}
  .leg-line {{ display:inline-block; width:28px; height:3px; border-radius:2px; }}
  .bci-line {{ background:#2563EB; }}
  .cop-line {{
    background:repeating-linear-gradient(90deg,#EA580C 0,#EA580C 5px,
    transparent 5px,transparent 8px);
    height:3px;
  }}

  .count-note {{
    font-size:12px; color:#475569; background:#f8fafc;
    border:1px solid #e2e8f0; border-radius:6px; padding:8px 12px;
    line-height:1.8;
  }}
</style>
</head>
<body>

<h1>BCI Copilot — Cursor Trajectory Visualization</h1>
<h2>Supervised LSTM Copilot &nbsp;·&nbsp;
    Data: online_arm_trajectories.csv &nbsp;·&nbsp;
    7 subjects · 11,760 trials · 8 directions</h2>

<div class="section-title">Performance Summary</div>
<div class="section-sub">
  Per-direction accuracy and outcome counts across all 7 subjects.
  Outcome counts reflect the full dataset (all 1,470 trials per direction).
</div>
{acc_table}

<hr class="divider">

<div class="section-title">Trajectory Explorer</div>
<div class="section-sub">
  Select a direction and outcome category to browse real trial trajectories.
  Up to {N_SAMPLE} trials per category are available for replay.
  Use &#8592; &#8594; to step through trials or &#x21BB; for a random trial.
</div>
{explorer}

</body>
</html>"""

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    print("  Double-click to open in browser (self-contained, works offline)")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BCI Copilot Visualization — Supervised LSTM")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {CSV_PATH}")
    print()

    print("Loading supervised LSTM copilot...")
    model = LSTMCopilot()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("Done.\n")

    by_dir, by_subj = load_trials(CSV_PATH)
    print()

    results = compute_all(model, by_dir, by_subj)
    print()

    print("Assembling HTML...")
    save_html(results)

    print()
    print("=" * 60)
    print("Per-direction summary:")
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
