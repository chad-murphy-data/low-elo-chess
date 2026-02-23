"""Three-panel research dashboard template.

Panel 1: Exploratory Landscape
Panel 2: Hazard Model
Panel 3: Live Coefficient Plot
"""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess Blunder Hazard Model</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2129;
    --border: #30363d;
    --text: #c9d1d9;
    --text2: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --orange: #db6d28;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }

  /* Header */
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .header h1 { font-size: 15px; font-weight: 600; }
  .header h1 .icon { color: var(--accent); margin-right: 6px; }
  .header-right { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--text2); }
  .badge {
    font-size: 10px; padding: 2px 10px; border-radius: 10px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
  }
  .badge-ok { background: #1b3a2d; color: var(--green); }
  .badge-wait { background: #2a1f0f; color: var(--yellow); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

  /* Layout */
  .container { max-width: 1600px; margin: 0 auto; padding: 16px 24px; }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .panel-header {
    background: var(--surface2);
    padding: 10px 16px;
    font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .panel-header .panel-num {
    background: var(--accent); color: var(--bg);
    width: 20px; height: 20px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; margin-right: 8px;
  }
  .panel-body { padding: 16px; }
  .panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .panel-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

  /* Chart cards */
  .chart-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }
  .chart-title {
    font-size: 11px; font-weight: 600; color: var(--text2);
    margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .chart-subtitle {
    font-size: 10px; color: var(--text2); margin-top: -4px; margin-bottom: 8px;
    font-style: italic;
  }
  .chart-container { position: relative; width: 100%; height: 320px; }
  .chart-container.tall { height: 380px; }

  /* Annotation callout */
  .callout {
    position: absolute;
    background: rgba(88,166,255,0.15);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 10px;
    color: var(--accent);
    pointer-events: none;
    white-space: nowrap;
  }

  /* Coefficient plot */
  .coef-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
  }
  .coef-row:last-child { border-bottom: none; }
  .coef-name { width: 220px; font-size: 12px; color: var(--text2); text-align: right; flex-shrink: 0; }
  .coef-bar-area { flex: 1; position: relative; height: 24px; }
  .coef-zero {
    position: absolute; left: 50%; top: 0; bottom: 0;
    border-left: 1px dashed var(--border);
  }
  .coef-ci {
    position: absolute; top: 8px; height: 8px;
    border-radius: 4px; opacity: 0.5;
  }
  .coef-dot {
    position: absolute; top: 6px;
    width: 12px; height: 12px; border-radius: 50%;
    transform: translateX(-50%);
  }
  .coef-val { width: 80px; font-size: 11px; font-family: monospace; flex-shrink: 0; }
  .coef-p { width: 60px; font-size: 10px; color: var(--text2); text-align: right; flex-shrink: 0; }

  /* Slider */
  .slider-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px;
    background: var(--surface2);
    border-top: 1px solid var(--border);
    font-size: 12px; color: var(--text2);
  }
  .slider-row input[type=range] {
    flex: 1; accent-color: var(--accent);
  }
  .slider-value { font-weight: 600; color: var(--accent); min-width: 60px; }

  /* Clock-by-band mini table */
  .mini-table { font-size: 11px; width: 100%; border-collapse: collapse; }
  .mini-table th { color: var(--text2); font-weight: 600; text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .mini-table td { padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .mini-table .sig { color: var(--green); font-weight: 600; }
  .mini-table .ns { color: var(--text2); }

  /* Dropdown */
  .dropdown {
    background: var(--surface2); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 2px 8px; font-size: 11px;
  }

  /* Legend */
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 4px; color: var(--text2); }
  .legend-swatch { width: 12px; height: 3px; border-radius: 2px; }

  /* Findings section */
  .findings-panel { font-size: 12px; color: var(--text2); padding: 12px 16px; }
  .findings-panel strong { color: var(--accent); }

  /* Empty state */
  .empty { text-align: center; padding: 60px 20px; color: var(--text2); font-size: 13px; }

  @media (max-width: 1100px) {
    .panel-grid, .panel-grid-3 { grid-template-columns: 1fr; }
    .container { padding: 8px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span class="icon">&#9818;</span>Chess Blunder Hazard Model</h1>
  <div class="header-right">
    <span id="sample-info"></span>
    <span id="status-badge" class="badge badge-wait">loading</span>
  </div>
</div>

<div class="container">

  <!-- ============================================================ -->
  <!-- PANEL 1: Exploratory Landscape -->
  <!-- ============================================================ -->
  <div class="panel" id="panel1">
    <div class="panel-header">
      <span><span class="panel-num">1</span>Exploratory Landscape</span>
    </div>
    <div class="panel-body">
      <div class="panel-grid">
        <div class="chart-card">
          <div class="chart-title">Blunder Rate by ELO and Piece Type</div>
          <div class="chart-subtitle">Smoothed curves (LOESS) -- continuous ELO axis, not binned</div>
          <div class="chart-container" id="p1a-container">
            <canvas id="p1a-chart"></canvas>
          </div>
          <div class="legend" id="p1a-legend"></div>
        </div>

        <div class="chart-card">
          <div class="chart-title">Hang Rate vs Punishment Rate by ELO</div>
          <div class="chart-subtitle">
            Gap = "free material nobody noticed"
            <select class="dropdown" id="p1b-piece-filter" style="margin-left:8px;">
              <option value="all">All pieces</option>
              <option value="pawn">Pawns</option>
              <option value="knight">Knights</option>
              <option value="bishop">Bishops</option>
              <option value="rook">Rooks</option>
              <option value="queen">Queens</option>
            </select>
          </div>
          <div class="chart-container" id="p1b-container">
            <canvas id="p1b-chart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- PANEL 2: Hazard Model -->
  <!-- ============================================================ -->
  <div class="panel" id="panel2">
    <div class="panel-header">
      <span><span class="panel-num">2</span>Hazard Model -- When Do Blunders Arrive?</span>
    </div>
    <div class="panel-body">
      <div class="panel-grid">
        <div class="chart-card">
          <div class="chart-title">Survival Curve: Games Still Blunder-Free</div>
          <div class="chart-subtitle">Kaplan-Meier estimate by ELO band</div>
          <div class="chart-container tall">
            <canvas id="p2a-chart"></canvas>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-title">First Blunder: Clock vs Move Number</div>
          <div class="chart-subtitle">Scatter + LOESS smooth by ELO band. Flat = blunder independent of clock.</div>
          <div class="chart-container tall" id="p2b-container">
            <canvas id="p2b-chart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- PANEL 3: Live Coefficient Plot -->
  <!-- ============================================================ -->
  <div class="panel" id="panel3">
    <div class="panel-header">
      <span><span class="panel-num">3</span>Live Coefficient Plot</span>
      <span id="p3-info" style="font-size:10px;color:var(--text2);font-weight:400;"></span>
    </div>
    <div class="panel-body">
      <div class="panel-grid">
        <div class="chart-card">
          <div class="chart-title">P(blunder) ~ ELO + clock + move# + recency + ELO:clock</div>
          <div class="chart-subtitle">Dot = coefficient, bar = 95% CI. Green = significant in expected direction.</div>
          <div id="coef-plot"></div>
        </div>

        <div class="chart-card">
          <div class="chart-title">Clock Effect by ELO Band</div>
          <div class="chart-subtitle">Separate logistic: P(blunder) ~ clock_remaining, per band. Key hypothesis test.</div>
          <div id="clock-by-band"></div>
        </div>
      </div>
    </div>
    <div class="slider-row">
      <span>Sample size:</span>
      <input type="range" id="sample-slider" min="500" max="200000" step="500" value="200000">
      <span class="slider-value" id="slider-label">Full</span>
      <button id="play-btn" style="background:var(--accent);color:var(--bg);border:none;border-radius:4px;padding:3px 12px;font-size:11px;cursor:pointer;font-weight:600;">&#9654; Animate</button>
    </div>
  </div>

</div>

<script>
// =====================================================================
// GLOBALS & THEME
// =====================================================================
const ELO_BANDS = ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"];
const BAND_COLORS = {
  "500-700":  "#f85149",
  "700-900":  "#db6d28",
  "900-1100": "#d29922",
  "1100-1300":"#58a6ff",
  "1300-1500":"#3fb950",
};
const PIECE_COLORS = {
  pawn:   "#8b949e",
  knight: "#bc8cff",
  bishop: "#d29922",
  rook:   "#f85149",
  queen:  "#58a6ff",
};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
Chart.defaults.font.size = 11;

let p1aChart = null, p1bChart = null, p2aChart = null, p2bChart = null;
let panel1Data = null, panel2Data = null;
let sliderDebounce = null;
let isAnimating = false;

// =====================================================================
// PANEL 1: Exploratory Landscape
// =====================================================================

function renderPanel1(data) {
  if (!data || !data.ready) return;
  panel1Data = data;

  // --- Chart A: Blunder rate by ELO per piece type ---
  const datasets = [];
  const legendEl = document.getElementById('p1a-legend');
  legendEl.innerHTML = '';

  for (const [pt, curve] of Object.entries(data.piece_curves || {})) {
    const color = PIECE_COLORS[pt] || '#8b949e';
    datasets.push({
      label: pt.charAt(0).toUpperCase() + pt.slice(1),
      data: curve.elo.map((x, i) => ({x, y: curve.rate[i] * 100})),
      borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.4,
    });
    legendEl.innerHTML += `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${pt}</span>`;
  }

  // Add hung/missed direction as dashed lines
  for (const [dir, curve] of Object.entries(data.direction_curves || {})) {
    const color = dir === 'hung' ? '#f8514980' : '#58a6ff80';
    datasets.push({
      label: dir === 'hung' ? 'Hung piece' : 'Missed capture',
      data: curve.elo.map((x, i) => ({x, y: curve.rate[i] * 100})),
      borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      borderDash: [6, 3],
      pointRadius: 0,
      tension: 0.4,
    });
    legendEl.innerHTML += `<span class="legend-item"><span class="legend-swatch" style="background:${color};height:1px;border-top:2px dashed ${color};"></span>${dir}</span>`;
  }

  if (!p1aChart) {
    p1aChart = new Chart(document.getElementById('p1a-chart'), {
      type: 'line',
      data: {datasets},
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: {duration: 300},
        plugins: {legend: {display: false}, tooltip: {mode: 'nearest', intersect: false}},
        scales: {
          x: {type: 'linear', title: {display: true, text: 'ELO Rating'}, grid: {color: '#30363d40'}},
          y: {title: {display: true, text: 'Blunder Rate %'}, beginAtZero: true, grid: {color: '#30363d40'},
              ticks: {callback: v => v.toFixed(1) + '%'}},
        },
      },
    });
  } else {
    p1aChart.data.datasets = datasets;
    p1aChart.update('none');
  }

  // --- Chart B: Hang vs Punishment ---
  renderPanel1B(data, 'all');
}

function renderPanel1B(data, pieceFilter) {
  const datasets = [];

  // Hang rate curve
  if (data.hang_curve && data.hang_curve.elo.length > 0) {
    datasets.push({
      label: 'Hang Rate (created hanging piece)',
      data: data.hang_curve.elo.map((x, i) => ({x, y: data.hang_curve.rate[i] * 100})),
      borderColor: '#f85149',
      backgroundColor: 'rgba(248,81,73,0.1)',
      borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
    });
  }

  // Punishment rate curve (overall or by piece)
  if (pieceFilter !== 'all' && data.punishment_by_piece && data.punishment_by_piece[pieceFilter]) {
    const curve = data.punishment_by_piece[pieceFilter];
    datasets.push({
      label: `Punishment Rate (${pieceFilter})`,
      data: curve.elo.map((x, i) => ({x, y: curve.rate[i] * 100})),
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63,185,80,0.1)',
      borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
    });
  } else if (data.punishment_curve && data.punishment_curve.elo.length > 0) {
    datasets.push({
      label: 'Punishment Rate (opponent captured)',
      data: data.punishment_curve.elo.map((x, i) => ({x, y: data.punishment_curve.rate[i] * 100})),
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63,185,80,0.1)',
      borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
    });
  }

  if (!p1bChart) {
    p1bChart = new Chart(document.getElementById('p1b-chart'), {
      type: 'line',
      data: {datasets},
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: {duration: 300},
        plugins: {
          legend: {display: true, position: 'bottom', labels: {boxWidth: 12, font: {size: 10}}},
          tooltip: {mode: 'nearest', intersect: false},
        },
        scales: {
          x: {type: 'linear', title: {display: true, text: 'ELO Rating'}, grid: {color: '#30363d40'}},
          y: {title: {display: true, text: 'Rate %'}, beginAtZero: true, grid: {color: '#30363d40'},
              ticks: {callback: v => v.toFixed(1) + '%'}},
        },
      },
    });
  } else {
    p1bChart.data.datasets = datasets;
    p1bChart.update('none');
  }
}

document.getElementById('p1b-piece-filter').addEventListener('change', function() {
  if (panel1Data) renderPanel1B(panel1Data, this.value);
});


// =====================================================================
// PANEL 2: Hazard Model
// =====================================================================

function renderPanel2(data) {
  if (!data || !data.ready) return;
  panel2Data = data;

  // --- Chart A: Survival curves ---
  const survDatasets = [];
  for (const band of ELO_BANDS) {
    const curve = data.survival_curves[band];
    if (!curve) continue;
    const color = BAND_COLORS[band];
    survDatasets.push({
      label: `${band} (n=${curve.n})`,
      data: curve.time.map((t, i) => ({x: t, y: curve.survival[i] * 100})),
      borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0,
      stepped: 'before',
    });
  }

  if (!p2aChart) {
    p2aChart = new Chart(document.getElementById('p2a-chart'), {
      type: 'line',
      data: {datasets: survDatasets},
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: {duration: 300},
        plugins: {
          legend: {display: true, position: 'bottom', labels: {boxWidth: 12, font: {size: 10}}},
          tooltip: {mode: 'nearest', intersect: false,
            callbacks: {label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}% blunder-free at move ${ctx.parsed.x}`}},
        },
        scales: {
          x: {type: 'linear', title: {display: true, text: 'Move Number (ply)'}, grid: {color: '#30363d40'},
              min: 0},
          y: {title: {display: true, text: '% Games Still Blunder-Free'}, grid: {color: '#30363d40'},
              min: 0, max: 100, ticks: {callback: v => v + '%'}},
        },
      },
    });
  } else {
    p2aChart.data.datasets = survDatasets;
    p2aChart.update('none');
  }

  // --- Chart B: Blunder arrival scatter ---
  const scatterDatasets = [];
  for (const band of ELO_BANDS) {
    const pts = data.scatter_by_band[band];
    const loess = data.loess_by_band[band];
    const color = BAND_COLORS[band];
    if (pts) {
      scatterDatasets.push({
        label: band,
        data: pts.clock.map((c, i) => ({x: c, y: pts.move[i]})),
        backgroundColor: color + '40',
        borderColor: color + '60',
        pointRadius: 2.5,
        pointHoverRadius: 4,
        type: 'scatter',
        showLine: false,
      });
    }
    if (loess) {
      scatterDatasets.push({
        label: band + ' trend',
        data: loess.clock.map((c, i) => ({x: c, y: loess.move[i]})),
        borderColor: color,
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.4,
        type: 'line',
      });
    }
  }

  if (!p2bChart) {
    p2bChart = new Chart(document.getElementById('p2b-chart'), {
      type: 'scatter',
      data: {datasets: scatterDatasets},
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: {duration: 300},
        plugins: {
          legend: {display: true, position: 'bottom', labels: {
            boxWidth: 10, font: {size: 10},
            filter: item => !item.text.includes('trend'),
          }},
          tooltip: {mode: 'nearest', intersect: true},
        },
        scales: {
          x: {type: 'linear', title: {display: true, text: 'Clock Remaining (sec)'}, grid: {color: '#30363d40'}},
          y: {title: {display: true, text: 'Move # of First Blunder'}, grid: {color: '#30363d40'}},
        },
      },
    });
  } else {
    p2bChart.data.datasets = scatterDatasets;
    p2bChart.update('none');
  }
}


// =====================================================================
// PANEL 3: Coefficient Plot
// =====================================================================

function renderPanel3(data) {
  if (!data || !data.ready) return;

  document.getElementById('p3-info').textContent =
    `n = ${data.n.toLocaleString()} / ${data.total_n.toLocaleString()}   pseudo-R\u00B2 = ${data.pseudo_r2.toFixed(4)}`;

  // Update slider max
  const slider = document.getElementById('sample-slider');
  slider.max = data.total_n;
  if (parseInt(slider.value) > data.total_n) slider.value = data.total_n;

  // --- Coefficient forest plot ---
  const plotEl = document.getElementById('coef-plot');
  const coefs = data.coefficients;

  // Find the range for scaling
  let minVal = 0, maxVal = 0;
  for (const c of coefs) {
    minVal = Math.min(minVal, c.ci_lo);
    maxVal = Math.max(maxVal, c.ci_hi);
  }
  const range = Math.max(Math.abs(minVal), Math.abs(maxVal)) * 1.3 || 0.001;

  const LABELS = {
    player_elo: 'Player ELO',
    clock_remaining: 'Clock Remaining',
    move_number: 'Move Number',
    moves_since_piece_last_moved: 'Piece Recency',
    elo_x_clock: 'ELO \u00D7 Clock (interaction)',
  };

  let html = '';
  for (const c of coefs) {
    const label = LABELS[c.name] || c.name;
    const colors = {green: 'var(--green)', grey: 'var(--text2)', red: 'var(--red)'};
    const col = colors[c.color] || colors.grey;
    const pStr = c.p < 0.001 ? '<0.001 ***' : c.p < 0.01 ? c.p.toFixed(3) + ' **' : c.p < 0.05 ? c.p.toFixed(3) + ' *' : c.p.toFixed(3);

    // Scale to 0-100% with zero at 50%
    const ciLeft = ((c.ci_lo / range + 1) / 2 * 100);
    const ciRight = ((c.ci_hi / range + 1) / 2 * 100);
    const dotPos = ((c.coef / range + 1) / 2 * 100);

    html += `<div class="coef-row">
      <div class="coef-name">${label}</div>
      <div class="coef-bar-area">
        <div class="coef-zero"></div>
        <div class="coef-ci" style="left:${ciLeft}%;width:${ciRight-ciLeft}%;background:${col};"></div>
        <div class="coef-dot" style="left:${dotPos}%;background:${col};"></div>
      </div>
      <div class="coef-val" style="color:${col};">${c.coef >= 0 ? '+' : ''}${c.coef.toFixed(5)}</div>
      <div class="coef-p">${pStr}</div>
    </div>`;
  }
  plotEl.innerHTML = html;

  // --- Clock-by-band table ---
  const tableEl = document.getElementById('clock-by-band');
  let thtml = '<table class="mini-table"><thead><tr><th>ELO Band</th><th>Clock Coef</th><th>95% CI</th><th>p-value</th><th>n</th></tr></thead><tbody>';

  // Detect crossover: first band where clock is significant
  let crossoverBand = null;

  for (const band of ELO_BANDS) {
    const cb = data.clock_by_band[band];
    if (!cb) {
      thtml += `<tr><td>${band}</td><td colspan="4" style="color:var(--text2)">Insufficient data</td></tr>`;
      continue;
    }
    const sig = cb.p < 0.05;
    if (sig && !crossoverBand) crossoverBand = band;

    const cls = sig ? 'sig' : 'ns';
    const pStr = cb.p < 0.001 ? '<0.001' : cb.p.toFixed(4);
    thtml += `<tr>
      <td>${band}</td>
      <td class="${cls}">${cb.coef >= 0 ? '+' : ''}${cb.coef.toFixed(5)}</td>
      <td>[${cb.ci_lo.toFixed(5)}, ${cb.ci_hi.toFixed(5)}]</td>
      <td class="${cls}">${pStr} ${sig ? '*' : ''}</td>
      <td>${cb.n.toLocaleString()}</td>
    </tr>`;
  }
  thtml += '</tbody></table>';

  if (crossoverBand) {
    thtml += `<div style="margin-top:8px;padding:6px 10px;background:rgba(88,166,255,0.1);border:1px solid var(--accent);border-radius:4px;font-size:11px;color:var(--accent);">
      <strong>&#8593; Clock effect begins at ${crossoverBand}</strong> &mdash;
      below this ELO, time pressure does not significantly predict blunders.
    </div>`;
  } else {
    thtml += `<div style="margin-top:8px;font-size:11px;color:var(--text2);">
      No significant clock effect detected in any band (wide CIs suggest more data needed).
    </div>`;
  }

  tableEl.innerHTML = thtml;
}


// =====================================================================
// SLIDER & ANIMATION
// =====================================================================

document.getElementById('sample-slider').addEventListener('input', function() {
  const val = parseInt(this.value);
  document.getElementById('slider-label').textContent = val >= parseInt(this.max) - 100 ? 'Full' : val.toLocaleString();
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => fetchPanel3(val >= parseInt(this.max) - 100 ? 0 : val), 200);
});

document.getElementById('play-btn').addEventListener('click', function() {
  if (isAnimating) {
    isAnimating = false;
    this.innerHTML = '&#9654; Animate';
    return;
  }
  isAnimating = true;
  this.innerHTML = '&#9632; Stop';
  const slider = document.getElementById('sample-slider');
  const maxVal = parseInt(slider.max);
  const steps = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, maxVal];

  let i = 0;
  function tick() {
    if (!isAnimating || i >= steps.length) {
      isAnimating = false;
      document.getElementById('play-btn').innerHTML = '&#9654; Animate';
      return;
    }
    const n = Math.min(steps[i], maxVal);
    slider.value = n;
    document.getElementById('slider-label').textContent = n >= maxVal ? 'Full' : n.toLocaleString();
    fetchPanel3(n >= maxVal ? 0 : n).then(() => {
      i++;
      setTimeout(tick, 800);
    });
  }
  tick();
});


// =====================================================================
// DATA FETCHING
// =====================================================================

async function fetchPanel3(n) {
  try {
    const url = n > 0 ? `/api/panel3?n=${n}` : '/api/panel3';
    const resp = await fetch(url);
    const data = await resp.json();
    renderPanel3(data);
  } catch(e) {}
}

async function fetchAll() {
  try {
    const [statusResp, p1Resp, p2Resp, p3Resp] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/panel1'),
      fetch('/api/panel2'),
      fetch('/api/panel3'),
    ]);

    const status = await statusResp.json();
    const badge = document.getElementById('status-badge');
    if (status.phase === 'complete') {
      badge.className = 'badge badge-ok';
      badge.textContent = 'COMPLETE';
    } else {
      badge.className = 'badge badge-wait';
      badge.textContent = status.phase.toUpperCase();
    }
    document.getElementById('sample-info').textContent =
      `${status.n_moves.toLocaleString()} moves \u00B7 ${status.n_games.toLocaleString()} games \u00B7 ${status.pgn_files} PGN files`;

    renderPanel1(await p1Resp.json());
    renderPanel2(await p2Resp.json());
    renderPanel3(await p3Resp.json());
  } catch(e) {}
}

// Initial load
fetchAll();
// Refresh every 15 seconds (slower -- heavy endpoints)
setInterval(fetchAll, 15000);
</script>
</body>
</html>"""
