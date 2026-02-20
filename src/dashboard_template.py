"""Single-file HTML template for the dashboard. Polls /api/status every 3s."""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess Blunder Analysis</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232733;
    --border: #2e3348;
    --text: #e1e4ed;
    --text2: #8b90a5;
    --accent: #6c9bff;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
    --orange: #fb923c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
  }
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .header h1 span { color: var(--accent); }
  .phase-badge {
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .phase-waiting { background: var(--surface2); color: var(--text2); }
  .phase-collecting { background: #1e3a5f; color: var(--accent); }
  .phase-processing { background: #3b2f1a; color: var(--orange); }
  .phase-analyzing { background: #1a3b2f; color: var(--green); }
  .phase-complete { background: #1a3b1a; color: var(--green); }
  .pulse { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

  .container { max-width: 1200px; margin: 0 auto; padding: 24px 32px; }

  /* Grid layout */
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-full { grid-column: 1 / -1; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }
  .card-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text2);
    margin-bottom: 12px;
  }

  /* Stats */
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; }
  .stat-label { color: var(--text2); font-size: 13px; }
  .stat-value { font-weight: 600; font-size: 13px; }
  .stat-big { font-size: 28px; font-weight: 700; color: var(--accent); margin: 4px 0 8px; }

  /* Progress bars */
  .bar-container { margin: 8px 0; }
  .bar-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
  .bar-bg {
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .bar-fill-blue { background: var(--accent); }
  .bar-fill-green { background: var(--green); }
  .bar-fill-yellow { background: var(--yellow); }
  .bar-fill-orange { background: var(--orange); }
  .bar-fill-red { background: var(--red); }

  /* ELO band table */
  .band-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .band-table th {
    text-align: left;
    font-weight: 600;
    color: var(--text2);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .band-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .band-table tr:last-child td { border-bottom: none; }
  .blunder-rate {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
  }
  .rate-low { background: #1a3b1a; color: var(--green); }
  .rate-mid { background: #3b3b1a; color: var(--yellow); }
  .rate-high { background: #3b1a1a; color: var(--red); }

  /* Model summary */
  .model-output {
    font-size: 12px;
    line-height: 1.5;
    white-space: pre-wrap;
    font-family: inherit;
    color: var(--text);
    max-height: 500px;
    overflow-y: auto;
    padding: 12px;
    background: var(--bg);
    border-radius: 6px;
    border: 1px solid var(--border);
  }

  /* Figures */
  .figures-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .figure-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .figure-card:hover { border-color: var(--accent); }
  .figure-card img {
    width: 100%;
    display: block;
    background: white;
  }
  .figure-card .fig-label {
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text2);
  }

  /* Lightbox */
  .lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    z-index: 200;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .lightbox.active { display: flex; }
  .lightbox img {
    max-width: 90vw;
    max-height: 90vh;
    border-radius: 8px;
    background: white;
  }

  /* Empty state */
  .empty {
    text-align: center;
    padding: 40px;
    color: var(--text2);
    font-size: 13px;
  }
  .empty-icon { font-size: 32px; margin-bottom: 8px; }

  /* Findings markdown (very basic rendering) */
  .findings { font-size: 13px; line-height: 1.7; }
  .findings h1 { font-size: 18px; margin: 16px 0 8px; color: var(--accent); }
  .findings h2 { font-size: 15px; margin: 14px 0 6px; color: var(--text); }
  .findings h3 { font-size: 13px; margin: 10px 0 4px; color: var(--text); }
  .findings ul { margin-left: 20px; }
  .findings table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px; }
  .findings th, .findings td { padding: 6px 10px; border: 1px solid var(--border); text-align: left; }
  .findings th { background: var(--surface2); font-weight: 600; }
  .findings hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
  .findings strong { color: var(--accent); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  @media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
    .figures-grid { grid-template-columns: 1fr; }
    .container { padding: 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span>&#9818;</span> Chess Blunder Analysis</h1>
  <div>
    <span id="phase-badge" class="phase-badge phase-waiting">waiting</span>
  </div>
</div>

<div class="container">

  <!-- Row 1: Overview + Collection -->
  <div class="grid">
    <div class="card" id="overview-card">
      <div class="card-title">Dataset Overview</div>
      <div id="overview-content">
        <div class="empty">
          <div class="empty-icon">&#9812;</div>
          Waiting for data...<br>
          Run <code>python run_pipeline.py</code> in another terminal
        </div>
      </div>
    </div>

    <div class="card" id="collection-card">
      <div class="card-title">Collection Progress</div>
      <div id="collection-content">
        <div class="empty">
          <div class="empty-icon">&#8987;</div>
          No collection started yet
        </div>
      </div>
    </div>
  </div>

  <!-- Row 2: ELO Band Table -->
  <div class="grid">
    <div class="card grid-full" id="bands-card" style="display:none;">
      <div class="card-title">Blunder Rate by ELO Band</div>
      <div id="bands-content"></div>
    </div>
  </div>

  <!-- Row 3: Figures -->
  <div class="card" id="figures-card" style="display:none; margin-bottom:16px;">
    <div class="card-title">Plots</div>
    <div class="figures-grid" id="figures-content"></div>
  </div>

  <!-- Row 4: Model Summary + Findings -->
  <div class="grid" id="results-row" style="display:none;">
    <div class="card">
      <div class="card-title">Model Results</div>
      <pre class="model-output" id="model-content"></pre>
    </div>
    <div class="card">
      <div class="card-title">Key Findings</div>
      <div class="findings" id="findings-content"></div>
    </div>
  </div>

</div>

<!-- Lightbox for figure zoom -->
<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const ELO_BANDS = ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"];
const BAR_COLORS = ["bar-fill-red", "bar-fill-orange", "bar-fill-yellow", "bar-fill-blue", "bar-fill-green"];
const TARGET_USERS = 200;

// Simple markdown to HTML (covers the basics we need)
function md(text) {
  if (!text) return "";
  let html = text
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '<li>$1</li>');

  // Wrap consecutive <li> in <ul>
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  // Simple table parsing
  html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/gm, (match, header, sep, body) => {
    const ths = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(row => {
      const tds = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
      return `<tr>${tds}</tr>`;
    }).join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
  });

  // Paragraphs (lines not already wrapped)
  html = html.split('\n').map(line => {
    if (/^<[hul\-t]/.test(line) || line.trim() === '') return line;
    return `<p>${line}</p>`;
  }).join('\n');

  return html;
}

// Cache for loaded figure images
const figCache = {};

function renderOverview(data) {
  const el = document.getElementById("overview-content");
  if (!data.dataset) {
    if (data.collection && data.collection.pgn_files > 0) {
      el.innerHTML = `
        <div class="stat-big">${data.collection.pgn_files}</div>
        <div class="stat-label">PGN files downloaded</div>
        <p style="color:var(--text2); font-size:12px; margin-top:12px;">
          Run <code>python run_pipeline.py features</code> to build the dataset
        </p>`;
    }
    return;
  }
  const d = data.dataset;
  el.innerHTML = `
    <div class="stat-big">${d.total_moves.toLocaleString()}</div>
    <div class="stat-label" style="margin-bottom:16px;">move observations</div>
    <div class="stat-row"><span class="stat-label">Games</span><span class="stat-value">${d.total_games.toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Blunder rate</span><span class="stat-value" style="color:var(--red)">${d.blunder_rate !== null ? d.blunder_rate + '%' : '—'}</span></div>
    <div class="stat-row"><span class="stat-label">Mistake rate</span><span class="stat-value" style="color:var(--orange)">${d.mistake_rate !== null ? d.mistake_rate + '%' : '—'}</span></div>
    <div class="stat-row"><span class="stat-label">ELO range</span><span class="stat-value">${d.elo_stats.min || '?'} — ${d.elo_stats.max || '?'}</span></div>
    <div class="stat-row"><span class="stat-label">Mean ELO</span><span class="stat-value">${d.elo_stats.mean || '?'}</span></div>
  `;
}

function renderCollection(data) {
  const el = document.getElementById("collection-content");
  const c = data.collection;
  if (!c.checkpoint) {
    if (c.pgn_files > 0) {
      el.innerHTML = `<div class="stat-row"><span class="stat-label">PGN files</span><span class="stat-value">${c.pgn_files}</span></div>`;
    }
    return;
  }
  const cp = c.checkpoint;
  let html = `
    <div class="stat-row"><span class="stat-label">Users found</span><span class="stat-value">${cp.total_users}</span></div>
    <div class="stat-row"><span class="stat-label">Users processed</span><span class="stat-value">${cp.processed_users}</span></div>
    <div class="stat-row" style="margin-bottom:12px"><span class="stat-label">PGN files</span><span class="stat-value">${c.pgn_files}</span></div>
  `;
  for (let i = 0; i < ELO_BANDS.length; i++) {
    const band = ELO_BANDS[i];
    const count = cp.bands[band] || 0;
    const pct = Math.min(100, (count / TARGET_USERS) * 100);
    html += `
      <div class="bar-container">
        <div class="bar-label">
          <span>${band}</span>
          <span>${count} / ${TARGET_USERS}</span>
        </div>
        <div class="bar-bg"><div class="bar-fill ${BAR_COLORS[i]}" style="width:${pct}%"></div></div>
      </div>`;
  }
  el.innerHTML = html;
}

function renderBands(data) {
  const card = document.getElementById("bands-card");
  const el = document.getElementById("bands-content");
  if (!data.dataset || !data.dataset.elo_bands || Object.keys(data.dataset.elo_bands).length === 0) {
    card.style.display = "none";
    return;
  }
  card.style.display = "";
  const bands = data.dataset.elo_bands;
  let html = `<table class="band-table">
    <thead><tr><th>ELO Band</th><th>Blunder Rate</th><th>Observations</th><th></th></tr></thead><tbody>`;
  for (const band of ELO_BANDS) {
    if (!bands[band]) continue;
    const b = bands[band];
    const rateClass = b.blunder_rate > 20 ? 'rate-high' : b.blunder_rate > 12 ? 'rate-mid' : 'rate-low';
    const barPct = Math.min(100, b.blunder_rate * 2.5);
    html += `<tr>
      <td style="font-weight:600">${band}</td>
      <td><span class="blunder-rate ${rateClass}">${b.blunder_rate}%</span></td>
      <td>${b.count.toLocaleString()}</td>
      <td style="width:40%"><div class="bar-bg"><div class="bar-fill bar-fill-red" style="width:${barPct}%"></div></div></td>
    </tr>`;
  }
  html += '</tbody></table>';
  el.innerHTML = html;
}

function renderFigures(data) {
  const card = document.getElementById("figures-card");
  const el = document.getElementById("figures-content");
  if (!data.figures || data.figures.length === 0) {
    card.style.display = "none";
    return;
  }
  card.style.display = "";

  // Only re-render if figure list changed
  const figKey = data.figures.map(f => f.filename).join(",");
  if (el.dataset.figKey === figKey) return;
  el.dataset.figKey = figKey;

  el.innerHTML = "";
  for (const fig of data.figures) {
    const div = document.createElement("div");
    div.className = "figure-card";
    div.innerHTML = `<div class="fig-label">${fig.name}</div><div style="min-height:200px;background:var(--surface2);display:flex;align-items:center;justify-content:center;color:var(--text2);font-size:12px;">Loading...</div>`;
    div.onclick = function() {
      const img = this.querySelector("img");
      if (img) {
        document.getElementById("lightbox-img").src = img.src;
        document.getElementById("lightbox").classList.add("active");
      }
    };
    el.appendChild(div);

    // Load figure
    if (figCache[fig.filename]) {
      div.innerHTML = `<div class="fig-label">${fig.name}</div><img src="data:image/png;base64,${figCache[fig.filename]}" alt="${fig.name}">`;
    } else {
      fetch("/api/figure/" + fig.filename)
        .then(r => r.json())
        .then(d => {
          if (d.data) {
            figCache[fig.filename] = d.data;
            div.innerHTML = `<div class="fig-label">${fig.name}</div><img src="data:image/png;base64,${d.data}" alt="${fig.name}">`;
          }
        });
    }
  }
}

function renderResults(data) {
  const row = document.getElementById("results-row");
  if (!data.model_summary && !data.findings) {
    row.style.display = "none";
    return;
  }
  row.style.display = "";
  if (data.model_summary) {
    document.getElementById("model-content").textContent = data.model_summary;
  }
  if (data.findings) {
    document.getElementById("findings-content").innerHTML = md(data.findings);
  }
}

function updatePhase(phase) {
  const badge = document.getElementById("phase-badge");
  badge.className = "phase-badge phase-" + phase;
  badge.classList.toggle("pulse", phase !== "complete" && phase !== "waiting");
  const labels = {
    waiting: "waiting",
    collecting: "collecting",
    processing: "processing features",
    analyzing: "analyzing",
    complete: "complete",
  };
  badge.textContent = labels[phase] || phase;
}

async function poll() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    updatePhase(data.phase);
    renderOverview(data);
    renderCollection(data);
    renderBands(data);
    renderFigures(data);
    renderResults(data);
  } catch (e) {
    // server might be restarting, ignore
  }
}

// Poll every 3 seconds
poll();
setInterval(poll, 3000);
</script>
</body>
</html>"""
