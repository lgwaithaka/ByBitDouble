"""
dashboard_api.py — ByBitDouble Visual Dashboard
Bright beige theme, auto-refreshes every 30 seconds.
"""
import os, json, sqlite3, time
from datetime import datetime, timezone
from contextlib import contextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ByBitDouble Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = os.getenv("DB_PATH", "./data/compound.db")

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def safe_query(query, params=()):
    try:
        with get_db() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]
    except:
        return []

def safe_one(query, params=()):
    rows = safe_query(query, params)
    return rows[0] if rows else {}


@app.get("/api/status")
def api_status():
    params = {}
    for r in safe_query("SELECT key, value FROM params"):
        params[r["key"]] = r["value"]

    stats = safe_one("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses,
               SUM(CASE WHEN outcome != 'OPEN' THEN pnl_usdt ELSE 0 END) as total_pnl,
               SUM(CASE WHEN outcome='OPEN' THEN 1 ELSE 0 END) as open_positions
        FROM trades
    """)

    today_stats = safe_one("""
        SELECT COALESCE(SUM(pnl_usdt), 0) as today_pnl,
               COUNT(*) as today_trades
        FROM trades
        WHERE outcome != 'OPEN'
        AND datetime(closed_at) >= datetime('now', '-24 hours')
    """)

    return {
        "epoch": params.get("current_epoch", "1"),
        "epoch_start_bal": params.get("epoch_start_bal", "0"),
        "initial_capital": params.get("initial_capital", "0"),
        "total_trades": stats.get("total", 0),
        "wins": stats.get("wins", 0),
        "losses": stats.get("losses", 0),
        "total_pnl": round(stats.get("total_pnl", 0) or 0, 4),
        "open_positions": stats.get("open_positions", 0),
        "today_pnl": round(today_stats.get("today_pnl", 0) or 0, 4),
        "today_trades": today_stats.get("today_trades", 0),
    }


@app.get("/api/trades")
def api_trades(limit: int = 50):
    return safe_query("""
        SELECT symbol, side, entry_price, exit_price, pnl_usdt, outcome,
               opened_at, closed_at, mode, confidence
        FROM trades ORDER BY opened_at DESC LIMIT ?
    """, (limit,))


@app.get("/api/epochs")
def api_epochs():
    return safe_query("SELECT * FROM epochs ORDER BY epoch_num DESC LIMIT 20")


@app.get("/api/hourly")
def api_hourly():
    return safe_query("SELECT * FROM hourly_snaps ORDER BY ts DESC LIMIT 48")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ByBitDouble Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #F5F0E8;
    color: #2D2A26;
    min-height: 100vh;
    padding: 20px;
  }
  .header {
    background: linear-gradient(135deg, #E8DFD0, #F2EDE4);
    border: 1px solid #D4C9B8;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  .header h1 {
    font-size: 24px;
    color: #3D3528;
    font-weight: 700;
  }
  .header .subtitle {
    font-size: 13px;
    color: #8B7E6A;
    margin-top: 4px;
  }
  .header .live-dot {
    width: 10px; height: 10px;
    background: #4CAF50;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .status-badge {
    background: #4CAF50;
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }
  .card {
    background: #FFFBF3;
    border: 1px solid #E0D6C4;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.15s;
  }
  .card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
  .card .label {
    font-size: 12px;
    color: #8B7E6A;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 8px;
  }
  .card .value {
    font-size: 28px;
    font-weight: 700;
    color: #3D3528;
  }
  .card .sub {
    font-size: 13px;
    color: #8B7E6A;
    margin-top: 4px;
  }
  .green { color: #2E7D32 !important; }
  .red { color: #C62828 !important; }
  .amber { color: #E65100 !important; }

  .section {
    background: #FFFBF3;
    border: 1px solid #E0D6C4;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .section h2 {
    font-size: 16px;
    color: #3D3528;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #E8DFD0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th {
    text-align: left;
    padding: 10px 12px;
    background: #F0E9DC;
    color: #5C5344;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 2px solid #D4C9B8;
  }
  td {
    padding: 10px 12px;
    border-bottom: 1px solid #EDE7DB;
    color: #3D3528;
  }
  tr:hover td {
    background: #F8F3EB;
  }
  .pnl-pos { color: #2E7D32; font-weight: 600; }
  .pnl-neg { color: #C62828; font-weight: 600; }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-win { background: #E8F5E9; color: #2E7D32; }
  .badge-loss { background: #FFEBEE; color: #C62828; }
  .badge-open { background: #FFF3E0; color: #E65100; }

  .refresh-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #8B7E6A;
    margin-bottom: 16px;
  }
  .refresh-bar .countdown {
    background: #E8DFD0;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: #E8DFD0;
    border-radius: 4px;
    margin-top: 8px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #66BB6A);
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  @media (max-width: 768px) {
    body { padding: 12px; }
    .header { flex-direction: column; align-items: flex-start; gap: 12px; }
    .grid { grid-template-columns: repeat(2, 1fr); }
    .card .value { font-size: 22px; }
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1><span class="live-dot"></span> ByBitDouble Engine</h1>
    <div class="subtitle">Perpetual Compounding | Auto-Refresh 30s</div>
  </div>
  <span class="status-badge" id="status-badge">LOADING...</span>
</div>

<div class="refresh-bar">
  <span id="last-update">Updating...</span>
  <span class="countdown" id="countdown">30s</span>
</div>

<div class="grid" id="cards-grid">
  <div class="card"><div class="label">Balance</div><div class="value" id="balance">—</div><div class="sub" id="balance-sub"></div></div>
  <div class="card"><div class="label">Total PnL</div><div class="value" id="total-pnl">—</div><div class="sub" id="pnl-sub"></div></div>
  <div class="card"><div class="label">Today PnL</div><div class="value" id="today-pnl">—</div><div class="sub" id="today-sub"></div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value" id="win-rate">—</div><div class="sub" id="wr-sub"></div></div>
  <div class="card"><div class="label">Epoch</div><div class="value" id="epoch">—</div><div class="sub" id="epoch-sub"></div>
    <div class="progress-bar"><div class="progress-fill" id="epoch-progress"></div></div>
  </div>
  <div class="card"><div class="label">Open Positions</div><div class="value" id="open-pos">—</div><div class="sub" id="open-sub"></div></div>
</div>

<div class="section">
  <h2>Recent Trades</h2>
  <div style="overflow-x:auto;">
    <table id="trades-table">
      <thead>
        <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Outcome</th><th>Mode</th><th>Conf</th><th>Time</th></tr>
      </thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
</div>

<script>
let countdown = 30;
let refreshInterval;

async function fetchData() {
  try {
    const [statusRes, tradesRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/trades?limit=30')
    ]);
    const status = await statusRes.json();
    const trades = await tradesRes.json();
    updateCards(status);
    updateTrades(trades);
    document.getElementById('last-update').textContent =
      'Last update: ' + new Date().toLocaleTimeString();
    document.getElementById('status-badge').textContent = 'LIVE';
    document.getElementById('status-badge').style.background = '#4CAF50';
  } catch (e) {
    document.getElementById('status-badge').textContent = 'OFFLINE';
    document.getElementById('status-badge').style.background = '#C62828';
  }
  countdown = 30;
}

function updateCards(s) {
  const startBal = parseFloat(s.epoch_start_bal) || 1;
  const target = startBal * 3;  // 200% target
  const pnl = s.total_pnl || 0;
  const estBal = startBal + pnl;
  const wins = s.wins || 0;
  const losses = s.losses || 0;
  const total = wins + losses;
  const wr = total > 0 ? ((wins / total) * 100).toFixed(1) : '0';
  const gainPct = startBal > 0 ? ((pnl / startBal) * 100).toFixed(1) : '0';
  const progress = Math.min(100, Math.max(0, ((estBal - startBal) / (target - startBal)) * 100));

  const balEl = document.getElementById('balance');
  balEl.textContent = '$' + estBal.toFixed(2);

  const pnlEl = document.getElementById('total-pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
  pnlEl.className = 'value ' + (pnl >= 0 ? 'green' : 'red');
  document.getElementById('pnl-sub').textContent = gainPct + '% gain';

  const todayEl = document.getElementById('today-pnl');
  const tp = s.today_pnl || 0;
  todayEl.textContent = (tp >= 0 ? '+' : '') + '$' + tp.toFixed(2);
  todayEl.className = 'value ' + (tp >= 0 ? 'green' : 'red');
  document.getElementById('today-sub').textContent = s.today_trades + ' trades today';

  document.getElementById('win-rate').textContent = wr + '%';
  document.getElementById('wr-sub').textContent = wins + 'W / ' + losses + 'L (' + total + ' total)';

  document.getElementById('epoch').textContent = '#' + s.epoch;
  document.getElementById('epoch-sub').textContent =
    'Start $' + parseFloat(s.epoch_start_bal).toFixed(2) + ' → Target $' + target.toFixed(2);
  document.getElementById('epoch-progress').style.width = progress + '%';

  document.getElementById('open-pos').textContent = s.open_positions || 0;
  document.getElementById('balance-sub').textContent = 'Epoch start: $' + parseFloat(s.epoch_start_bal).toFixed(2);
}

function updateTrades(trades) {
  const tbody = document.getElementById('trades-body');
  tbody.innerHTML = '';
  trades.forEach(t => {
    const pnl = t.pnl_usdt || 0;
    const pnlClass = pnl > 0 ? 'pnl-pos' : (pnl < 0 ? 'pnl-neg' : '');
    const badgeClass = t.outcome === 'WIN' ? 'badge-win' :
                       t.outcome === 'LOSS' ? 'badge-loss' : 'badge-open';
    const time = t.opened_at ? new Date(t.opened_at).toLocaleString() : '';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${t.symbol || ''}</strong></td>
      <td>${t.side || ''}</td>
      <td>$${parseFloat(t.entry_price || 0).toFixed(4)}</td>
      <td>${t.exit_price ? '$' + parseFloat(t.exit_price).toFixed(4) : '—'}</td>
      <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(4)}</td>
      <td><span class="badge ${badgeClass}">${t.outcome || 'OPEN'}</span></td>
      <td>${t.mode || ''}</td>
      <td>${t.confidence || ''}%</td>
      <td style="font-size:11px;color:#8B7E6A">${time}</td>
    `;
    tbody.appendChild(row);
  });
}

// Auto-refresh
fetchData();
refreshInterval = setInterval(fetchData, 30000);
setInterval(() => {
  countdown = Math.max(0, countdown - 1);
  document.getElementById('countdown').textContent = countdown + 's';
}, 1000);
</script>
</body>
</html>"""
