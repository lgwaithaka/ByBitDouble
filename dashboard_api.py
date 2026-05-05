import time
import json
"""
dashboard_api.py — Dashboard backend (port 8080)
Bright beige theme, auto-refresh 30s, 200% target
"""
import asyncio, json, time, os, logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from db import (
    init_db, all_time_stats, get_open_positions, get_trades,
    get_capital_log, all_symbol_stats, get_hourly_snaps,
    gp, sp, all_params, get_all_epochs
)
from compound_engine import CompoundEngine, EPOCH_DAYS, DAILY_REQUIRED_PCT

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Bybit Compound Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
init_db()

TARGET_MULT = float(os.getenv("TARGET_MULTIPLIER", "3.0"))

_ce = CompoundEngine()

def _rebuild_ce():
    initial  = float(gp("initial_capital", "100"))
    epoch_ts = int(gp("epoch_start_ts", str(int(time.time()))))
    epoch_bal= float(gp("epoch_start_bal", "100"))
    epoch_n  = int(gp("current_epoch", "1"))
    _ce.initialise(initial, epoch_n, epoch_ts, epoch_bal)
    _ce.state.epoch_target = epoch_bal * TARGET_MULT


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ByBitDouble Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif;background:#F5F0E8;color:#2D2A26;min-height:100vh;padding:20px}
.header{background:linear-gradient(135deg,#E8DFD0,#F2EDE4);border:1px solid #D4C9B8;border-radius:16px;padding:24px 32px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 12px rgba(0,0,0,0.06)}
.header h1{font-size:24px;color:#3D3528;font-weight:700}
.header .subtitle{font-size:13px;color:#8B7E6A;margin-top:4px}
.live-dot{width:10px;height:10px;background:#4CAF50;border-radius:50%;display:inline-block;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.status-badge{background:#4CAF50;color:white;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:600}
.refresh-bar{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#8B7E6A;margin-bottom:16px}
.countdown{background:#E8DFD0;padding:4px 12px;border-radius:12px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:20px}
.card{background:#FFFBF3;border:1px solid #E0D6C4;border-radius:14px;padding:18px 22px;box-shadow:0 2px 8px rgba(0,0,0,0.04);transition:transform 0.15s}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,0.08)}
.card .label{font-size:11px;color:#8B7E6A;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;margin-bottom:6px}
.card .value{font-size:26px;font-weight:700;color:#3D3528}
.card .sub{font-size:12px;color:#8B7E6A;margin-top:4px}
.green{color:#2E7D32!important}.red{color:#C62828!important}.amber{color:#E65100!important}
.section{background:#FFFBF3;border:1px solid #E0D6C4;border-radius:14px;padding:22px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.04)}
.section h2{font-size:15px;color:#3D3528;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #E8DFD0}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 10px;background:#F0E9DC;color:#5C5344;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #D4C9B8}
td{padding:8px 10px;border-bottom:1px solid #EDE7DB;color:#3D3528}
tr:hover td{background:#F8F3EB}
.pnl-pos{color:#2E7D32;font-weight:600}.pnl-neg{color:#C62828;font-weight:600}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.badge-win{background:#E8F5E9;color:#2E7D32}.badge-loss{background:#FFEBEE;color:#C62828}.badge-open{background:#FFF3E0;color:#E65100}
.progress-bar{width:100%;height:8px;background:#E8DFD0;border-radius:4px;margin-top:6px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#4CAF50,#66BB6A);border-radius:4px;transition:width 0.5s ease}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){body{padding:10px}.header{flex-direction:column;align-items:flex-start;gap:10px;padding:16px}.grid{grid-template-columns:repeat(2,1fr);gap:10px}.card .value{font-size:20px}.two-col{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header"><div><h1><span class="live-dot"></span> ByBitDouble Engine</h1><div class="subtitle">Perpetual Compounding | Target 200% | Auto-Refresh 30s</div></div><span class="status-badge" id="status-badge">LOADING</span></div>
<div class="refresh-bar"><span id="last-update">Updating...</span><span class="countdown" id="countdown">30s</span></div>
<div class="grid">
<div class="card"><div class="label">Balance</div><div class="value" id="balance">-</div><div class="sub" id="bal-sub"></div></div>
<div class="card"><div class="label">Total PnL</div><div class="value" id="total-pnl">-</div><div class="sub" id="pnl-sub"></div></div>
<div class="card"><div class="label">Today PnL</div><div class="value" id="today-pnl">-</div><div class="sub" id="today-sub"></div></div>
<div class="card"><div class="label">Win Rate</div><div class="value" id="win-rate">-</div><div class="sub" id="wr-sub"></div></div>
<div class="card"><div class="label">Epoch</div><div class="value" id="epoch">-</div><div class="sub" id="ep-sub"></div><div class="progress-bar"><div class="progress-fill" id="ep-prog"></div></div></div>
<div class="card"><div class="label">Open Positions</div><div class="value" id="open-pos">-</div><div class="sub" id="op-sub"></div></div>
<div class="card"><div class="label">Market Regime</div><div class="value" id="regime">-</div><div class="sub" id="reg-sub"></div></div>
<div class="card"><div class="label">Profit Factor</div><div class="value" id="pf">-</div><div class="sub" id="pf-sub"></div></div>
</div>
<div class="two-col">
<div class="section"><h2>Recent Trades</h2><div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table><thead><tr><th>Symbol</th><th>Side</th><th>PnL</th><th>Result</th><th>Mode</th><th>Time</th></tr></thead><tbody id="trades-body"></tbody></table></div></div>
<div class="section"><h2>Symbol Performance</h2><div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table><thead><tr><th>Symbol</th><th>Wins</th><th>Losses</th><th>PnL</th><th>Win Rate</th></tr></thead><tbody id="symbols-body"></tbody></table></div></div>
</div>
<div class="section"><h2>Whale Intelligence</h2><div id="whale-info" style="font-size:13px;color:#5C5344">Loading whale data...</div></div>
<div class="section"><h2>Epoch History</h2><div style="overflow-x:auto"><table><thead><tr><th>Epoch</th><th>Start</th><th>End</th><th>PnL</th><th>Win Rate</th></tr></thead><tbody id="epochs-body"></tbody></table></div></div>
<script>
let cd=30;const TM=3.0;
async function fetchAll(){try{const[ov,tr,sym,wh,pos]=await Promise.all([fetch('/api/overview').then(r=>r.json()),fetch('/api/trades?hours=72').then(r=>r.json()),fetch('/api/symbols').then(r=>r.json()),fetch('/api/whale').then(r=>r.json()),fetch('/api/positions').then(r=>r.json())]);renderCards(ov,pos);renderTrades(tr.trades||[]);renderSymbols(sym.symbols||[]);renderWhale(wh);renderEpochs(ov.epoch_history||[]);document.getElementById('status-badge').textContent='LIVE';document.getElementById('status-badge').style.background='#4CAF50'}catch(e){document.getElementById('status-badge').textContent='OFFLINE';document.getElementById('status-badge').style.background='#C62828'}document.getElementById('last-update').textContent='Updated: '+new Date().toLocaleTimeString();cd=30}
function renderCards(ov,pos){const s=ov.stats||{};const c=ov.compound||{};const est=ov.est_balance||0;const startBal=parseFloat(c.epoch_start_bal||ov.initial||1);const target=startBal*TM;const pnl=s.total_pnl||0;const wins=s.total_wins||0;const losses=s.total_losses||0;const total=wins+losses;const wr=total>0?(wins/total*100).toFixed(1):'0';const gainPct=ov.initial>0?(pnl/ov.initial*100).toFixed(1):'0';const progress=Math.min(100,Math.max(0,((est-startBal)/(target-startBal))*100));const avgW=s.avg_win||0;const avgL=Math.abs(s.avg_loss||0.01);const pf=avgL>0?(avgW/avgL).toFixed(2):'-';document.getElementById('balance').textContent='$'+est.toFixed(2);document.getElementById('bal-sub').textContent='Epoch start: $'+startBal.toFixed(2);const p=document.getElementById('total-pnl');p.textContent=(pnl>=0?'+':'')+pnl.toFixed(2);p.className='value '+(pnl>=0?'green':'red');document.getElementById('pnl-sub').textContent=gainPct+'% total gain';const tp=s.today_pnl||0;const tpEl=document.getElementById('today-pnl');tpEl.textContent=(tp>=0?'+':'')+tp.toFixed(2);tpEl.className='value '+(tp>=0?'green':'red');document.getElementById('today-sub').textContent=(s.today_trades||0)+' trades today';document.getElementById('win-rate').textContent=wr+'%';document.getElementById('wr-sub').textContent=wins+'W / '+losses+'L ('+total+')';document.getElementById('epoch').textContent='#'+(c.epoch_num||'?');document.getElementById('ep-sub').textContent='$'+startBal.toFixed(2)+' > $'+target.toFixed(2)+' (200%)';document.getElementById('ep-prog').style.width=Math.max(0,progress)+'%';document.getElementById('open-pos').textContent=pos.count||s.open_positions||0;document.getElementById('op-sub').textContent='Unrealised: $'+(pos.total_unrealised||0).toFixed(2);document.getElementById('regime').textContent=ov.market_regime||'-';document.getElementById('reg-sub').textContent=ov.regime_label||'';document.getElementById('pf').textContent=pf+'x';document.getElementById('pf-sub').textContent='Avg W $'+avgW.toFixed(2)+' / L $'+avgL.toFixed(2)}
function renderTrades(trades){const tb=document.getElementById('trades-body');tb.innerHTML='';trades.slice(0,40).forEach(t=>{const pnl=t.pnl_usdt||0;const cls=pnl>0?'pnl-pos':(pnl<0?'pnl-neg':'');const badge=t.outcome==='WIN'?'badge-win':(t.outcome==='LOSS'?'badge-loss':'badge-open');const time=t.closed_at?new Date(t.closed_at).toLocaleString():(t.opened_at?new Date(t.opened_at).toLocaleString():'');const row=document.createElement('tr');row.innerHTML='<td><b>'+t.symbol+'</b></td><td>'+t.side+'</td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(4)+'</td><td><span class="badge '+badge+'">'+t.outcome+'</span></td><td>'+t.mode+'</td><td style="font-size:10px;color:#8B7E6A">'+time+'</td>';tb.appendChild(row)})}
function renderSymbols(syms){const tb=document.getElementById('symbols-body');tb.innerHTML='';syms.sort((a,b)=>(b.total_pnl||0)-(a.total_pnl||0));syms.forEach(s=>{const pnl=s.total_pnl||0;const cls=pnl>0?'pnl-pos':(pnl<0?'pnl-neg':'');const total=(s.wins||0)+(s.losses||0);const wr=total>0?((s.wins/total)*100).toFixed(0)+'%':'-';const row=document.createElement('tr');row.innerHTML='<td><b>'+s.symbol+'</b></td><td class="green">'+(s.wins||0)+'</td><td class="red">'+(s.losses||0)+'</td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(2)+'</td><td>'+wr+'</td>';tb.appendChild(row)})}
function renderWhale(wh){const el=document.getElementById('whale-info');if(!wh||!wh.summary){el.textContent='No whale data yet';return}const s=wh.summary;const age=wh.age_minutes?wh.age_minutes.toFixed(0)+'m ago':'unknown';let html='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px"><div><b>Market Bias:</b> '+(s.market_bias||'-')+'</div><div><b>Tracked:</b> '+(s.symbols_tracked||0)+' symbols</div><div><b>Long Bias:</b> '+(s.long_bias_count||0)+'</div><div><b>Short Bias:</b> '+(s.short_bias_count||0)+'</div><div><b>Updated:</b> '+age+'</div></div>';if(wh.opportunities&&wh.opportunities.length>0){html+='<div style="margin-top:10px"><b>Top Opportunities:</b></div><table style="margin-top:6px"><thead><tr><th>Symbol</th><th>Score</th><th>Bias</th><th>Funding</th></tr></thead><tbody>';wh.opportunities.slice(0,5).forEach(o=>{html+='<tr><td><b>'+o.symbol+'</b></td><td>'+((o.score||0).toFixed(2))+'</td><td>'+(o.bias||'-')+'</td><td>'+(o.funding_rate?(o.funding_rate*100).toFixed(3)+'%':'-')+'</td></tr>'});html+='</tbody></table>'}el.innerHTML=html}
function renderEpochs(epochs){const tb=document.getElementById('epochs-body');tb.innerHTML='';epochs.forEach(e=>{const pnl=(e.end_bal||0)-(e.start_bal||0);const cls=pnl>0?'pnl-pos':(pnl<0?'pnl-neg':'');const total=(e.wins||0)+(e.losses||0);const wr=total>0?((e.wins/total)*100).toFixed(0)+'%':'-';const row=document.createElement('tr');row.innerHTML='<td><b>#'+e.epoch_num+'</b></td><td>$'+(e.start_bal||0).toFixed(2)+'</td><td>$'+(e.end_bal||0).toFixed(2)+'</td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(2)+'</td><td>'+wr+'</td>';tb.appendChild(row)})}
fetchAll();setInterval(fetchAll,30000);setInterval(()=>{cd=Math.max(0,cd-1);document.getElementById('countdown').textContent=cd+'s'},1000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/overview")
async def overview():
    loop = asyncio.get_event_loop()
    _rebuild_ce()
    stats  = await loop.run_in_executor(None, all_time_stats)
    epochs = await loop.run_in_executor(None, get_all_epochs)
    initial= float(gp("initial_capital", "100"))
    est    = initial + stats["total_pnl"]
    ce     = _ce
    return JSONResponse({
        "stats":            stats,
        "compound":         ce.get_status(est),
        "est_balance":      round(est, 4),
        "initial":          initial,
        "epoch_history":    epochs,
        "projection":       ce.project_compounding(ce.state.epoch_start_bal, 10),
        "market_regime":    gp("market_regime",    "RANGING"),
        "regime_label":     gp("regime_label",     ""),
        "hourly_win_rate":  gp("hourly_win_rate",  "0"),
        "hourly_pnl":       gp("hourly_pnl",       "0"),
        "last_strategy_ts": gp("last_strategy_ts", "0"),
        "target_multiplier": TARGET_MULT,
    })


@app.get("/api/positions")
async def positions():
    loop = asyncio.get_event_loop()
    pos  = await loop.run_in_executor(None, get_open_positions)
    unr  = sum(p.get("unrealised_pnl", 0) or 0 for p in pos)
    not_ = sum(p.get("notional", 0) or 0 for p in pos)
    return JSONResponse({"positions": pos, "count": len(pos),
                         "total_unrealised": round(unr, 4),
                         "total_notional": round(not_, 2)})


@app.get("/api/trades")
async def trades(hours: int = 24, symbol: str = None, epoch: int = None):
    loop = asyncio.get_event_loop()
    ts   = await loop.run_in_executor(None, lambda: get_trades(hours=hours, symbol=symbol, epoch=epoch))
    cl   = [t for t in ts if t["outcome"] != "OPEN"]
    return JSONResponse({
        "count": len(cl),
        "pnl":   round(sum(t["pnl_usdt"] or 0 for t in cl), 4),
        "trades": cl,
    })


@app.get("/api/capital-curve")
async def capital_curve():
    loop  = asyncio.get_event_loop()
    _rebuild_ce()
    ce    = _ce
    log   = await loop.run_in_executor(None, lambda: get_capital_log(300))
    for pt in log:
        elapsed = (pt["ts"] - ce.state.epoch_start_ts) / (EPOCH_DAYS * 86400)
        elapsed = max(0, elapsed)
        pt["target"] = round(ce.state.epoch_start_bal * (TARGET_MULT ** elapsed), 4)
    return JSONResponse({"log": log})


@app.get("/api/pnl-curve")
async def pnl_curve():
    loop   = asyncio.get_event_loop()
    trades = await loop.run_in_executor(None, lambda: get_trades(hours=720))
    cl     = sorted([t for t in trades if t["outcome"] != "OPEN"], key=lambda x: x["ts_open"])
    initial= float(gp("initial_capital", "100"))
    cum    = initial; pts = []
    for t in cl:
        pnl = t.get("pnl_usdt") or 0
        cum += pnl
        pts.append({
            "ts": t["ts_open"],
            "dt": datetime.fromtimestamp(t["ts_open"], tz=timezone.utc).isoformat(),
            "pnl": round(pnl, 4), "balance": round(cum, 4),
            "symbol": t["symbol"], "outcome": t["outcome"],
            "epoch": t["epoch"],
        })
    return JSONResponse({"initial": initial, "points": pts})


@app.get("/api/hourly")
async def hourly():
    loop = asyncio.get_event_loop()
    s    = await loop.run_in_executor(None, lambda: get_hourly_snaps(96))
    return JSONResponse({"snapshots": s})


@app.get("/api/symbols")
async def symbols():
    loop = asyncio.get_event_loop()
    s    = await loop.run_in_executor(None, all_symbol_stats)
    return JSONResponse({"symbols": s})


@app.get("/api/epochs")
async def epochs():
    loop = asyncio.get_event_loop()
    e    = await loop.run_in_executor(None, get_all_epochs)
    return JSONResponse({"epochs": e})


@app.get("/api/params")
async def get_params():
    return JSONResponse(all_params())


@app.post("/api/params")
async def set_params(body: dict):
    valid = ["epoch_max_dd_pct","daily_max_dd_pct","vol_scan_n","scan_interval_s"]
    upd   = {}
    for k, v in body.items():
        if k in valid:
            sp(k, str(v)); upd[k] = v
    return JSONResponse({"updated": upd})


@app.get("/api/whale")
async def whale_data():
    summary_str = gp("whale_summary", "{}")
    opps_str    = gp("whale_opportunities", "[]")
    last_upd    = int(gp("whale_last_update", "0") or "0")
    try:
        summary = json.loads(summary_str or "{}")
        opps    = json.loads(opps_str or "[]")
    except Exception:
        summary = {}; opps = []
    return JSONResponse({
        "summary":      summary,
        "opportunities":opps,
        "last_update":  last_upd,
        "age_minutes":  round((time.time()-last_upd)/60,1) if last_upd else None,
    })


@app.get("/api/market-feed")
async def market_feed():
    fg_val = gp("fear_greed_value", "50")
    fg_label = gp("fear_greed_label", "Neutral")
    fg_updated = gp("fear_greed_updated", "")
    return JSONResponse({
        "fear_greed_value": int(fg_val or 50),
        "fear_greed_label": fg_label,
        "last_updated": fg_updated,
        "target_multiplier": TARGET_MULT,
        "target_pct": f"{int((TARGET_MULT-1)*100)}%",
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "ts": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
