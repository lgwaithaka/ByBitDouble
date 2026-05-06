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
body{font-family:'Segoe UI',-apple-system,sans-serif;background:#F5F0E8;color:#2D2A26;height:100vh;padding:12px;overflow:hidden;display:flex;flex-direction:column}
.header{background:linear-gradient(135deg,#E8DFD0,#F2EDE4);border:1px solid #D4C9B8;border-radius:12px;padding:12px 20px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
.header h1{font-size:18px;color:#3D3528;font-weight:700}
.header .subtitle{font-size:11px;color:#8B7E6A;margin-top:2px}
.live-dot{width:8px;height:8px;background:#4CAF50;border-radius:50%;display:inline-block;margin-right:5px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.status-badge{background:#4CAF50;color:white;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600}
.refresh-bar{display:flex;justify-content:space-between;font-size:10px;color:#8B7E6A;margin-bottom:8px;flex-shrink:0}
.countdown{background:#E8DFD0;padding:2px 8px;border-radius:8px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;flex-shrink:0}
.card{background:#FFFBF3;border:1px solid #E0D6C4;border-radius:10px;padding:10px 14px}
.card .label{font-size:9px;color:#8B7E6A;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:3px}
.card .value{font-size:20px;font-weight:700;color:#3D3528}
.card .sub{font-size:10px;color:#8B7E6A;margin-top:2px}
.green{color:#2E7D32!important}.red{color:#C62828!important}
.progress-bar{width:100%;height:6px;background:#E8DFD0;border-radius:3px;margin-top:4px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#4CAF50,#66BB6A);border-radius:3px;transition:width 0.5s}
.mid-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;flex-shrink:0}
.chart-box{background:#FFFBF3;border:1px solid #E0D6C4;border-radius:10px;padding:10px 14px;height:160px;position:relative}
.chart-box h3{font-size:11px;color:#3D3528;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px}
.chart-box .live-tag{position:absolute;top:10px;right:14px;font-size:9px;color:#4CAF50;font-weight:700}
.chart-box canvas{width:100%!important;height:110px!important}
.bottom-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;flex:1;min-height:0}
.section{background:#FFFBF3;border:1px solid #E0D6C4;border-radius:10px;padding:10px;overflow:hidden;display:flex;flex-direction:column}
.section h2{font-size:11px;color:#3D3528;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #E8DFD0;text-transform:uppercase;letter-spacing:0.5px;flex-shrink:0}
.section .scroll{overflow-y:auto;flex:1;min-height:0}
table{width:100%;border-collapse:collapse;font-size:10px}
th{text-align:left;padding:4px 6px;background:#F0E9DC;color:#5C5344;font-weight:600;font-size:9px;text-transform:uppercase;border-bottom:1px solid #D4C9B8;position:sticky;top:0;z-index:1}
td{padding:4px 6px;border-bottom:1px solid #EDE7DB}
tr:hover td{background:#F8F3EB}
.pnl-pos{color:#2E7D32;font-weight:600}.pnl-neg{color:#C62828;font-weight:600}
.badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:9px;font-weight:600}
.badge-win{background:#E8F5E9;color:#2E7D32}.badge-loss{background:#FFEBEE;color:#C62828}.badge-open{background:#FFF3E0;color:#E65100}
@media(max-width:768px){.grid{grid-template-columns:repeat(2,1fr)}.mid-row{grid-template-columns:1fr}.bottom-row{grid-template-columns:1fr}.card .value{font-size:16px}body{overflow:auto;height:auto}}
</style>
</head>
<body>
<div class="header"><div><h1><span class="live-dot"></span> ByBitDouble Engine</h1><div class="subtitle">Daily +20% Target | Auto-Refresh 30s</div></div><span class="status-badge" id="status-badge">LOADING</span></div>
<div class="refresh-bar"><span id="last-update">Updating...</span><span class="countdown" id="countdown">30s</span></div>
<div class="grid">
<div class="card"><div class="label">Balance</div><div class="value" id="balance">-</div><div class="sub" id="bal-sub"></div></div>
<div class="card"><div class="label">Total PnL</div><div class="value" id="total-pnl">-</div><div class="sub" id="pnl-sub"></div></div>
<div class="card"><div class="label">Win Rate</div><div class="value" id="win-rate">-</div><div class="sub" id="wr-sub"></div></div>
<div class="card"><div class="label">Daily Target</div><div class="value" id="epoch">-</div><div class="sub" id="ep-sub"></div><div class="progress-bar"><div class="progress-fill" id="ep-prog"></div></div></div>
</div>
<div class="mid-row">
<div class="chart-box"><h3>Balance vs Target</h3><span class="live-tag">&#9679; LIVE</span><canvas id="balChart"></canvas></div>
<div class="chart-box"><h3>PnL Curve</h3><span class="live-tag">&#9679; LIVE</span><canvas id="pnlChart"></canvas></div>
</div>
<div class="bottom-row">
<div class="section"><h2>Recent Trades</h2><div class="scroll"><table><thead><tr><th>Symbol</th><th>PnL</th><th>Result</th><th>Mode</th></tr></thead><tbody id="trades-body"></tbody></table></div></div>
<div class="section"><h2>Symbol Stats</h2><div class="scroll"><table><thead><tr><th>Symbol</th><th>W</th><th>L</th><th>PnL</th></tr></thead><tbody id="symbols-body"></tbody></table></div></div>
<div class="section"><h2>Whale Intel</h2><div class="scroll" id="whale-info" style="font-size:10px">Loading...</div></div>
</div>
<script>
let cd=30;let balData=[];let pnlData=[];
async function fetchAll(){try{const[ov,tr,sym,wh,pos,cap]=await Promise.all([fetch('/api/overview').then(r=>r.json()),fetch('/api/trades?hours=168').then(r=>r.json()),fetch('/api/symbols').then(r=>r.json()),fetch('/api/whale').then(r=>r.json()),fetch('/api/positions').then(r=>r.json()),fetch('/api/capital-curve').then(r=>r.json())]);renderCards(ov,pos);renderTrades(tr.trades||[]);renderSymbols(sym.symbols||[]);renderWhale(wh);renderBalChart(cap.log||[],ov);renderPnlChart(tr.trades||[],ov);document.getElementById('status-badge').textContent='LIVE';document.getElementById('status-badge').style.background='#4CAF50'}catch(e){document.getElementById('status-badge').textContent='OFFLINE';document.getElementById('status-badge').style.background='#C62828'}document.getElementById('last-update').textContent='Updated: '+new Date().toLocaleTimeString();cd=30}
function renderCards(ov,pos){const s=ov.stats||{};const c=ov.compound||{};const est=ov.est_balance||0;const pnl=s.total_pnl||0;const wins=s.total_wins||0;const losses=s.total_losses||0;const total=wins+losses;const wr=total>0?(wins/total*100).toFixed(1):'0';const gainPct=ov.initial>0?(pnl/ov.initial*100).toFixed(1):'0';document.getElementById('balance').textContent='$'+est.toFixed(2);document.getElementById('bal-sub').textContent='Init $'+parseFloat(ov.initial||0).toFixed(2)+' | Today '+(s.today_pnl>=0?'+':'')+parseFloat(s.today_pnl||0).toFixed(2);const p=document.getElementById('total-pnl');p.textContent=(pnl>=0?'+$':'-$')+Math.abs(pnl).toFixed(2);p.className='value '+(pnl>=0?'green':'red');document.getElementById('pnl-sub').textContent=gainPct+'% gain | PF '+(Math.abs(s.avg_loss||0.01)>0?(Math.abs(s.avg_win||0)/Math.abs(s.avg_loss||0.01)).toFixed(1)+'x':'-');document.getElementById('win-rate').textContent=wr+'%';document.getElementById('wr-sub').textContent=wins+'W/'+losses+'L | '+(pos.count||0)+' open';const ds=parseFloat(ov.daily_start||c.epoch_start_bal||ov.initial||1);const dt=ds*1.2;const dProg=Math.min(100,Math.max(0,((est-ds)/(dt-ds))*100));document.getElementById('epoch').textContent=dProg.toFixed(0)+'%';document.getElementById('ep-sub').textContent='$'+ds.toFixed(2)+' > $'+dt.toFixed(2);document.getElementById('ep-prog').style.width=Math.max(0,dProg)+'%'}
function drawLine(canvasId,points,color,fillColor,yMin,yMax,targetPts){const c=document.getElementById(canvasId);if(!c||!points.length)return;const ctx=c.getContext('2d');const dpr=window.devicePixelRatio||1;const rect=c.parentElement.getBoundingClientRect();c.width=(rect.width-28)*dpr;c.height=110*dpr;c.style.width=(rect.width-28)+'px';c.style.height='110px';ctx.scale(dpr,dpr);const w=rect.width-28;const h=110;const pad=4;const pw=w-pad*2;const ph=h-pad*2;if(!yMin&&yMin!==0)yMin=Math.min(...points);if(!yMax&&yMax!==0)yMax=Math.max(...points);if(yMax===yMin){yMax+=1;yMin-=1}ctx.clearRect(0,0,w,h);ctx.strokeStyle='#E0D6C4';ctx.lineWidth=0.5;for(let i=0;i<3;i++){const y=pad+ph*(i/2);ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke();ctx.fillStyle='#8B7E6A';ctx.font='8px sans-serif';ctx.fillText((yMax-(yMax-yMin)*(i/2)).toFixed(1),pad,y-2)}if(targetPts&&targetPts.length){ctx.strokeStyle='#2E75B6';ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();for(let i=0;i<targetPts.length;i++){const x=pad+(i/(targetPts.length-1))*pw;const y=pad+ph*(1-(targetPts[i]-yMin)/(yMax-yMin));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.stroke();ctx.setLineDash([])}ctx.beginPath();for(let i=0;i<points.length;i++){const x=pad+(i/(points.length-1))*pw;const y=pad+ph*(1-(points[i]-yMin)/(yMax-yMin));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();if(fillColor){ctx.lineTo(pad+(points.length-1)/(points.length-1)*pw,pad+ph);ctx.lineTo(pad,pad+ph);ctx.closePath();ctx.fillStyle=fillColor;ctx.fill()}}
function renderBalChart(log,ov){if(!log.length)return;const bals=log.map(l=>l.balance||0);const targets=log.map(l=>l.target||0);const all=bals.concat(targets);const mn=Math.min(...all)*0.95;const mx=Math.max(...all)*1.05;drawLine('balChart',bals,'#2E7D32','rgba(46,125,50,0.1)',mn,mx,targets)}
function renderPnlChart(trades,ov){if(!trades.length)return;const init=ov.initial||0;let cum=init;const pts=[init];const sorted=trades.slice().sort((a,b)=>(a.ts_open||0)-(b.ts_open||0));sorted.forEach(t=>{if(t.outcome!=='OPEN'){cum+=(t.pnl_usdt||0);pts.push(cum)}});drawLine('pnlChart',pts,'#1565C0','rgba(21,101,192,0.08)')}
function renderTrades(trades){const tb=document.getElementById('trades-body');tb.innerHTML='';trades.slice(0,20).forEach(t=>{const pnl=t.pnl_usdt||0;const cls=pnl>0?'pnl-pos':(pnl<0?'pnl-neg':'');const badge=t.outcome==='WIN'?'badge-win':(t.outcome==='LOSS'?'badge-loss':'badge-open');const row=document.createElement('tr');row.innerHTML='<td><b>'+t.symbol+'</b></td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(3)+'</td><td><span class="badge '+badge+'">'+t.outcome+'</span></td><td>'+t.mode+'</td>';tb.appendChild(row)})}
function renderSymbols(syms){const tb=document.getElementById('symbols-body');tb.innerHTML='';syms.sort((a,b)=>(b.total_pnl||0)-(a.total_pnl||0));syms.forEach(s=>{const pnl=s.total_pnl||0;const cls=pnl>0?'pnl-pos':(pnl<0?'pnl-neg':'');const row=document.createElement('tr');row.innerHTML='<td><b>'+s.symbol+'</b></td><td class="green">'+(s.wins||0)+'</td><td class="red">'+(s.losses||0)+'</td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(2)+'</td>';tb.appendChild(row)})}
function renderWhale(wh){const el=document.getElementById('whale-info');if(!wh||!wh.summary){el.textContent='No whale data';return}const s=wh.summary;const age=wh.age_minutes?wh.age_minutes.toFixed(0)+'m ago':'?';let h='<div><b>Bias:</b> '+(s.market_bias||'-')+' | <b>Tracked:</b> '+(s.symbols_tracked||0)+' | <b>Updated:</b> '+age+'</div>';if(wh.opportunities&&wh.opportunities.length){h+='<table style="margin-top:4px"><thead><tr><th>Sym</th><th>Score</th><th>Bias</th></tr></thead><tbody>';wh.opportunities.slice(0,4).forEach(o=>{h+='<tr><td><b>'+o.symbol+'</b></td><td>'+((o.score||0).toFixed(2))+'</td><td>'+(o.bias||'-')+'</td></tr>'});h+='</tbody></table>'}el.innerHTML=h}
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
        "daily_start":      float(gp("daily_start_bal", str(initial))),
        "daily_target":     float(gp("daily_target", str(initial * 1.2))),
        "daily_target_pct": 20,
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
