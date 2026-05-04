import time
import json
"""
dashboard_api.py — Dashboard backend (port 8080)
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

# Reconstruct compound engine state for read-only dashboard
_ce = CompoundEngine()

def _rebuild_ce():
    initial  = float(gp("initial_capital", "100"))
    epoch_ts = int(gp("epoch_start_ts", str(int(time.time()))))
    epoch_bal= float(gp("epoch_start_bal", "100"))
    epoch_n  = int(gp("current_epoch", "1"))
    _ce.initialise(initial, epoch_n, epoch_ts, epoch_bal)


@app.get("/", response_class=HTMLResponse)
async def root():
    f = STATIC / "dashboard.html"
    return HTMLResponse(f.read_text()) if f.exists() else HTMLResponse("<h1>Not found</h1>")


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
    # Build target curve
    for pt in log:
        elapsed = (pt["ts"] - ce.state.epoch_start_ts) / (EPOCH_DAYS * 86400)
        elapsed = max(0, elapsed)
        pt["target"] = round(ce.state.epoch_start_bal * (2.0 ** elapsed), 4)
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
    """Whale intelligence data for dashboard."""
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "ts": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
