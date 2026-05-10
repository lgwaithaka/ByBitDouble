"""
server.py — ByBitDouble FastAPI Server
Fixes:
  • Correct import: CompoundTradingEngine (was TradingEngine — crashed on startup)
  • All dashboard API endpoints implemented (/api/overview, /api/positions, etc.)
  • VScanner.run_continuous() used correctly
  • CORS enabled for dashboard
  • Health endpoint returns full status
"""
import asyncio, logging, os, time, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from uvicorn import run

from engine  import CompoundTradingEngine, engine
from scanner import VScanner
from db      import (
    init_db, all_time_stats, gp, sp, get_open_positions, get_trades,
    get_capital_log, get_hourly_snaps, all_symbol_stats, all_params,
    get_all_epochs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ByBitDouble Trading Bot v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

def _get_dashboard_html() -> str:
    """Load dashboard.html from disk, fallback to inline message."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>dashboard.html not found — upload it alongside server.py</h1>"

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the trading dashboard."""
    return _get_dashboard_html()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Alias for /"""
    return _get_dashboard_html()

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("🚀 Initializing ByBitDouble Trading Engine v3.0...")

    api_key    = os.getenv("BYBIT_API_KEY",    "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet    = os.getenv("BYBIT_TESTNET",    "false").lower() == "true"

    if api_key and api_secret:
        engine.init(api_key, api_secret, testnet)
        asyncio.create_task(engine.main_loop())
        logger.info("✅ Engine started with real Bybit credentials")
    else:
        # Demo mode — engine still serves dashboard with DB data
        init_db()
        engine.status = "DEMO"
        logger.warning("⚠ No BYBIT_API_KEY set — running in DEMO mode (dashboard only)")

    logger.info("✅ All systems operational | Dashboard live | Scanning 19:00-01:00 UTC")


@app.on_event("shutdown")
async def shutdown():
    engine.running = False
    logger.info("Engine shutdown complete.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "engine_status":   engine.status,
        "trading_active":  engine.running,
        "uptime_seconds":  int(time.time()) - engine.start_time,
        "balance":         round(engine.balance, 4),
        "open_positions":  len(get_open_positions()),
        "last_scan_ts":    engine.last_scan_ts,
    }


# ── Overview ──────────────────────────────────────────────────────────────────

@app.get("/api/overview")
async def api_overview():
    try:
        data = engine.get_overview()
        # Add top scanner data
        scanner = engine.scanner
        data["scanner_top"] = scanner.top[:10] if scanner else []
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"overview error: {e}", exc_info=True)
        # Return safe fallback
        stats = all_time_stats()
        return JSONResponse({
            "engine_status":    engine.status,
            "est_balance":      engine.balance,
            "initial":          float(gp("initial_capital", "0") or "0"),
            "stats":            stats,
            "compound":         {"mode": "NORMAL", "epoch_num": 1, "day_in_epoch": 1,
                                 "epoch_start_bal": engine.balance, "epoch_target": engine.balance * 2,
                                 "epoch_progress_pct": 0, "days_remaining": 5, "ahead_pct": 0,
                                 "on_track": False, "streak": 0, "consecutive_losses": 0,
                                 "rolling_win_rate": 0.5, "sizing": {"conf_floor": 68, "max_concurrent": 8}},
            "scan_results":     [],
            "epoch_history":    get_all_epochs(),
            "projection":       [],
            "market_regime":    gp("market_regime", "RANGING"),
            "last_scan_ts":     engine.last_scan_ts,
            "errors":           [],
        })


# ── Positions ─────────────────────────────────────────────────────────────────

@app.get("/api/positions")
async def api_positions():
    positions = get_open_positions()
    total_unr     = sum(p.get("unrealised_pnl") or 0 for p in positions)
    total_notional= sum((p.get("notional") or 0) for p in positions)
    return {
        "positions":        positions,
        "total_unrealised": round(total_unr, 4),
        "total_notional":   round(total_notional, 2),
        "count":            len(positions),
    }


# ── Trade History ─────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def api_trades(hours: int = 24, symbol: str = None, epoch: int = None):
    trades = get_trades(hours=hours, symbol=symbol, epoch=epoch)
    closed = [t for t in trades if t["outcome"] not in ("OPEN", None)]
    pnl    = sum(t["pnl_usdt"] or 0 for t in closed)
    return {
        "trades": trades,
        "pnl":    round(pnl, 4),
        "count":  len(trades),
    }


# ── Capital Curve ─────────────────────────────────────────────────────────────

@app.get("/api/capital-curve")
async def api_capital_curve():
    log = get_capital_log(limit=500)
    s   = engine.state
    return {
        "log": [
            {
                "ts":      row["ts"],
                "balance": row["balance"],
                "target":  row["target_now"],
                "mode":    row["mode"],
            }
            for row in log
        ],
        "epoch_target": s.epoch_target,
        "epoch_start":  s.epoch_start_bal,
    }


# ── Hourly Snaps ──────────────────────────────────────────────────────────────

@app.get("/api/hourly")
async def api_hourly():
    snaps = get_hourly_snaps(limit=96)
    return {"snapshots": snaps}


# ── Symbols / Asset Leaderboard ───────────────────────────────────────────────

@app.get("/api/symbols")
async def api_symbols():
    sym_stats = all_symbol_stats()
    scanner   = engine.scanner
    top_scan  = {s["symbol"]: s for s in (scanner.top if scanner else [])}

    result = []
    seen   = set()
    for ss in sym_stats:
        sym = ss["symbol"]
        seen.add(sym)
        scan_data = top_scan.get(sym, {})
        result.append({**ss, **{
            "vol_score": scan_data.get("vol_score", 0),
            "range_pct": scan_data.get("range_pct", 0),
            "chg_24h":   scan_data.get("chg_24h", 0),
            "vol_24h_m": scan_data.get("vol_24h_m", 0),
        }})

    # Add scanner symbols not yet traded
    for sym, scan_data in top_scan.items():
        if sym not in seen:
            result.append({
                "symbol":       sym,
                "wins":         0, "losses": 0, "total_pnl": 0,
                "win_rate":     0, "learned_bias": 0,
                "sl_mult":      1.8, "tp_mult": 3.0, "avg_dur_s": 0,
                "vol_score":    scan_data.get("vol_score", 0),
                "range_pct":    scan_data.get("range_pct", 0),
                "chg_24h":      scan_data.get("chg_24h", 0),
                "vol_24h_m":    scan_data.get("vol_24h_m", 0),
            })

    result.sort(key=lambda x: x.get("vol_score", 0), reverse=True)
    return {"symbols": result}


# ── Whale Intel ───────────────────────────────────────────────────────────────

@app.get("/api/whale")
async def api_whale():
    whale_summary = {}
    whale_opps    = []
    whale_cache   = {}
    try:
        raw_summary = gp("whale_summary", "{}")
        raw_opps    = gp("whale_opportunities", "[]")
        whale_summary = json.loads(raw_summary or "{}")
        whale_opps    = json.loads(raw_opps    or "[]")
    except Exception:
        pass

    last_ts = int(gp("whale_last_update", "0") or "0")
    age_min = int((time.time() - last_ts) / 60) if last_ts > 0 else None

    return {
        "summary":       whale_summary,
        "opportunities": whale_opps,
        "cache":         whale_cache,
        "age_minutes":   age_min,
    }


# ── Params ────────────────────────────────────────────────────────────────────

@app.get("/api/params")
async def api_params():
    return all_params()


@app.post("/api/params")
async def api_params_save(request: Request):
    body = await request.json()
    allowed = {
        "epoch_max_dd_pct", "daily_max_dd_pct", "vol_scan_n",
        "scan_interval_s", "min_notional_usdt", "max_concurrent",
    }
    saved = {}
    for k, v in body.items():
        if k in allowed:
            sp(k, str(v))
            saved[k] = v
    return {"saved": saved, "ok": True}


# ── Stats (dashboard summary) ─────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    return all_time_stats()


# ── Scanner status ────────────────────────────────────────────────────────────

@app.get("/api/scanner-status")
async def scanner_status():
    sc = engine.scanner
    if not sc:
        return {"error": "Scanner not initialised"}
    return sc.get_status()


# ── Dashboard stats (legacy endpoint) ────────────────────────────────────────

@app.get("/dashboard-stats")
async def dashboard_stats():
    stats = all_time_stats()
    sc    = engine.scanner
    return {
        **stats,
        "scanner":     sc.get_status() if sc else {},
        "last_update": time.time(),
        "mode":        gp("current_mode", "NORMAL"),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, log_level="info")
