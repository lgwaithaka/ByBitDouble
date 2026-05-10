"""
server.py — ByBitDouble MCP Trading Server
═══════════════════════════════════════════
  • Connects to REAL Bybit via API keys (not MockClient)
  • Serves MCP tools for Claude Desktop interaction
  • Dashboard API endpoints
  • Static file serving for dashboard.html
  • Health check for Render.com
"""
import asyncio, json, logging, os, sys, time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from uvicorn import run as uvicorn_run

from engine import engine, CompoundTradingEngine
from db import init_db, all_time_stats, gp, sp, all_params, get_trades
from db import get_open_positions, get_all_epochs, get_hourly_snaps
from db import get_capital_log, all_symbol_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ByBitDouble Trading Engine")

# ── Static files ─────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    api_key    = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet    = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

    if not api_key or not api_secret:
        logger.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set!")
        logger.info("Engine running in MONITOR mode (no trades)")
    else:
        engine.init(api_key, api_secret, testnet)
        asyncio.create_task(engine.main_loop())
        logger.info("✅ Engine started — trading active")

@app.on_event("shutdown")
async def shutdown():
    if engine.running:
        await engine.stop()
    logger.info("Engine shutdown complete.")

# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    dash_path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(dash_path):
        return FileResponse(dash_path)
    return JSONResponse({"error": "Dashboard not found. Place dashboard.html in static/"})

# Mount static directory for CSS/JS/images
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Health Check (Render) ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "trading_active": engine.running,
        "balance":        round(engine.balance, 4),
        "uptime_seconds": int(time.time()) - engine.start_time,
        "mode":           engine.compound.state.mode,
        "epoch":          engine.compound.state.epoch_num,
    }

# ── Dashboard API ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """Full engine status for dashboard."""
    ce    = engine.compound
    s     = ce.state
    stats = all_time_stats()
    return {
        "running":        engine.running,
        "status":         engine.status,
        "balance":        round(engine.balance, 4),
        "mode":           s.mode,
        "epoch":          s.epoch_num,
        "day_in_epoch":   s.day_in_epoch,
        "epoch_start":    round(s.epoch_start_bal, 4),
        "epoch_target":   round(s.epoch_target, 4),
        "target_now":     round(ce.target_at_now(), 4),
        "target_eod":     round(ce.target_at_day_end(), 4),
        "ahead_pct":      round(ce.ahead_pct(engine.balance), 2),
        "consecutive_w":  s.consecutive_wins,
        "consecutive_l":  s.consecutive_loss,
        "circuit_breaker": s.circuit_breaker,
        "cb_reason":      s.cb_reason,
        "market_regime":  engine.regime.regime,
        "regime_label":   engine.regime.adjustments()["label"],
        "uptime":         int(time.time()) - engine.start_time,
        "last_scan_ts":   engine.last_scan_ts,
        "errors":         engine.errors[-10:],
        "trading_window": engine._in_trading_window()[1],
        **stats,
    }

@app.get("/api/scans")
async def api_scans():
    return {"scans": engine.scan_results, "ts": engine.last_scan_ts}

@app.get("/api/positions")
async def api_positions():
    return {"positions": get_open_positions()}

@app.get("/api/trades")
async def api_trades(hours: int = 72):
    return {"trades": get_trades(hours=hours)}

@app.get("/api/epochs")
async def api_epochs():
    return {"epochs": get_all_epochs()}

@app.get("/api/symbols")
async def api_symbols():
    return {"symbols": all_symbol_stats()}

@app.get("/api/hourly")
async def api_hourly():
    return {"hourly": get_hourly_snaps(96)}

@app.get("/api/capital")
async def api_capital():
    return {"capital": get_capital_log(300)}

@app.get("/api/params")
async def api_params():
    return {"params": all_params()}

@app.get("/api/whale")
async def api_whale():
    return {
        "summary":       engine.whales.last_summary,
        "opportunities": engine.whales.get_top_opportunities(10),
        "cache":         {
            sym: {
                "score":   intel.composite,
                "bias":    intel.bias,
                "oi":      intel.oi_spike,
                "funding": intel.funding_rate,
                "ob":      intel.ob_wall,
            }
            for sym, intel in engine.whales._cache.items()
        },
    }

# ── MCP Tool Endpoints (for Claude Desktop) ─────────────────────────────────

@app.get("/api/mcp/status")
async def mcp_status():
    """Full engine status — equivalent to get_status MCP tool."""
    return await api_status()

@app.post("/api/mcp/start")
async def mcp_start():
    """Start the trading engine."""
    if engine.running:
        return {"result": "Already running"}
    api_key    = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    if not api_key:
        return {"error": "No API keys configured"}
    engine.init(api_key, api_secret, os.getenv("BYBIT_TESTNET", "false").lower() == "true")
    asyncio.create_task(engine.main_loop())
    return {"result": "Engine started"}

@app.post("/api/mcp/stop")
async def mcp_stop():
    """Stop the trading engine gracefully."""
    await engine.stop()
    return {"result": "Engine stopped"}

@app.post("/api/mcp/set-mode")
async def mcp_set_mode(mode: str = "NORMAL"):
    """Set trading mode. CONSERVATIVE is blocked."""
    engine.compound.set_mode(mode.upper())
    return {"result": f"Mode set to {engine.compound.state.mode}"}

@app.post("/api/mcp/set-capital")
async def mcp_set_capital(amount: float = 10.0):
    """Manually override capital amount."""
    engine.balance = amount
    sp("initial_capital", str(round(amount, 4)))
    sp("epoch_start_bal", str(round(amount, 4)))
    engine.compound.initialise(amount, engine.compound.state.epoch_num,
                                engine.compound.state.epoch_start_ts, amount)
    return {"result": f"Capital set to ${amount:.4f}"}

@app.post("/api/mcp/close-all")
async def mcp_close_all():
    """Emergency close all positions."""
    if not engine.client:
        return {"error": "No client connected"}
    closed = []
    for pos in get_open_positions():
        try:
            sym  = pos["symbol"]
            side = pos["side"]
            qty  = pos["qty"]
            await engine.client.close_pos(sym, side, qty)
            closed.append(sym)
        except Exception as e:
            logger.error(f"Close {pos['symbol']}: {e}")
    return {"result": f"Closed {len(closed)} positions", "symbols": closed}

@app.post("/api/mcp/force-whale-update")
async def mcp_force_whale():
    """Force immediate whale intelligence refresh."""
    if not engine.client:
        return {"error": "No client"}
    symbols = [x["symbol"] for x in engine.scanner.top] if engine.scanner else []
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ARBUSDT", "PEPEUSDT"]
    summary = await engine.whales.update_all(symbols, engine.client)
    return {"result": "Whale update complete", "summary": summary}

@app.post("/api/mcp/update-param")
async def mcp_update_param(key: str, value: str):
    """Update a strategy parameter."""
    sp(key, value)
    return {"result": f"Set {key}={value}"}

@app.get("/api/mcp/performance")
async def mcp_performance():
    """Performance summary with epoch history."""
    stats  = all_time_stats()
    epochs = get_all_epochs()
    return {"stats": stats, "epochs": epochs, "balance": engine.balance}

@app.get("/api/mcp/compound-progress")
async def mcp_compound_progress():
    """Compound growth progress."""
    ce = engine.compound
    s  = ce.state
    return {
        "epoch":         s.epoch_num,
        "day":           s.day_in_epoch,
        "start_bal":     s.epoch_start_bal,
        "current":       engine.balance,
        "target":        s.epoch_target,
        "ahead_pct":     ce.ahead_pct(engine.balance),
        "target_now":    ce.target_at_now(),
        "mode":          s.mode,
        "projection":    ce.project_growth(10),
    }

# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn_run("server:app", host="0.0.0.0", port=port, log_level="info")
