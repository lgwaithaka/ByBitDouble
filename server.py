"""
server.py — FastMCP Compound Trading Server (port 8000)
fastmcp 3.x: use transport="http"
"""
import asyncio, json, os, time, threading, logging, sys
from typing import Optional

# Configure logging FIRST so all errors are visible in Render logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import with clear error messages
try:
    from fastmcp import FastMCP
    logger.info("fastmcp imported OK")
except ImportError as e:
    logger.critical(f"fastmcp import failed: {e}. Run: pip install fastmcp>=3.0.0")
    sys.exit(1)

try:
    from engine import engine
    from compound_engine import DAILY_REQUIRED_PCT, EPOCH_DAYS
    from db import (
        gp, sp, get_open_positions, get_trades, all_time_stats,
        get_all_epochs, get_capital_log, all_symbol_stats,
        get_hourly_snaps, all_params, init_db
    )
    logger.info("All local modules imported OK")
except ImportError as e:
    logger.critical(f"Local module import failed: {e}")
    sys.exit(1)

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SEC = os.getenv("BYBIT_API_SECRET", "")
TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

mcp  = FastMCP("Bybit Compound Trader")
_loop: Optional[asyncio.AbstractEventLoop] = None


# ── Root status page (visible in browser at https://your-service.onrender.com/) ──
@mcp.custom_route("/", methods=["GET"])
async def root_status(request):
    """Human-readable status page so the service URL isn't blank."""
    from starlette.responses import HTMLResponse
    try:
        stats = all_time_stats()
        ce    = engine.compound
        bal   = engine.balance
        cs    = ce.get_status(bal)
        body  = f"""
        <h2>Bybit Compound Trader — MCP Server</h2>
        <p><b>Status:</b> {engine.status}</p>
        <p><b>Epoch:</b> {cs.get('epoch_num','?')} | Day {cs.get('day_in_epoch','?')} of 5</p>
        <p><b>Mode:</b> {cs.get('mode','?')}</p>
        <p><b>Balance:</b> ${bal:.4f} | <b>Target:</b> ${cs.get('epoch_target',0):.4f}</p>
        <p><b>Total Trades:</b> {stats['total_trades']} | <b>Win Rate:</b> {stats['win_rate']}%</p>
        <p><b>Open Positions:</b> {stats['open_positions']}</p>
        <hr>
        <p>MCP endpoint for Claude Desktop: <code>{request.url.scheme}://{request.url.netloc}/mcp</code></p>
        <p>Dashboard: deploy the <b>bybit-compound-dashboard</b> service from the same repo.</p>
        """
    except Exception as e:
        body = f"<h2>Bybit Compound MCP</h2><p>Engine starting up... ({e})</p><p>MCP endpoint: <code>/mcp</code></p>"
    return HTMLResponse(f"<html><body style='font-family:monospace;padding:30px'>{body}</body></html>")


def _bg():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    engine.init(API_KEY, API_SEC, TESTNET)
    _loop.run_until_complete(engine.main_loop())


def _ensure():
    if not engine.running and API_KEY and API_SEC:
        threading.Thread(target=_bg, daemon=True).start()
        time.sleep(3)


def _call(coro, t=15):
    if _loop and _loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, _loop).result(t)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def start_trading() -> str:
    """Start the perpetual compound trading engine. Doubles every 5 days forever."""
    if not API_KEY or not API_SEC:
        return "ERROR: Set BYBIT_API_KEY and BYBIT_API_SECRET env vars."
    if engine.running: return f"Already running. Status: {engine.status}"
    _ensure()
    mode = "TESTNET" if TESTNET else "LIVE"
    return (f"Compound engine started [{mode}]. Epoch {engine.compound.state.epoch_num}. "
            f"Target: ${engine.compound.state.epoch_target:.2f} in {EPOCH_DAYS} days. "
            f"Scanning top-14 volatile perps every 40s.")


@mcp.tool()
def stop_trading() -> str:
    """Gracefully stop the engine."""
    if not engine.running: return "Engine not running."
    if _loop: asyncio.run_coroutine_threadsafe(engine.stop(), _loop)
    return "Stop signal sent."


@mcp.tool()
def get_status() -> str:
    """Full engine status: epoch, mode, balance, circuit breakers, streak."""
    ce    = engine.compound
    bal   = engine.balance
    stats = all_time_stats()
    return json.dumps({
        "engine":      engine.status,
        "testnet":     TESTNET,
        "balance_est": round(bal, 4),
        "compound":    ce.get_status(bal),
        "performance": stats,
        "last_scan_ago_s": int(time.time()) - engine.last_scan_ts if engine.last_scan_ts else None,
        "recent_signals":  engine.scan_results[-8:],
        "cooldowns":       {k: max(0, round(v-time.time())) for k, v in engine.cooldown_until.items() if v > time.time()},
        "errors":          engine.errors[-5:],
    }, indent=2, default=str)


@mcp.tool()
def get_compound_progress() -> str:
    """
    Detailed compound progress: epoch, target, ahead/behind, projection table.
    Shows what balance will be at each future epoch if targets are met.
    """
    ce    = engine.compound
    bal   = engine.balance
    stats = all_time_stats()
    est   = ce.state.epoch_start_bal + stats["total_pnl"]

    proj = ce.project_compounding(ce.state.epoch_start_bal, 12)
    return json.dumps({
        "current_epoch":      ce.state.epoch_num,
        "day_in_epoch":       ce.state.day_in_epoch,
        "mode":               ce.state.mode,
        "epoch_start_bal":    ce.state.epoch_start_bal,
        "epoch_target":       ce.state.epoch_target,
        "est_balance":        round(est, 4),
        "target_now":         round(ce.target_at_now(), 4),
        "target_eod":         round(ce.target_at_day_end(), 4),
        "ahead_pct":          round(ce.ahead_pct(est) * 100, 2),
        "on_track":           ce.state.on_track if hasattr(ce.state, 'on_track') else ce.ahead_pct(est) >= -0.08,
        "days_remaining":     round(ce.days_remaining(), 2),
        "daily_required_pct": round(DAILY_REQUIRED_PCT, 3),
        "epoch_projection":   proj,
        "all_epochs":         get_all_epochs(),
    }, indent=2, default=str)


@mcp.tool()
def get_all_positions() -> str:
    """All open positions: symbol, direction, entry, current price, unrealised PnL, SL, TP."""
    pos   = get_open_positions()
    stats = all_time_stats()
    return json.dumps({
        "open_count":        len(pos),
        "total_unrealised":  stats["unrealised_pnl"],
        "total_notional":    round(sum(p.get("notional",0) for p in pos), 2),
        "positions":         pos,
    }, indent=2, default=str)


@mcp.tool()
def get_performance(hours: int = 24) -> str:
    """Performance: PnL, win rate, epoch summary, capital curve."""
    stats  = all_time_stats()
    ce     = engine.compound
    trades = get_trades(hours=hours)
    closed = [t for t in trades if t["outcome"] != "OPEN"]
    h_pnl  = sum(t["pnl_usdt"] or 0 for t in closed)
    return json.dumps({
        "all_time":         stats,
        "compound_status":  ce.get_status(engine.balance),
        f"last_{hours}h": {
            "trades": len(closed),
            "pnl":    round(h_pnl, 4),
            "wins":   sum(1 for t in closed if t["outcome"] == "WIN"),
            "losses": sum(1 for t in closed if t["outcome"] == "LOSS"),
        },
        "all_epochs": get_all_epochs(),
    }, indent=2, default=str)


@mcp.tool()
def get_trade_history(hours: int = 24, symbol: str = None, epoch: int = None) -> str:
    """Closed trades with full details. Filter by hours, symbol, or epoch number."""
    trades = get_trades(hours=hours, symbol=symbol, epoch=epoch)
    closed = [t for t in trades if t["outcome"] != "OPEN"]
    return json.dumps({
        "count":     len(closed),
        "total_pnl": round(sum(t["pnl_usdt"] or 0 for t in closed), 4),
        "trades":    closed[:60],
    }, indent=2, default=str)


@mcp.tool()
def get_epoch_history() -> str:
    """All completed and current epochs with start/end balance, PnL, win rate."""
    return json.dumps({"epochs": get_all_epochs()}, indent=2, default=str)


@mcp.tool()
def get_symbol_breakdown() -> str:
    """Per-symbol: wins, losses, PnL, win rate, learned bias, adaptive SL/TP."""
    return json.dumps(all_symbol_stats(), indent=2, default=str)


@mcp.tool()
def get_hot_assets() -> str:
    """Current volatility scan: which assets are being targeted and their scores."""
    sc = engine.scanner
    return json.dumps({
        "last_scan_ts": sc.last_ts if sc else 0,
        "top_symbols":  sc.top if sc else [],
    }, indent=2)


@mcp.tool()
def get_market_snapshot(symbol: str = "BTCUSDT") -> str:
    """Live 20-factor signal analysis for any symbol without trading."""
    if not engine.client: return "Engine not initialised."
    from db import get_symbol_stats
    async def _snap():
        data = await engine.fetch_data(symbol)
        if not data: return {"error": f"No data for {symbol}"}
        from scanner import atr as _atr
        k5 = data["k5m"]
        price = float(k5[-1, 4]) if k5 is not None else 0
        ss    = get_symbol_stats(symbol)
        a     = engine.signals.analyze(
            sym=symbol, k3m=data["k3m"], k5m=data["k5m"],
            k15m=data["k15m"], k1h=data["k1h"],
            funding=data["funding"], ls=data["ls"],
            oi_pct=data["oi_pct"], ob_imb=data["ob_imb"],
            liqs=data["liqs"],
            learned_bias=ss.get("learned_bias",0.0),
            mode=engine.compound.state.mode,
        )
        return {"symbol":symbol,"price":price,"atr":_atr(k5,14),
                **a,"mode":engine.compound.state.mode}
    if _loop:
        r = asyncio.run_coroutine_threadsafe(_snap(), _loop).result(20)
        return json.dumps(r, indent=2)
    return "Loop not running."


@mcp.tool()
def update_param(key: str, value: str) -> str:
    """Update a strategy parameter live. Keys: epoch_max_dd_pct, daily_max_dd_pct,
    vol_scan_n, scan_interval_s, initial_capital"""
    valid = ["epoch_max_dd_pct","daily_max_dd_pct","vol_scan_n","scan_interval_s","initial_capital"]
    if key not in valid: return f"Invalid. Valid keys: {valid}"
    sp(key, value); return f"Set {key}={value}"


@mcp.tool()
def get_all_params() -> str:
    """All current parameters."""
    return json.dumps(all_params(), indent=2)


@mcp.tool()
def reset_circuit_breaker(level: str = "daily") -> str:
    """Reset circuit breaker. level='daily' or 'epoch'."""
    if level == "epoch":
        engine.compound.reset_epoch_cb()
        return "Epoch circuit breaker reset. Trading resumes."
    else:
        engine.compound.reset_daily_cb()
        return "Daily circuit breaker reset. Trading resumes."


@mcp.tool()
def set_mode(mode: str) -> str:
    """Manually override mode: CONSERVATIVE | NORMAL | AGGRESSIVE | TURBO."""
    valid = {"CONSERVATIVE", "NORMAL", "AGGRESSIVE", "TURBO"}
    if mode not in valid: return f"Invalid. Choose from: {valid}"
    engine.compound.state.mode = mode
    return f"Mode set to {mode}. Will adapt automatically on next trade outcome."


@mcp.tool()
def close_position(symbol: str) -> str:
    """Emergency market close a single position."""
    if not engine.client: return "Engine not initialised."
    pos = next((p for p in get_open_positions() if p["symbol"] == symbol), None)
    if not pos: return f"No open position for {symbol}."
    if _loop:
        r = asyncio.run_coroutine_threadsafe(
            engine.client.close_pos(symbol, pos["side"], str(pos["qty"])), _loop
        ).result(15)
        return f"Close {'OK' if r.get('retCode',0)==0 else 'FAILED: ' + r.get('retMsg','?')}"
    return "Loop not running."


@mcp.tool()
def close_all_positions() -> str:
    """Emergency close ALL open positions."""
    if not engine.client: return "Engine not initialised."
    pos = get_open_positions()
    if not pos: return "No open positions."
    results = []
    for p in pos:
        if _loop:
            r = asyncio.run_coroutine_threadsafe(
                engine.client.close_pos(p["symbol"], p["side"], str(p["qty"])), _loop
            ).result(15)
            results.append(f"{p['symbol']}: {'OK' if r.get('retCode',0)==0 else 'FAIL'}")
    return "\n".join(results)


@mcp.tool()
def force_epoch_advance() -> str:
    """Force advance to next epoch (use if you want to reset epoch clock manually)."""
    ce  = engine.compound
    old = ce.state.epoch_num
    ce.state.epoch_start_ts = int(time.time()) - 5 * 86400  # trick boundary check
    if _loop:
        asyncio.run_coroutine_threadsafe(engine.check_epoch_boundary(), _loop).result(15)
    return f"Forced epoch advance from {old} to {engine.compound.state.epoch_num}."


@mcp.tool()
def project_compound_growth(epochs: int = 15) -> str:
    """Project compound balance milestones for N future epochs."""
    ce   = engine.compound
    proj = ce.project_compounding(ce.state.epoch_start_bal, epochs)
    return json.dumps({
        "current_epoch":    ce.state.epoch_num,
        "current_stake":    ce.state.epoch_start_bal,
        "projections":      proj,
        "note": "Each row assumes 2× achieved in that 5-day epoch."
    }, indent=2)


if __name__ == "__main__":
    # Robust startup — full error logging so Render shows what failed
    try:
        init_db()
        logger.info("DB initialised OK")
    except Exception as e:
        logger.error(f"DB init error: {e}")

    if API_KEY and API_SEC:
        _ensure()
        mode = "TESTNET" if TESTNET else "LIVE"
        logger.info(f"Compound engine starting [{mode}]")
    else:
        logger.warning(
            "BYBIT_API_KEY / BYBIT_API_SECRET not set. "
            "Add them in Render > Environment Variables, then redeploy."
        )

    # fastmcp 3.x: transport must be 'http' not 'streamable-http'
    logger.info("FastMCP HTTP server on 0.0.0.0:8000")
    mcp.run(transport="http", host="0.0.0.0", port=8000)
