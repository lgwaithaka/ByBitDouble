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

# 2. Add heartbeat monitoring (prevent silent failures)
#    In server.py, add to main_loop():
if time.time() - self.last_heartbeat_ts > 300:  # 5 min
    await self._send_heartbeat()  # Ping Discord/Telegram webhook

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD REST API — served directly from the MCP server
# This eliminates the shared-disk problem of a separate dashboard service.
# Dashboard URL: https://bybitdouble.onrender.com/
# ════════════════════════════════════════════════════════════════════════════

from starlette.responses import HTMLResponse, JSONResponse as SJSONResponse
from starlette.middleware.cors import CORSMiddleware as SCORSMiddleware
from pathlib import Path as _Path


def _safe_ce():
    """Get compound engine status safely even when balance is 0."""
    ce  = engine.compound
    bal = max(engine.balance, 0.0)
    try:
        return ce.get_status(bal)
    except Exception:
        return {
            "epoch_num": ce.state.epoch_num, "day_in_epoch": 1,
            "epoch_target": ce.state.epoch_target, "epoch_start_bal": ce.state.epoch_start_bal,
            "epoch_progress_pct": 0.0, "mode": ce.state.mode,
            "on_track": False, "ahead_pct": 0.0, "streak": 0,
            "days_remaining": 5.0, "target_eod": ce.state.epoch_target, "target_now": ce.state.epoch_start_bal,
        }


@mcp.custom_route("/", methods=["GET"])
async def dashboard_html(request):
    """Serve the live dashboard HTML."""
    f = _Path(__file__).parent / "static" / "dashboard.html"
    if f.exists():
        return HTMLResponse(f.read_text())
    return HTMLResponse("<h2>Dashboard not found. Push static/dashboard.html to repo.</h2>")


@mcp.custom_route("/api/overview", methods=["GET"])
async def api_overview(request):
    stats   = all_time_stats()
    initial = float(gp("initial_capital","0") or "0")
    bal     = engine.balance if engine.balance > 0 else initial
    est     = bal if bal > 0 else (initial + stats["total_pnl"]) if initial > 0 else 0.0
    cs      = _safe_ce()
    epochs  = get_all_epochs()
    ce      = engine.compound
    return SJSONResponse({
        "stats":           stats,
        "compound":        cs,
        "est_balance":     round(est, 4),
        "initial":         round(initial, 4),
        "epoch_history":   epochs,
        "projection":      ce.project_compounding(max(ce.state.epoch_start_bal, est, 10.0), 10),
        "engine_status":   engine.status,
        "market_regime":   gp("market_regime","RANGING"),
        "regime_label":    gp("regime_label",""),
        "hourly_win_rate": gp("hourly_win_rate","0"),
        "hourly_pnl":      gp("hourly_pnl","0"),
        "last_strategy_ts":gp("last_strategy_ts","0"),
        "testnet":         TESTNET,
        "whale_summary":   _safe_json(gp("whale_summary","{}")),
        "market_feeds": {
            "fear_greed_value": gp("fear_greed_value","50"),
            "fear_greed_label": gp("fear_greed_label","Neutral"),
            "trading_hour_active": engine.whales and hasattr(engine,"_is_trading_hour") and engine._is_trading_hour(),
            "auto_blacklisted_count": len(getattr(engine,"_last_blacklist",[])),
        },
    })


@mcp.custom_route("/api/positions", methods=["GET"])
async def api_positions(request):
    pos  = get_open_positions()
    unr  = sum(p.get("unrealised_pnl",0) or 0 for p in pos)
    not_ = sum(p.get("notional",0) or 0 for p in pos)
    return SJSONResponse({"positions":pos,"count":len(pos),
                          "total_unrealised":round(unr,4),"total_notional":round(not_,2)})


@mcp.custom_route("/api/trades", methods=["GET"])
async def api_trades(request):
    params = dict(request.query_params)
    hours  = int(params.get("hours","24"))
    symbol = params.get("symbol") or None
    epoch  = int(params.get("epoch","0")) or None
    ts     = get_trades(hours=hours, symbol=symbol, epoch=epoch)
    cl     = [t for t in ts if t["outcome"] != "OPEN"]
    return SJSONResponse({"count":len(cl),"pnl":round(sum(t["pnl_usdt"] or 0 for t in cl),4),"trades":cl})


@mcp.custom_route("/api/capital-curve", methods=["GET"])
async def api_capital_curve(request):
    ce  = engine.compound
    log = get_capital_log(300)
    for pt in log:
        elapsed = max(0,(pt["ts"]-ce.state.epoch_start_ts)/(EPOCH_DAYS*86400))
        pt["target"] = round(ce.state.epoch_start_bal*(2.0**elapsed),4)
    return SJSONResponse({"log":log})


@mcp.custom_route("/api/hourly", methods=["GET"])
async def api_hourly(request):
    return SJSONResponse({"snapshots":get_hourly_snaps(96)})


@mcp.custom_route("/api/symbols", methods=["GET"])
async def api_symbols(request):
    return SJSONResponse({"symbols":all_symbol_stats()})


@mcp.custom_route("/api/epochs", methods=["GET"])
async def api_epochs(request):
    return SJSONResponse({"epochs":get_all_epochs()})


@mcp.custom_route("/api/params", methods=["GET"])
async def api_params_get(request):
    return SJSONResponse(all_params())


@mcp.custom_route("/api/whale", methods=["GET"])
async def api_whale(request):
    summary = _safe_json(gp("whale_summary","{}"))
    opps    = _safe_json(gp("whale_opportunities","[]"))
    last_upd= int(gp("whale_last_update","0") or "0")
    return SJSONResponse({
        "summary":       summary,
        "opportunities": opps,
        "last_update":   last_upd,
        "age_minutes":   round((time.time()-last_upd)/60,1) if last_upd else None,
        "cache":         engine.whales.get_all_cache() if hasattr(engine,"whales") else {},
    })


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health(request):
    return SJSONResponse({
        "status":  "ok",
        "engine":  engine.status,
        "balance": round(engine.balance,4),
        "testnet": TESTNET,
        "ts":      int(time.time()),
    })


def _safe_json(s, default=None):
    if default is None: default = {}
    try:    return json.loads(s or "{}")
    except: return default


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
def get_whale_intelligence() -> str:
    """
    Get current smart money / whale tracking data for all scanned symbols.
    Shows OI trends, funding extremes, liquidation cascades, large trade flows.
    Updates every hour automatically.
    """
    cache    = engine.whales.get_all_cache()
    summary  = engine.whales.last_summary
    opps     = engine.whales.get_top_opportunities(8)
    last_upd = int(gp("whale_last_update", "0") or "0")
    age_min  = round((time.time() - last_upd) / 60, 1) if last_upd else None

    return json.dumps({
        "last_update_ago_minutes": age_min,
        "market_summary":          summary,
        "top_opportunities":       opps,
        "symbol_detail":           cache,
    }, indent=2, default=str)


@mcp.tool()
def get_whale_opportunities() -> str:
    """
    Get top whale-backed trading opportunities right now.
    These are symbols where smart money signals are strongest.
    """
    opps = engine.whales.get_top_opportunities(10)
    return json.dumps({
        "opportunities": opps,
        "market_bias":   engine.whales.last_summary.get("market_bias", "UNKNOWN"),
        "short_squeezes":engine.whales.last_summary.get("short_squeezes", []),
        "funding_extremes": engine.whales.last_summary.get("funding_extremes", []),
    }, indent=2)


@mcp.tool()
def force_whale_update() -> str:
    """Force an immediate whale intelligence refresh for all scanned symbols."""
    if not engine.client or not engine.scanner:
        return "Engine not initialised."
    async def _update():
        syms = [x["symbol"] for x in engine.scanner.top] if engine.scanner.top else []
        if not syms:
            return {"error": "No symbols scanned yet. Call start_trading first."}
        result = await engine.whales.update_all(syms, engine.client)
        return {
            "symbols_updated": len(result),
            "summary":         engine.whales.last_summary,
            "top_opportunities": engine.whales.get_top_opportunities(5),
        }
    if _loop:
        r = asyncio.run_coroutine_threadsafe(_update(), _loop).result(60)
        return json.dumps(r, indent=2, default=str)
    return "Engine loop not running."


@mcp.tool()
def analyze_trade_performance() -> str:
    """
    Analyze your recent trade outcomes and identify patterns.
    Shows win rate by symbol, time of day, signal type, and market regime.
    """
    trades = get_trades(hours=168)  # Last 7 days
    closed = [t for t in trades if t["outcome"] not in ("OPEN", None)]

    if len(closed) < 4:
        return json.dumps({
            "message": f"Only {len(closed)} closed trades. Need more data for pattern analysis.",
            "suggestion": "The system needs at least 10+ trades to identify meaningful patterns.",
        }, indent=2)

    # By symbol
    sym_perf = {}
    for t in closed:
        s = t["symbol"]
        if s not in sym_perf:
            sym_perf[s] = {"wins":0,"losses":0,"pnl":0.0}
        if t["outcome"]=="WIN":   sym_perf[s]["wins"]+=1
        elif t["outcome"]=="LOSS":sym_perf[s]["losses"]+=1
        sym_perf[s]["pnl"] += t["pnl_usdt"] or 0

    for s in sym_perf:
        n = sym_perf[s]["wins"] + sym_perf[s]["losses"]
        sym_perf[s]["win_rate"] = round(sym_perf[s]["wins"]/n*100,1) if n>0 else 0
        sym_perf[s]["total_trades"] = n

    # By hour of day
    from datetime import datetime, timezone as tz
    hour_perf = {}
    for t in closed:
        h = datetime.fromtimestamp(t["ts_open"], tz=tz.utc).hour
        if h not in hour_perf:
            hour_perf[h] = {"wins":0,"losses":0,"pnl":0.0}
        if t["outcome"]=="WIN":    hour_perf[h]["wins"]+=1
        elif t["outcome"]=="LOSS": hour_perf[h]["losses"]+=1
        hour_perf[h]["pnl"] += t["pnl_usdt"] or 0

    # By mode
    mode_perf = {}
    for t in closed:
        m = t.get("mode","NORMAL") or "NORMAL"
        if m not in mode_perf:
            mode_perf[m] = {"wins":0,"losses":0,"pnl":0.0}
        if t["outcome"]=="WIN":    mode_perf[m]["wins"]+=1
        elif t["outcome"]=="LOSS": mode_perf[m]["losses"]+=1
        mode_perf[m]["pnl"] += t["pnl_usdt"] or 0

    wins   = sum(1 for t in closed if t["outcome"]=="WIN")
    losses = sum(1 for t in closed if t["outcome"]=="LOSS")
    total_pnl = sum(t["pnl_usdt"] or 0 for t in closed)
    avg_win   = sum(t["pnl_usdt"] or 0 for t in closed if t["outcome"]=="WIN") / max(wins,1)
    avg_loss  = sum(t["pnl_usdt"] or 0 for t in closed if t["outcome"]=="LOSS") / max(losses,1)

    return json.dumps({
        "summary": {
            "total_trades": len(closed),
            "wins": wins, "losses": losses,
            "win_rate": round(wins/len(closed)*100,1) if closed else 0,
            "total_pnl": round(total_pnl,4),
            "avg_win": round(avg_win,4),
            "avg_loss": round(avg_loss,4),
            "profit_factor": round(abs(avg_win/avg_loss),2) if avg_loss!=0 else 0,
        },
        "by_symbol": {k: v for k,v in sorted(sym_perf.items(), key=lambda x: x[1]["pnl"], reverse=True)},
        "by_hour_utc": {str(k): v for k,v in sorted(hour_perf.items())},
        "by_mode": mode_perf,
        "problems_identified": _identify_problems(closed, wins, losses, sym_perf),
    }, indent=2, default=str)


def _identify_problems(closed, wins, losses, sym_perf):
    problems = []
    n = len(closed)
    wr = wins/n*100 if n>0 else 0

    if wr < 40:
        problems.append(f"WIN RATE CRITICAL: {wr:.1f}% — signals firing on too-low confidence. Increase confidence_floor.")
    if wr < 50 and n>=5:
        problems.append("Consider: stop_trading and run check_bybit_connection to verify feed quality.")

    bad_syms = [s for s,v in sym_perf.items() if v["losses"]>2 and (v["wins"]/(v["wins"]+v["losses"]))<0.35]
    if bad_syms:
        problems.append(f"Poor symbols: {bad_syms} — these are consistently losing. Consider blacklisting.")

    if not problems:
        problems.append("No major problems detected. Keep running!")

    return problems



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
    """Full engine status: live balance, epoch, mode, circuit breakers, streak."""
    ce    = engine.compound
    bal   = engine.balance
    stats = all_time_stats()
    initial = float(gp("initial_capital","0") or "0")
    epoch_bal = float(gp("epoch_start_bal","0") or "0")

    status = {
        "engine":       engine.status,
        "testnet":      TESTNET,
        "live_balance": round(bal, 4),
        "initial_capital": round(initial, 4),
        "epoch_start_bal": round(epoch_bal, 4),
        "capital_source": (
            "AUTO-DETECTED from Bybit" if not engine.capital_auto_detected and initial > 0
            else "PENDING — engine fetching live balance from Bybit..."
            if engine.capital_auto_detected
            else "NOT SET — check API keys and Bybit Unified Trading Account"
        ),
        "compound":     ce.get_status(bal),
        "performance":  stats,
        "last_scan_ago_s": int(time.time()) - engine.last_scan_ts if engine.last_scan_ts else None,
        "recent_signals":  engine.scan_results[-8:],
        "cooldowns":       {k: max(0, round(v-time.time())) for k, v in engine.cooldown_until.items() if v > time.time()},
        "errors":          engine.errors[-5:],
    }
    return json.dumps(status, indent=2, default=str)


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
    """
    Manually override trading mode: NORMAL | AGGRESSIVE | TURBO
    CONSERVATIVE has been removed — minimum mode is NORMAL.
    TURBO = maximum frequency, largest positions, shortest cooldowns.
    """
    valid = {"NORMAL", "AGGRESSIVE", "TURBO"}
    if mode.upper() == "CONSERVATIVE":
        return "CONSERVATIVE mode removed. Minimum is NORMAL. Try: set_mode AGGRESSIVE"
    if mode not in valid: return f"Invalid. Choose: {valid}"
    engine.compound.state.mode = mode
    risk  = round(engine.compound.compute_risk_pct()*100, 1)
    conc  = engine.compound.compute_max_concurrent()
    conf  = engine.compound.compute_confidence_floor()
    return (f"Mode set to {mode}.\n"
            f"Risk/trade: {risk}% | Max positions: {conc} | Conf floor: {conf}\n"
            f"Pyramid (add to winners): {'YES' if mode in ('AGGRESSIVE','TURBO') else 'NO'}")


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




@mcp.tool()
def check_bybit_connection() -> str:
    """
    DIAGNOSTIC TOOL — Run this first if balance shows 0.
    Tests the Bybit API connection, shows balance across ALL account types,
    confirms testnet/live mode, checks API key permissions.
    """
    if not engine.client:
        return "Engine not initialised. Call start_trading first."

    async def _diag():
        import time as _time
        out = {}

        # 1. Basic connectivity
        try:
            ticker = await engine.client._get("/v5/market/tickers",
                {"category": "linear", "symbol": "BTCUSDT"}, auth=False)
            price = ticker.get("result",{}).get("list",[{}])[0].get("lastPrice","?")
            out["connectivity"] = f"OK — BTCUSDT price: ${price}"
        except Exception as e:
            out["connectivity"] = f"FAILED: {e}"

        # 2. API key mode
        out["mode"]     = "TESTNET" if engine.client.base == "https://api-testnet.bybit.com" else "LIVE"
        out["base_url"] = engine.client.base
        out["api_key_prefix"] = engine.client.api_key[:8] + "..." if engine.client.api_key else "NOT SET"

        # 3. All account balances
        try:
            all_bals = await engine.client.wallet_all_types()
            out["account_balances"] = all_bals
        except Exception as e:
            out["account_balances"] = f"Error: {e}"

        # 4. Account info (shows if unified trading enabled)
        try:
            acct = await engine.client.account_info()
            info = acct.get("result", {})
            out["account_info"] = {
                "unifiedMarginStatus": info.get("unifiedMarginStatus"),
                "marginMode":          info.get("marginMode"),
                "accountType":         info.get("accountType"),
                "dcpStatus":           info.get("dcpStatus"),
                "retCode":             acct.get("retCode"),
                "retMsg":              acct.get("retMsg"),
            }
        except Exception as e:
            out["account_info"] = f"Error: {e}"

        # 5. Recommendation
        rec = []
        if out["mode"] == "TESTNET":
            rec.append("ISSUE: Still on TESTNET. Set BYBIT_TESTNET=false in Render env vars and redeploy.")
        if not engine.client.api_key:
            rec.append("ISSUE: BYBIT_API_KEY not set in Render environment variables.")
        bals = out.get("account_balances", {})
        if isinstance(bals, dict):
            has_funds = any(
                float(v.get("totalWalletBalance","0") or "0") > 0
                for v in bals.values() if isinstance(v, dict) and "error" not in v
            )
            if not has_funds:
                rec.append("ISSUE: Zero balance across all account types. Check:")
                rec.append("  1. Bybit → Assets → make sure funds are in Unified Trading (not Funding)")
                rec.append("  2. API key has 'Unified Trading Read' permission")
                rec.append("  3. Transfer: Bybit Assets > Transfer > Funding → Unified Trading")
        out["recommendations"] = rec if rec else ["All looks good! Call refresh_balance then get_status."]
        return out

    if _loop:
        result = asyncio.run_coroutine_threadsafe(_diag(), _loop).result(20)
        return json.dumps(result, indent=2, default=str)
    return "Engine loop not running. Call start_trading first."


@mcp.tool()
def refresh_balance() -> str:
    """Force an immediate balance refresh from Bybit and return the result."""
    if not engine.client:
        return "Engine not initialised."
    if _loop:
        bal = asyncio.run_coroutine_threadsafe(engine.refresh_balance(), _loop).result(15)
        sp("epoch_start_bal", str(round(bal, 4)))
        return (
            f"Live Balance: ${bal:.4f} USDT\n"
            f"Epoch Target: ${engine.compound.state.epoch_target:.4f} USDT\n"
            f"Mode: {engine.compound.state.mode}"
        )
    return "Engine loop not running."


@mcp.tool()
def set_capital(amount: float) -> str:
    """
    Manually override the capital amount.
    Normally the engine auto-detects this from your live Bybit balance on startup.
    Only use this if auto-detection failed or you want to reset the epoch.
    Example: set_capital(10.0)
    """
    if amount < 5:
        return "Minimum capital is $5 USDT (Bybit minimum order size constraints)."
    import time as _t
    sp("initial_capital", str(round(amount, 4)))
    sp("epoch_start_bal", str(round(amount, 4)))
    sp("epoch_start_ts",  str(int(_t.time())))
    sp("current_epoch",   "1")

    # Adjust operational params to balance size
    if amount < 20:
        sp("min_notional_usdt","5.5"); sp("max_concurrent","3"); sp("vol_scan_n","8")
    elif amount < 100:
        sp("min_notional_usdt","5.5"); sp("max_concurrent","5"); sp("vol_scan_n","10")
    elif amount < 500:
        sp("min_notional_usdt","10.0"); sp("max_concurrent","8"); sp("vol_scan_n","12")
    else:
        sp("min_notional_usdt","20.0"); sp("max_concurrent","12"); sp("vol_scan_n","14")

    if engine.running:
        engine.balance = amount
        engine.capital_auto_detected = False
        engine.compound.initialise(
            start_balance   = amount,
            epoch_num       = 1,
            epoch_start_ts  = int(_t.time()),
            epoch_start_bal = amount,
        )
    from db import open_epoch_record
    open_epoch_record(1, amount)

    proj = engine.compound.project_compounding(amount, 8)
    return (
        f"Capital manually set to ${amount:.4f} USDT\n"
        f"Epoch 1 target: ${amount * 2:.4f} USDT in 5 days\n\n"
        f"Compound projection:\n"
        + "\n".join(
            f"  Epoch {p['epoch']}: ${p['start']:.2f} → ${p['target']:.2f}  (Day {p['days_elapsed']})"
            for p in proj
        )
    )


@mcp.tool()
def get_micro_status() -> str:
    """
    Status report optimised for micro-accounts ($5-$50).
    Shows balance, positions, what the bot can afford to trade.
    """
    bal   = engine.balance
    min_n = float(gp("min_notional_usdt", "5.5"))
    ce    = engine.compound
    pos   = get_open_positions()
    stats = all_time_stats()

    affordable = []
    for mode_lev in [10, 15, 20, 25]:
        margin_needed = min_n / mode_lev
        if bal >= margin_needed * 1.5:
            affordable.append(f"  {mode_lev}x leverage → needs ${margin_needed:.2f} margin (✓ affordable)")
        else:
            affordable.append(f"  {mode_lev}x leverage → needs ${margin_needed:.2f} margin (✗ too low)")

    return json.dumps({
        "balance_usdt":        round(bal, 4),
        "epoch":               ce.state.epoch_num,
        "epoch_target":        ce.state.epoch_target,
        "epoch_progress_pct":  round(ce.epoch_progress_pct(bal), 1),
        "mode":                ce.state.mode,
        "open_positions":      len(pos),
        "max_concurrent":      gp("max_concurrent", "3"),
        "bybit_min_notional":  min_n,
        "leverage_affordability": affordable,
        "total_pnl":           stats["total_pnl"],
        "win_rate":            stats["win_rate"],
        "positions":           pos,
    }, indent=2, default=str)



@mcp.tool()
def get_regime_status() -> str:
    """Get current market regime (TRENDING/RANGING/VOLATILE) and strategy adjustments."""
    regime = engine.regime
    adj    = regime.adjustments()
    return json.dumps({
        "regime":           regime.regime,
        "label":            adj["label"],
        "trend_strength":   round(regime.trend_strength, 5),
        "volatility_pct":   round(regime.volatility_pct, 2),
        "last_updated":     regime.last_updated,
        "adjustments":      adj,
        "conf_bonus":       adj["conf_bonus"],
        "losing_symbols":   list(engine.losing_symbols),
        "cooldowns":        {k: max(0,round(v-time.time())) for k,v in engine.cooldown_until.items() if v>time.time()},
        "hourly_pnl":       gp("hourly_pnl","0"),
        "hourly_win_rate":  gp("hourly_win_rate","0"),
        "last_strategy_ts": gp("last_strategy_ts","0"),
    }, indent=2)


@mcp.tool()
def get_diversification_status() -> str:
    """Show open positions by asset class and diversification health."""
    from engine import get_asset_class
    pos    = get_open_positions()
    by_class = {}
    for p in pos:
        cls = get_asset_class(p["symbol"])
        by_class[cls] = by_class.get(cls, []) + [p["symbol"]]
    return json.dumps({
        "open_positions":     len(pos),
        "by_asset_class":     by_class,
        "losing_symbols":     list(engine.losing_symbols),
        "blocked_from_open":  list(engine.losing_symbols),
        "cooldowns":          {k: max(0,round(v-time.time())) for k,v in engine.cooldown_until.items() if v>time.time()},
        "max_concurrent":     engine.compound.compute_max_concurrent(),
        "positions":          pos,
    }, indent=2, default=str)


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
