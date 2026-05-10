import asyncio, logging, sys, time
from fastapi import FastAPI
from uvicorn import run
from engine import TradingEngine
from scanner import MarketScanner
from db import init_db, all_time_stats, gp, sp
from alerting import AlertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ByBitDouble Trading Bot")
engine: TradingEngine = None
scanner: MarketScanner = None

@app.on_event("startup")
async def startup():
    global engine, scanner
    init_db()
    logger.info("🚀 Initializing ByBitDouble Trading Engine...")
    
    # Mock client (replace with actual Bybit client)
    class MockClient: 
        pass
    client = MockClient()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ARBUSDT", "PEPEUSDT"]
    engine = TradingEngine(client, symbols)
    scanner = MarketScanner(engine, symbols)
    
    # Start engine
    await engine.start()
    
    # Start scanner as background task
    asyncio.create_task(scanner.run_continuous())
    
    # Start heartbeat
    asyncio.create_task(engine.alerts.heartbeat())
    
    logger.info("✅ All systems operational | Trading 24/7 | Scanning every 60s")
    await engine.alerts.send("✅ System started | 60s scanning active", "SUCCESS", force=True)

@app.get("/health")
async def health():
    """Health check endpoint for Render."""
    return {
        "status": "ok",
        "trading_active": engine.running if engine else False,
        "scanner_active": scanner.last_scan > 0 if scanner else False,
        "uptime_seconds": int(time.time()) - engine.start_time if engine else 0
    }

@app.get("/scanner-status")
async def scanner_status():
    """Get current scanner status."""
    if not scanner:
        return {"error": "Scanner not initialized"}
    return scanner.get_status()

@app.get("/dashboard-stats")
async def dashboard_stats():
    """Get comprehensive dashboard statistics."""
    stats = all_time_stats()
    scan_status = scanner.get_status() if scanner else {}
    
    return {
        **stats,
        "scanner": scan_status,
        "last_update": time.time(),
        "mode": gp("mode", "NORMAL")
    }

@app.on_event("shutdown")
async def shutdown():
    if engine:
        engine.running = False
        logger.info("Engine shutdown complete.")

if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, log_level="info")