"""
server.py — FastAPI wrapper for the ByBitDouble Trading Engine
Exposes health checks and dashboard endpoints while running the bot in the background.
"""
import os, asyncio, logging, time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from contextlib import asynccontextmanager

# Internal modules
from engine import TradingEngine, CompoundTradingEngine
from bybit_client import BybitClient
from db import init_db, all_time_stats, get_open_positions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Lifespan & Startup ──────────────────────────────────────────────────────

engine: Optional[CompoundTradingEngine] = None
scanner = None  # Scanner is now internal to engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    • Startup: Init DB, Bybit client, TradingEngine, and background loop.
    • Shutdown: Gracefully stop the engine.
    """
    global engine
    logger.info("🚀 Initializing ByBitDouble Trading Engine...")

    try:
        # 1. Database
        init_db()

        # 2. Bybit Client
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

        if not api_key or not api_secret:
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set.")

        client = BybitClient(api_key, api_secret, testnet=testnet)

        # 3. Trading Engine
        # We pass the client and a default symbol list to the constructor
        # The engine.py __init__ should handle these arguments.
        symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT"
        ]
        
        # Initialize engine
        engine = TradingEngine() # Initialize with defaults if arguments are optional
        engine.init(api_key, api_secret, testnet=testnet)
        
        # If engine requires client/symbols in __init__ (per previous fixes), use:
        # engine = TradingEngine(client=client, symbols=symbols)
        # engine.init(api_key, api_secret, testnet=testnet)

        # 4. Start Background Loop
        # We run main_loop as a background task so FastAPI can still serve endpoints
        asyncio.create_task(engine.main_loop())
        
        logger.info("✅ Engine started successfully.")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

    yield  # Application is running here

    # Shutdown logic
    logger.info("🛑 Shutting down engine...")
    if engine:
        await engine.stop()
    logger.info("👋 Bye.")


app = FastAPI(
    title="ByBitDouble API",
    description="Dashboard & Health API for ByBitDouble Trading Bot",
    version="2.0",
    lifespan=lifespan
)

# CORS for local development / dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "ByBitDouble API is running."}

@app.get("/health")
async def health():
    """System health and status."""
    if not engine:
        return {"status": "error", "message": "Engine not initialized."}
    return {
        "status": "running" if engine.running else "stopped",
        "balance": engine.balance,
        "mode": engine.compound.state.mode if engine.compound else "N/A",
        "epoch": engine.compound.state.epoch_num if engine.compound else 0,
        "open_positions": len(engine.live_positions),
        "errors": engine.errors[-3:],
        "uptime": time.time() - engine.start_time if hasattr(engine, 'start_time') else 0
    }

@app.get("/positions")
async def positions():
    """Current open positions from DB."""
    try:
        pos = get_open_positions()
        return {"count": len(pos), "positions": pos}
    except Exception as e:
        return {"error": str(e)}

@app.get("/stats")
async def stats():
    """All-time performance statistics."""
    try:
        return all_time_stats()
    except Exception as e:
        return {"error": str(e)}

@app.get("/errors")
async def errors():
    """Recent errors."""
    if not engine: return []
    return engine.errors[-10:]

# ── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")