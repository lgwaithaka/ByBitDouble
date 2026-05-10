import asyncio, logging, sys, time
from fastapi import FastAPI
from uvicorn import run
from engine import TradingEngine
from scanner import MarketScanner
from db import init_db
from alerting import AlertManager
from config import BYBIT_TESTNET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
engine: TradingEngine = None
scanner: MarketScanner = None

@app.on_event("startup")
async def startup():
    global engine, scanner
    init_db()
    logger.info("Initializing 24/7 Trading Engine...")
    
    # Mock client for testing (replace with actual Bybit client)
    class MockClient: 
        pass
    client = MockClient()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ARBUSDT", "PEPEUSDT"]
    engine = TradingEngine(client, symbols)
    scanner = MarketScanner(engine, symbols)
    
    await engine.start()
    asyncio.create_task(scanner.run_continuous())
    asyncio.create_task(engine.alerts.heartbeat())
    logger.info("✅ All systems operational. Trading 24/7.")

@app.get("/health")
async def health():
    return {"status": "ok", "trading_active": engine.running if engine else False}

@app.on_event("shutdown")
async def shutdown():
    if engine:
        engine.running = False
        logger.info("Engine shutdown complete.")

if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, log_level="info")