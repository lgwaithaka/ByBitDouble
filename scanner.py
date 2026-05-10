import asyncio, logging, time
from typing import List, Dict, Optional
from db import gp, sp, get_open_positions, all_time_stats
from alerting import AlertManager

logger = logging.getLogger(__name__)

class MarketScanner:
    """Continuous market scanner that runs every 60 seconds and updates dashboard."""
    
    def __init__(self, engine, symbols: List[str]):
        self.engine = engine
        self.symbols = symbols
        self.alerts = engine.alerts if hasattr(engine, 'alerts') else AlertManager()
        self.last_scan = 0
        self.scan_interval_s = 60  # Fixed: Scan every 60 seconds
        self.total_scans = 0
        self.signals_found = 0
        
    async def run_continuous(self):
        """Main scanning loop - runs indefinitely every 60 seconds."""
        logger.info(f"🔍 Scanner started | Interval: {self.scan_interval_s}s | Symbols: {len(self.symbols)}")
        await self.alerts.send("🔍 Scanner started | 60s continuous scan active", "INFO", force=True)
        
        while self.engine.running if hasattr(self.engine, 'running') else True:
            try:
                start_time = time.time()
                
                # Run scan cycle
                await self.scan_cycle()
                
                # Update last scan timestamp
                self.last_scan = int(time.time())
                self.total_scans += 1
                
                # Calculate duration
                duration = time.time() - start_time
                logger.info(f"✅ Scan #{self.total_scans} complete | Duration: {duration:.1f}s | Next in {self.scan_interval_s}s")
                
                # Wait until next scan interval
                await asyncio.sleep(self.scan_interval_s)
                
            except asyncio.CancelledError:
                logger.info("Scanner stopped (cancelled)")
                break
            except Exception as e:
                logger.error(f"❌ Scanner error: {e}", exc_info=True)
                await self.alerts.error_alert("Scanner crash", str(e))
                # Don't crash - wait and retry
                await asyncio.sleep(10)
    
    async def scan_cycle(self):
        """Single scan cycle: fetch data, evaluate signals, update dashboard."""
        mode = gp("mode", "NORMAL")
        logger.info(f"🔄 Starting scan cycle | Mode: {mode} | Symbols: {len(self.symbols)}")
        
        # Get market context (Fear & Greed, BTC dominance, etc.)
        market_ctx = await self._fetch_market_context()
        
        # Scan each symbol
        tasks = [self._evaluate_symbol(sym, market_ctx, mode) for sym in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        signals = [r for r in results if isinstance(r, dict) and r.get("signal")]
        self.signals_found += len(signals)
        
        # Update dashboard stats
        await self._update_dashboard_stats(signals)
        
        if signals:
            logger.info(f"📊 Found {len(signals)} potential signals this cycle")
        
    async def _evaluate_symbol(self, sym: str, market_ctx: Dict, mode: str) -> Optional[Dict]:
        """Evaluate a single symbol for trading signals."""
        try:
            # Fetch market data
            data = await self._fetch_symbol_data(sym)
            
            # Check if we can open position
            if not self._can_open_position(sym, mode):
                return None
            
            # Analyze signal
            signal = await self._analyze_signal(sym, data, market_ctx, mode)
            
            if signal and signal.get("confidence", 0) >= signal.get("threshold", 60):
                logger.info(f"🎯 Signal: {sym} {signal['side']} | Conf: {signal['confidence']:.1f}% | TP: {signal.get('target_tp', 20)}%")
                return signal
                
        except Exception as e:
            logger.error(f"Error evaluating {sym}: {e}")
            
        return None
    
    async def _fetch_market_context(self) -> Dict:
        """Fetch overall market context (Fear & Greed, BTC trend, etc.)."""
        # Replace with actual API calls
        return {
            "fear_greed": 55,  # 0-100
            "btc_trend": "NEUTRAL",  # BULL/BEAR/NEUTRAL
            "market_volatility": "NORMAL",  # LOW/NORMAL/HIGH
            "timestamp": time.time()
        }
    
    async def _fetch_symbol_data(self, sym: str) -> Dict:
        """Fetch symbol-specific market data."""
        # Replace with actual Bybit API calls
        return {
            "symbol": sym,
            "price": 1.0,  # Mock price
            "volume_24h": 1000000,
            "funding_rate": 0.0001,
            "oi_change": 0.02,
            "timestamp": time.time()
        }
    
    def _can_open_position(self, sym: str, mode: str) -> bool:
        """Check if we can open a new position for this symbol."""
        # Check if already in position
        open_positions = get_open_positions()
        if any(pos["symbol"] == sym for pos in open_positions):
            return False
            
        # Check mode-specific limits
        max_concurrent = int(gp("max_concurrent", "8"))
        if len(open_positions) >= max_concurrent:
            logger.debug(f"Max concurrent positions reached ({max_concurrent})")
            return False
            
        return True
    
    async def _analyze_signal(self, sym: str, data: Dict, market_ctx: Dict, mode: str) -> Optional[Dict]:
        """
        Analyze symbol and generate trading signal.
        Returns dict with: side, confidence, threshold, entry_price, target_tp
        """
        # This is where your signal logic goes
        # For now, return placeholder - replace with your actual analysis
        
        # Example logic:
        confidence = 65.0  # Replace with actual calculation
        threshold = 60.0 if mode == "NORMAL" else 55.0
        
        if confidence >= threshold:
            return {
                "symbol": sym,
                "side": "Buy",  # or "Sell"
                "confidence": confidence,
                "threshold": threshold,
                "entry_price": data["price"],
                "target_tp": 20.0,  # Minimum 20% TP
                "mode": mode,
                "timestamp": time.time()
            }
            
        return None
    
    async def _update_dashboard_stats(self, signals: List[Dict]):
        """Update database with latest scan statistics."""
        try:
            stats = all_time_stats()
            sp("last_scan_ts", str(int(time.time())))
            sp("total_scans", str(self.total_scans))
            sp("signals_found", str(self.signals_found))
            sp("active_signals", str(len(signals)))
            logger.debug(f"📊 Dashboard updated | Total scans: {self.total_scans} | Signals: {len(signals)}")
        except Exception as e:
            logger.error(f"Failed to update dashboard stats: {e}")
    
    def get_status(self) -> Dict:
        """Get current scanner status for API/dashboard."""
        return {
            "status": "running" if self.engine.running else "stopped",
            "last_scan": self.last_scan,
            "scan_interval_s": self.scan_interval_s,
            "total_scans": self.total_scans,
            "signals_found": self.signals_found,
            "symbols_watched": len(self.symbols),
            "seconds_since_last_scan": int(time.time()) - self.last_scan if self.last_scan else None
        }


# Export for server.py
__all__ = ["MarketScanner"]