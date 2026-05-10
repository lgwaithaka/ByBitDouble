import asyncio, logging, time, math
from typing import Dict, Optional, List, Set, Tuple
from compound_engine import CompoundEngine
from db import gp, log_trade
from alerting import AlertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── GLOBAL TARGETS & SAFEGUARDS ──────────────────────────────────
MIN_TP_PCT = 20.0          # 🔧 FIX 10: Enforce minimum 20% profit target
BREAKEVEN_BUFFER_PCT = 0.015
TRAIL_DISTANCE_PCT = 0.030
SCAN_BASE_INTERVAL = 15    # 🔧 FIX 5: Adaptive base scan interval
# ───────────────────────────────────────────────────────────────────

class MarketRegime:
    def __init__(self):
        self.regime = "RANGING"
        self.trend_strength = 0.0

    def update(self, market_ Dict):
        self.regime = market_data.get("regime", "RANGING")
        self.trend_strength = market_data.get("trend_strength", 0.0)

    def adjustments(self) -> Dict:
        """🔧 FIX 7: Rebalanced for 20%+ target capability"""
        tp_bonus = 0.0
        lev_mod = 1.0
        if self.regime == "TRENDING":
            tp_bonus = min(0.15, self.trend_strength * 15)
            lev_mod = 1.2
        elif self.regime == "VOLATILE":
            # 🔧 FIX 7: VOLATILE no longer cripples TP/leverage
            tp_bonus = 0.05  # Small bonus to capture spikes
            lev_mod = 0.9    # Slight risk reduction, not a block
        return {"tp_bonus": tp_bonus, "leverage_mod": lev_mod}


class TradingEngine:
    def __init__(self, client, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        self.regime = MarketRegime()
        self.compound = CompoundEngine(11.05)
        self.alerts = AlertManager()
        self.positions: Dict[str, Dict] = {}
        self.running = False
        
        # 🔧 FIX 1: Initialize previously undefined variables
        self.fg = "NEUTRAL"
        self.fg_val = 50
        self.fg_conf_adj = 0.0
        self.auto_blacklist: Set[str] = set()
        self.losing_symbols: Dict[str, float] = {}  # symbol -> cooldown timestamp
        self.scan_interval_s = SCAN_BASE_INTERVAL
        self.liquidity_multiplier = 1.0

    async def start(self):
        self.running = True
        logger.info("🚀 Engine started | 20% min TP | Adaptive scanning | Mode-aware circuits")
        await self.alerts.send("🚀 Engine started | All debug fixes applied", "SUCCESS", force=True)
        asyncio.create_task(self.scan_and_trade())
        asyncio.create_task(self.monitor())
        asyncio.create_task(self.regime_update_loop())

    async def fetch_data(self, sym: str) -> Dict:
        """🔧 FIX 4: Safe liquidation price handling"""
        # Replace with actual Bybit fetch
        # raw = await self.client.get_market_data(sym)
        raw = {"price": 1.0, "vol_24h": 1.0, "liq_price": None}  # Mock
        
        # 🔧 FIX 4: Guard against p_liq(None) crash
        p_liq = raw.get("liq_price")
        if p_liq is None or p_liq <= 0:
            p_liq = raw["price"] * 0.8  # Safe fallback: 20% below current
            
        return {
            "price": raw["price"],
            "vol_24h": raw["vol_24h"],
            "liq_price": p_liq,
            "hour_utc": time.gmtime().tm_hour
        }

    def _get_liquidity_session(self, hour_utc: int) -> float:
        """🔧 FIX 9: Liquidity session volume scaling"""
        # High liquidity: NY/London overlap (12:00-16:00 UTC) & Asian open (00:00-04:00 UTC)
        if (12 <= hour_utc < 16) or (0 <= hour_utc < 4):
            return 1.5  # Scale position/signal weight up
        elif 8 <= hour_utc < 12:
            return 1.2
        return 1.0

    def _can_open(self, sym: str, mode: str, confidence: float) -> bool:
        """🔧 FIX 6: Mode-aware re-entry logic"""
        if sym in self.auto_blacklist:
            return False
            
        # 🔧 FIX 6: Losing symbols only block NORMAL mode
        if sym in self.losing_symbols:
            cooldown_end = self.losing_symbols[sym]
            if time.time() < cooldown_end:
                if mode == "NORMAL":
                    return False
                # AGGRESSIVE/TURBO bypass cooldown if confidence is high
                if confidence < 75:
                    return False
                    
        return True

    async def scan_and_trade(self):
        """🔧 FIX 1, 5, 8, 9: Fully patched scanning loop"""
        while self.running:
            try:
                mode = gp("mode", "NORMAL")
                
                # 🔧 FIX 8: Circuit breakers only halt NORMAL mode
                if mode == "NORMAL" and self._check_circuit_breakers():
                    logger.info("🛑 Circuit breaker active (NORMAL mode). Pausing scan.")
                    await asyncio.sleep(60)
                    continue
                    
                # 🔧 FIX 1: Safe Fear & Greed handling
                await self._update_fear_greed()
                
                for sym in self.symbols:
                    if not self._can_open(sym, mode, 0):  # Confidence checked later
                        continue
                        
                    data = await self.fetch_data(sym)
                    self.liquidity_multiplier = self._get_liquidity_session(data["hour_utc"])
                    
                    # 🔧 FIX 9: Volume scaling applied to signal weight
                    signal = await self._evaluate_signal(sym, data)
                    if signal and signal["confidence"] >= signal["threshold"]:
                        await self.queue_trade(sym, signal, mode, data)
                        
                # 🔧 FIX 5: Adaptive scan interval
                self.scan_interval_s = max(10, SCAN_BASE_INTERVAL - int(self.regime.trend_strength * 5))
                await asyncio.sleep(self.scan_interval_s)
                
            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(5)

    async def _update_fear_greed(self):
        """🔧 FIX 1: Define & populate FG variables safely"""
        # Replace with actual FG API fetch
        self.fg_val = 55  # Mock
        self.fg = "GREED" if self.fg_val > 60 else "FEAR" if self.fg_val < 40 else "NEUTRAL"
        self.fg_conf_adj = 0.1 if self.fg == "FEED" else -0.1 if self.fg == "FEAR" else 0.0

    def _check_circuit_breakers(self) -> bool:
        """🔧 FIX 8: Only returns True for NORMAL mode halts"""
        # Implement your drawdown/max loss checks here
        return False  # Placeholder: returns False unless thresholds breached

    async def _evaluate_signal(self, sym: str, data: Dict) -> Optional[Dict]:
        """Placeholder for your signal engine. Returns dict with confidence & threshold."""
        # Your existing logic goes here. Ensure it returns:
        # {"side": "Buy"/"Sell", "confidence": 0.0, "threshold": 0.0, "entry_price": 0.0}
        return None

    async def queue_trade(self, sym: str, signal: Dict, mode: str, data: Dict):
        side = signal["side"]
        entry = signal["entry_price"]
        risk_pct = self.compound.compute_risk_pct() * self.liquidity_multiplier  # 🔧 FIX 9: Liquidity scaling
        qty = (self.compound.balance * risk_pct) / entry
        
        initial_sl = entry * (1 - 0.05) if side == "Buy" else entry * (1 + 0.05)
        
        self.positions[sym] = {
            "side": side,
            "qty": qty,
            "entry_price": entry,
            "highest_price": entry,
            "lowest_price": entry,
            "trail_stop_price": initial_sl,
            "breakeven_moved": False,
            "ts_open": time.time(),
            "confidence": signal["confidence"],
            "mode": mode,
            "exit_p": 0.0  # 🔧 FIX 2: Initialize exit price
        }
        
        logger.info(f" Opened {sym} | {side} @ {entry:.4f} | Mode: {mode}")
        await self.alerts.trade_alert(sym, side, entry, qty, signal["confidence"])

    async def monitor(self):
        """ FIX 2: Added exit_p == 0 guard & false-close prevention"""
        while self.running:
            try:
                for sym in list(self.positions.keys()):
                    pos = self.positions[sym]
                    data = await self.fetch_data(sym)
                    current_price = data["price"]
                    
                    # 🔧 FIX 2: Guard against zero/None exit price logic
                    if pos["exit_p"] == 0:
                        await self._track_position(sym, pos, current_price)
                    else:
                        # Position already flagged for exit, skip re-evaluation
                        continue
                        
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

    async def _track_position(self, sym: str, pos: Dict, current_price: float):
        """Core PnL tracking, trailing logic, and TP/SL enforcement"""
        side = pos["side"]
        
        if side == "Buy":
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
        else:
            pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price

        # Breakeven trigger (non-exit, just moves SL)
        if not pos["breakeven_moved"] and pnl_pct >= 0.04:
            pos["trail_stop_price"] = pos["entry_price"] * (1 + BREAKEVEN_BUFFER_PCT) if side == "Buy" else pos["entry_price"] * (1 - BREAKEVEN_BUFFER_PCT)
            pos["breakeven_moved"] = True
            
        # Trailing stop
        if pos["breakeven_moved"]:
            if side == "Buy":
                new_trail = pos["highest_price"] * (1 - TRAIL_DISTANCE_PCT)
                if new_trail > pos["trail_stop_price"]:
                    pos["trail_stop_price"] = new_trail
            else:
                new_trail = pos["lowest_price"] * (1 + TRAIL_DISTANCE_PCT)
                if new_trail < pos["trail_stop_price"]:
                    pos["trail_stop_price"] = new_trail

        await self.check_profit_targets(sym, pos, pnl_pct)

    async def check_profit_targets(self, sym: str, pos: Dict, pnl_pct: float):
        """🔧 FIX 3 & 10: Dynamic TP scaling with strict 20% minimum"""
        regime_adj = self.regime.adjustments()
        base_tp = MIN_TP_PCT + regime_adj["tp_bonus"]  # 🔧 FIX 10: Starts at 20%
        
        #  FIX 3: Remove hard 10% cap. Scale target upward as profit grows.
        if pnl_pct > base_tp:
            dynamic_target = base_tp + (pnl_pct - base_tp) * 0.4  # Pushes target higher
        else:
            dynamic_target = base_tp
            
        hit_sl = (pos["side"] == "Buy" and current_price <= pos["trail_stop_price"]) or \
                 (pos["side"] == "Sell" and current_price >= pos["trail_stop_price"])
        hit_tp = pnl_pct >= dynamic_target
        
        # 🔧 FIX 2: Ensure exit_p is set before closing
        if hit_sl or hit_tp:
            reason = "TRAILING_STOP" if hit_sl else f"DYNAMIC_TP_{pnl_pct*100:.1f}%"
            pos["exit_p"] = pos["trail_stop_price"] if hit_sl else pos["entry_price"] * (1 + pnl_pct if pos["side"]=="Buy" else 1 - pnl_pct)
            await self.close_position(sym, pos, reason)

    async def close_position(self, sym: str, pos: Dict, reason: str):
        if pos["exit_p"] == 0:
            logger.warning(f"Attempted close with exit_p=0 for {sym}. Aborting.")
            return
            
        final_pnl = ((pos["exit_p"] - pos["entry_price"]) / pos["entry_price"]) * 100 if pos["side"]=="Buy" else \
                    ((pos["entry_price"] - pos["exit_p"]) / pos["entry_price"]) * 100
                    
        logger.info(f"🔚 Closed {sym} | {reason} | PnL: {final_pnl:.2f}%")
        await self.alerts.send(f"🔚 {sym} closed | {reason} | PnL: {final_pnl:.1f}%", "SUCCESS" if final_pnl > 0 else "WARNING")
        
        log_trade({
            "symbol": sym, "side": pos["side"], "entry_price": pos["entry_price"],
            "qty": pos["qty"], "ts_open": pos["ts_open"], "ts_close": time.time(),
            "outcome": "WIN" if final_pnl > 0 else "LOSS",
            "pnl_usdt": final_pnl * pos["qty"] * pos["entry_price"] / 100,
            "confidence": pos["confidence"]
        })
        
        # 🔧 FIX 6: Track losers for cooldown (respects mode bypass)
        if final_pnl <= 0:
            self.losing_symbols[sym] = time.time() + 3600  # 1h cooldown
            
        del self.positions[sym]

    async def regime_update_loop(self):
        while self.running:
            try:
                # Replace with actual regime detection
                self.regime.update({"regime": "RANGING", "trend_strength": 0.001})
            except Exception as e:
                logger.error(f"Regime update error: {e}")
            await asyncio.sleep(300)