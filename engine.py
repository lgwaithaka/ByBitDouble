"""
engine.py — Perpetual Compounding Trading Engine
Doubles every 5-day epoch, indefinitely. Adapts aggression to epoch progress.
"""
import asyncio, json, os, time, logging, smtplib, threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set
import numpy as np

from bybit_client   import BybitClient
from scanner        import VScanner, parse_klines, atr, p_funding, p_ls, p_oi, p_ob, p_liq, price_precision
from signals        import SignalEngine, sl_tp
from compound_engine import CompoundEngine, DAILY_REQUIRED_PCT, EPOCH_DAYS
from db import (
    init_db, gp, sp, open_trade, close_trade, update_pos_price,
    get_open_positions, get_trades, all_time_stats, log_capital,
    get_symbol_stats, snap_hour, close_epoch_record, open_epoch_record,
    get_all_epochs, get_hourly_snaps, all_time_stats
)

logger = logging.getLogger(__name__)
REPORT_EMAIL = os.getenv("REPORT_EMAIL", "lgwaithaka@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")

# Target multiplier — 3.0 = 200% gain (triple), 2.0 = 100% gain (double)
TARGET_MULTIPLIER = float(os.getenv("TARGET_MULTIPLIER", "3.0"))


# ── Market Feed: Fear & Greed Index ──────────────────────────────────────────

class MarketFeed:
    """Aggregates external market sentiment data from free APIs."""

    def __init__(self):
        self.fear_greed_value = 50       # 0-100, 50 = neutral
        self.fear_greed_label = "Neutral"
        self.last_updated = None
        self._session = None

    async def _get_session(self):
        import aiohttp
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def update(self):
        """Fetch latest Fear & Greed Index from Alternative.me (free, no key)."""
        try:
            sess = await self._get_session()
            async with sess.get("https://api.alternative.me/fng/?limit=1") as r:
                if r.status == 200:
                    data = await r.json()
                    item = data.get("data", [{}])[0]
                    self.fear_greed_value = int(item.get("value", 50))
                    self.fear_greed_label = item.get("value_classification", "Neutral")
                    self.last_updated = datetime.now(timezone.utc).isoformat()
                    logger.info(f"Fear & Greed: {self.fear_greed_value} ({self.fear_greed_label})")
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")

    def confidence_adjustment(self, signal: str) -> int:
        """
        Adjust confidence based on sentiment.
        Extreme Greed (>75): penalize LONG (+10 conf needed), boost SHORT (-5 conf needed)
        Extreme Fear (<25): penalize SHORT (+10 conf needed), boost LONG (-5 conf needed)
        """
        fg = self.fear_greed_value
        if fg >= 75:  # Extreme Greed — crowd is long, contrarian SHORT edge
            return 10 if signal == "LONG" else -5
        elif fg <= 25:  # Extreme Fear — crowd is short, contrarian LONG edge
            return -5 if signal == "LONG" else 10
        elif fg >= 60:  # Greed — mild LONG penalty
            return 3 if signal == "LONG" else -2
        elif fg <= 40:  # Fear — mild SHORT penalty
            return -2 if signal == "LONG" else 3
        return 0

    def to_dict(self) -> Dict:
        return {
            "fear_greed_value": self.fear_greed_value,
            "fear_greed_label": self.fear_greed_label,
            "last_updated": self.last_updated,
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ── Asset Classification ─────────────────────────────────────────────────────

ASSET_CLASSES = {
    "BTCUSDT":"BTC","ETHUSDT":"ETH",
    "SOLUSDT":"L1","AVAXUSDT":"L1","SUIUSDT":"L1","APTUSDT":"L1",
    "NEARUSDT":"L1","DOTUSDT":"L1","ATOMUSDT":"L1","ADAUSDT":"L1",
    "DOGEUSDT":"MEME","SHIBUSDT":"MEME","PEPEUSDT":"MEME","FLOKIUSDT":"MEME",
    "WIFUSDT":"MEME","BONKUSDT":"MEME",
    "LINKUSDT":"DEFI","AAVEUSDT":"DEFI","UNIUSDT":"DEFI","MKRUSDT":"DEFI",
    "ARBUSDT":"L2","OPUSDT":"L2","MATICUSDT":"L2",
    "BNBUSDT":"EXCHANGE","FTMUSDT":"L1","INJUSDT":"L1",
}

def get_asset_class(symbol: str) -> str:
    return ASSET_CLASSES.get(symbol, "OTHER")


# ── Whale Tracker ─────────────────────────────────────────────────────────────

class WhaleTracker:
    """Tracks smart money signals: OI trends, funding extremes, liquidation cascades."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self.last_summary: Dict = {
            "market_bias": "UNKNOWN",
            "short_squeezes": [],
            "funding_extremes": [],
            "oi_surges": [],
        }

    def get_all_cache(self) -> Dict:
        return self._cache

    def get_top_opportunities(self, n: int = 5) -> List[Dict]:
        scored = []
        for sym, data in self._cache.items():
            score = abs(data.get("funding_signal", 0)) + abs(data.get("oi_change_pct", 0)) / 10
            if data.get("liq_cascade"):
                score += 1.0
            scored.append({"symbol": sym, "score": round(score, 3), **data})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n]

    async def update_all(self, symbols: List[str], client: BybitClient) -> Dict:
        results = {}
        for sym in symbols:
            try:
                data = await self._analyze_symbol(sym, client)
                self._cache[sym] = data
                results[sym] = data
            except Exception as e:
                logger.error(f"Whale update {sym}: {e}")
        self._update_summary()
        sp("whale_last_update", str(int(time.time())))
        return results

    async def _analyze_symbol(self, sym: str, client: BybitClient) -> Dict:
        fr_resp = await client.funding(sym)
        oi_resp = await client.open_interest(sym, "1h")
        liq_resp = await client.liquidations(sym)

        funding_rate = 0.0
        fr_list = fr_resp.get("result", {}).get("list", [])
        if fr_list:
            funding_rate = float(fr_list[0].get("fundingRate", "0") or "0")

        oi_change = 0.0
        oi_list = oi_resp.get("result", {}).get("list", [])
        if len(oi_list) >= 2:
            oi_now = float(oi_list[0].get("openInterest", "0") or "0")
            oi_prev = float(oi_list[1].get("openInterest", "0") or "0")
            if oi_prev > 0:
                oi_change = (oi_now - oi_prev) / oi_prev * 100

        liq_data = p_liq(liq_resp)
        long_liq = liq_data.get("long_liq", 0)
        short_liq = liq_data.get("short_liq", 0)

        funding_signal = 0.0
        if funding_rate > 0.001: funding_signal = -1.0
        elif funding_rate > 0.0003: funding_signal = -0.5
        elif funding_rate < -0.001: funding_signal = 1.0
        elif funding_rate < -0.0003: funding_signal = 0.5

        return {
            "funding_rate": round(funding_rate, 6),
            "funding_signal": funding_signal,
            "oi_change_pct": round(oi_change, 2),
            "long_liq": long_liq,
            "short_liq": short_liq,
            "liq_cascade": (long_liq + short_liq) > 1000000,
            "bias": "LONG" if funding_signal > 0 else ("SHORT" if funding_signal < 0 else "NEUTRAL"),
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    def _update_summary(self):
        if not self._cache:
            return
        long_bias = sum(1 for d in self._cache.values() if d.get("bias") == "LONG")
        short_bias = sum(1 for d in self._cache.values() if d.get("bias") == "SHORT")
        total = len(self._cache)

        if long_bias > total * 0.6: market_bias = "BULLISH"
        elif short_bias > total * 0.6: market_bias = "BEARISH"
        else: market_bias = "MIXED"

        self.last_summary = {
            "market_bias": market_bias,
            "symbols_tracked": total,
            "long_bias_count": long_bias,
            "short_bias_count": short_bias,
            "short_squeezes": [s for s, d in self._cache.items()
                               if d.get("funding_signal", 0) > 0.5 and d.get("oi_change_pct", 0) > 3],
            "funding_extremes": [s for s, d in self._cache.items()
                                 if abs(d.get("funding_rate", 0)) > 0.001],
            "oi_surges": [s for s, d in self._cache.items()
                          if abs(d.get("oi_change_pct", 0)) > 5],
        }


# ── Regime Detector ───────────────────────────────────────────────────────────

class RegimeDetector:
    """Detects market regime: TRENDING, RANGING, or VOLATILE."""

    def __init__(self):
        self.regime = "UNKNOWN"
        self.trend_strength = 0.0
        self.volatility_pct = 0.0
        self.last_updated = None

    async def update(self, client: BybitClient):
        try:
            resp = await client.klines("BTCUSDT", "60", 50)
            k = parse_klines(resp)
            if k is None or len(k) < 30:
                return
            closes = k[:, 4]
            highs = k[:, 2]
            lows = k[:, 3]

            # ADX for trend strength
            from signals import calc_adx
            adx = calc_adx(k, 14)
            self.trend_strength = adx / 100.0

            # Volatility: ATR as % of price
            atr_val = atr(k, 14)
            price = float(closes[-1])
            self.volatility_pct = (atr_val / price * 100) if price > 0 else 0

            # Classify
            if adx > 30 and self.volatility_pct < 3:
                self.regime = "TRENDING"
            elif adx < 20:
                self.regime = "RANGING"
            elif self.volatility_pct > 3:
                self.regime = "VOLATILE"
            else:
                self.regime = "TRENDING"

            self.last_updated = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error(f"Regime update: {e}")

    def adjustments(self) -> Dict:
        if self.regime == "TRENDING":
            return {"label": "Strong trend — ride momentum", "conf_bonus": 5,
                    "sl_mult": 1.0, "tp_mult": 1.2, "size_mult": 1.0}
        elif self.regime == "RANGING":
            return {"label": "Range-bound — mean reversion", "conf_bonus": -5,
                    "sl_mult": 0.8, "tp_mult": 0.8, "size_mult": 0.7}
        elif self.regime == "VOLATILE":
            return {"label": "High volatility — tight risk", "conf_bonus": -10,
                    "sl_mult": 1.5, "tp_mult": 1.5, "size_mult": 0.5}
        return {"label": "Unknown regime", "conf_bonus": 0,
                "sl_mult": 1.0, "tp_mult": 1.0, "size_mult": 1.0}


class CompoundTradingEngine:
    def __init__(self):
        self.client:  Optional[BybitClient]     = None
        self.scanner: Optional[VScanner]        = None
        self.signals  = SignalEngine()
        self.compound = CompoundEngine()
        self.whales   = WhaleTracker()
        self.regime   = RegimeDetector()
        self.market_feed = MarketFeed()

        self.running       = False
        self.status        = "IDLE"
        self.balance       = 0.0
        self.live_positions: Dict[str, Dict]    = {}
        self.cooldown_until: Dict[str, float]   = {}   # symbol → ts
        self.losing_symbols: Set[str]           = set()
        self.capital_auto_detected              = False

        # ── Daily 20% Target System ──
        self.daily_start_balance = 0.0
        self.daily_target        = 0.0
        self.daily_target_pct    = float(os.getenv("DAILY_TARGET_PCT", "20"))  # 20% daily
        self.daily_reset_hour    = 0  # midnight UTC

        self.scan_results: List[Dict]           = []
        self.last_scan_ts  = 0
        self.last_hour_ts  = 0
        self.last_daily_ts = 0
        self.errors: List[str]                  = []

    def init(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client  = BybitClient(api_key, api_secret, testnet)
        self.scanner = VScanner(self.client)
        init_db()

        # Restore epoch state from DB
        epoch_num  = int(gp("current_epoch", "1"))
        epoch_ts   = int(gp("epoch_start_ts", str(int(time.time()))))
        epoch_bal  = float(gp("epoch_start_bal", "100.0"))
        initial    = float(gp("initial_capital", "100.0"))

        self.compound.initialise(
            start_balance   = initial,
            epoch_num       = epoch_num,
            epoch_start_ts  = epoch_ts,
            epoch_start_bal = epoch_bal,
        )
        # Override target to 200% (3x) instead of 100% (2x)
        self.compound.state.epoch_target = epoch_bal * TARGET_MULTIPLIER
        logger.info(f"Engine init | Epoch {epoch_num} | start=${epoch_bal:.2f} | target=${epoch_bal * TARGET_MULTIPLIER:.2f} ({int((TARGET_MULTIPLIER-1)*100)}% gain)")

    # ── Balance ──────────────────────────────────────────────────────────────

    async def refresh_balance(self) -> float:
        try:
            resp = await self.client.wallet()
            for acc in resp.get("result", {}).get("list", []):
                b = acc.get("totalWalletBalance")
                if b:
                    self.balance = float(b); return self.balance
        except Exception as e:
            logger.error(f"refresh_balance: {e}")
        return self.balance

    async def refresh_live_positions(self):
        try:
            resp = await self.client.positions()
            self.live_positions = {
                p["symbol"]: p
                for p in resp.get("result", {}).get("list", [])
                if float(p.get("size", "0")) > 0
            }
        except Exception as e:
            logger.error(f"refresh_live_positions: {e}")

    # ── Daily 20% Target ────────────────────────────────────────────────────

    def reset_daily_target(self):
        """Reset daily target to 20% of current balance. Called at midnight UTC."""
        self.daily_start_balance = self.balance
        self.daily_target = self.balance * (1 + self.daily_target_pct / 100)
        sp("daily_start_bal", str(round(self.daily_start_balance, 4)))
        sp("daily_target", str(round(self.daily_target, 4)))
        # Also override the compound engine target
        self.compound.state.epoch_target = self.daily_target
        logger.info(f"DAILY TARGET RESET | start=${self.daily_start_balance:.2f} | target=${self.daily_target:.2f} (+{self.daily_target_pct}%)")

    def daily_progress(self) -> Dict:
        """Get daily target progress."""
        if self.daily_start_balance <= 0:
            return {"progress_pct": 0, "ahead_pct": 0, "on_track": False}
        gained = self.balance - self.daily_start_balance
        needed = self.daily_target - self.daily_start_balance
        progress = (gained / needed * 100) if needed > 0 else 0
        return {
            "daily_start": round(self.daily_start_balance, 4),
            "daily_target": round(self.daily_target, 4),
            "daily_target_pct": self.daily_target_pct,
            "current_balance": round(self.balance, 4),
            "gained": round(gained, 4),
            "needed": round(needed, 4),
            "progress_pct": round(progress, 1),
            "on_track": self.balance >= self.daily_start_balance,
        }

    # ── Keep-Alive (prevents Render free tier spin-down) ──────────────────

    async def _keep_alive(self):
        """Self-ping every 10 minutes to prevent Render free tier spin-down."""
        import aiohttp
        service_url = os.getenv("RENDER_EXTERNAL_URL", "")
        if not service_url:
            logger.info("KEEP-ALIVE: No RENDER_EXTERNAL_URL set, skipping self-ping")
            return
        logger.info(f"KEEP-ALIVE: Pinging {service_url} every 10 minutes")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
            while self.running:
                try:
                    async with sess.get(f"{service_url}/") as r:
                        logger.debug(f"Keep-alive ping: {r.status}")
                except Exception:
                    pass
                await asyncio.sleep(600)  # 10 minutes

    # ── Market Data ──────────────────────────────────────────────────────────

    async def fetch_data(self, symbol: str) -> Optional[Dict]:
        try:
            results = await asyncio.gather(
                self.client.klines(symbol, "3",  80),
                self.client.klines(symbol, "5",  130),
                self.client.klines(symbol, "15", 60),
                self.client.klines(symbol, "60", 40),
                self.client.klines(symbol, "240", 60),
                self.client.funding(symbol),
                self.client.ls_ratio(symbol),
                self.client.open_interest(symbol, "1h"),
                self.client.orderbook(symbol, 50),
                self.client.liquidations(symbol),
                return_exceptions=True
            )
            k3, k5, k15, k1h, k4h, fr, lsr, oi, ob, liq = results
            return {
                "k3m":  parse_klines(k3)   if not isinstance(k3,  Exception) else None,
                "k5m":  parse_klines(k5)   if not isinstance(k5,  Exception) else None,
                "k15m": parse_klines(k15)  if not isinstance(k15, Exception) else None,
                "k1h":  parse_klines(k1h)  if not isinstance(k1h, Exception) else None,
                "k4h":  parse_klines(k4h)  if not isinstance(k4h, Exception) else None,
                "funding":  p_funding(fr)  if not isinstance(fr,  Exception) else 0.0,
                "ls":       p_ls(lsr)      if not isinstance(lsr, Exception) else 1.0,
                "oi_pct":   p_oi(oi)       if not isinstance(oi,  Exception) else 0.0,
                "ob_imb":   p_ob(ob)       if not isinstance(ob,  Exception) else 1.0,
                "liqs":     p_liq(liq)     if not isinstance(liq, Exception) else {},
            }
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}"); return None

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(self, symbol: str, analysis: Dict, meta: Dict) -> bool:
        ce     = self.compound
        signal = analysis["signal"]
        conf   = analysis["confidence"]
        comp   = analysis["composite"]
        mode   = ce.state.mode

        # Fetch fresh price + ATR
        raw5  = await self.client.klines(symbol, "5", 50)
        k5    = parse_klines(raw5)
        if k5 is None or len(k5) == 0: return False

        price    = float(k5[-1, 4])
        atr_val  = atr(k5, 14)

        sym_stats= get_symbol_stats(symbol)
        sl_mult, tp_mult = ce.compute_sl_tp_mults(
            sym_stats.get("sl_mult", 1.2),
            sym_stats.get("tp_mult", 2.4)
        )
        leverage = ce.compute_leverage(meta.get("range_pct", 5.0))

        # Position size — micro-account safe sizing
        risk_pct = ce.compute_risk_pct()
        risk_usd = self.balance * risk_pct
        sl_dist  = atr_val * sl_mult if atr_val > 0 else price * 0.009
        qty      = (risk_usd * leverage) / price
        qty      = max(qty, 0.001)

        # Bybit minimum notional check ($5.5 USDT minimum order value)
        min_notional = float(gp("min_notional_usdt", "5.5"))
        if qty * price < min_notional:
            # Bump qty up to meet minimum — use leverage to keep margin small
            qty = (min_notional * 1.05) / price   # 5% buffer over minimum

        # Cap single-trade notional at 40% of balance * leverage
        # For $10: max notional = $10 * 0.40 * 15x = $60 (margin used = $4)
        max_notional = self.balance * 0.40 * leverage
        if qty * price > max_notional:
            qty = max_notional / price

        # Round to correct precision for the asset price
        if price >= 1000:
            qty = round(qty, 3)
        elif price >= 1:
            qty = round(qty, 2)
        else:
            qty = round(qty, 0)  # e.g. SHIB, very small price = whole units
            qty = max(qty, 1)

        # Final sanity: if balance too low to meet minimum, skip trade
        required_margin = (min_notional) / leverage
        if self.balance < required_margin * 1.5:
            logger.warning(f"Balance ${self.balance:.2f} too low for {symbol} minimum order. Skipping.")
            return False

        side = "Buy" if signal == "LONG" else "Sell"
        sl, tp = sl_tp(side, price, atr_val, sl_mult=sl_mult, tp_mult=tp_mult)

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception: pass

        resp = await self.client.place_order(symbol, side, qty, sl=sl, tp=tp)
        if resp.get("retCode", -1) != 0:
            err = resp.get("retMsg", "?")
            logger.error(f"Order fail {symbol}: {err}")
            self.errors.append(f"{datetime.now().isoformat()} | {symbol} | {err}")
            return False

        order_id = resp.get("result", {}).get("orderId", "")
        epoch    = ce.state.epoch_num

        tid = open_trade(
            epoch=epoch, symbol=symbol, side=side, signal=signal,
            confidence=conf, composite=comp,
            entry_price=price, qty=qty, leverage=leverage,
            sl=sl, tp=tp, order_id=order_id,
            tag=f"EP{epoch}_L{leverage}_C{conf}_{mode}",
            mode=mode, components=analysis.get("components", {}),
            vol_score=meta.get("vol_score", 0),
            range_pct=meta.get("range_pct", 0),
        )
        logger.info(
            f"OPEN [{mode}] {symbol} {signal} | qty={qty} @ ${price:.4f} "
            f"| SL={sl:.4f} TP={tp:.4f} | L={leverage}x | ep={epoch} | conf={conf}%"
        )
        return True

    # ── Position Monitor ─────────────────────────────────────────────────────

    async def monitor(self):
        await self.refresh_live_positions()
        db_open = get_open_positions()
        initial = float(gp("initial_capital", "100.0"))
        epoch   = self.compound.state.epoch_num
        min_profit_pct = float(gp("min_profit_pct", "10.0"))  # Hold until 10% profit

        for pos in db_open:
            sym = pos["symbol"]
            tid = pos["trade_id"]

            if sym in self.live_positions:
                live = self.live_positions[sym]
                try:
                    unr  = float(live.get("unrealisedPnl", "0") or "0")
                    cprc = float(live.get("markPrice",     "0") or "0")
                    size = float(live.get("size",          "0") or "0")
                    entry= float(live.get("avgPrice",      "0") or "0")
                    side = live.get("side", "")
                    update_pos_price(sym, cprc, unr)

                    # ── SMART TP: Hold until min 10% profit ──
                    if entry > 0 and size > 0:
                        notional = entry * size
                        pnl_pct = (unr / notional) * 100 if notional > 0 else 0

                        # If profitable but below 10%, remove existing TP to let it run
                        if 0 < pnl_pct < min_profit_pct:
                            logger.debug(f"HOLD {sym}: +{pnl_pct:.1f}% < {min_profit_pct}% target, letting it run")

                        # If above 10% profit, set a trailing stop to lock gains
                        elif pnl_pct >= min_profit_pct:
                            atr_val = 0
                            try:
                                k5 = await self.client.klines(sym, "5", 20)
                                k5_data = parse_klines(k5)
                                if k5_data is not None:
                                    atr_val = atr(k5_data, 14)
                            except: pass

                            if atr_val > 0:
                                # Set trailing stop at 40% of current profit distance
                                trail = max(atr_val * 1.5, abs(cprc - entry) * 0.4)
                                try:
                                    await self.client.set_tpsl(sym, trailing=round(trail, price_precision(cprc)))
                                    logger.info(f"TRAILING STOP {sym}: +{pnl_pct:.1f}% profit, trail={trail:.4f}")
                                except Exception as e:
                                    logger.debug(f"Trail set error {sym}: {e}")

                except Exception as e:
                    logger.error(f"monitor {sym}: {e}")
            else:
                # Closed by SL/TP — fetch result
                pnl = 0.0; exit_p = 0.0
                try:
                    pr   = await self.client.closed_pnl(sym, 10)
                    pls  = pr.get("result", {}).get("list", [])
                    if pls:
                        pnl    = float(pls[0].get("closedPnl",    "0") or "0")
                        exit_p = float(pls[0].get("avgExitPrice", "0") or "0")
                except Exception: pass

                outcome = close_trade(tid, exit_p, pnl, initial)
                win     = outcome == "WIN"

                # Feed compound engine
                self.compound.record_outcome(win, pnl, self.balance)

                # Cooldown on loss — adaptive based on loss severity
                if not win:
                    sym_stats = get_symbol_stats(sym)
                    consecutive = sym_stats.get("losses", 0) - sym_stats.get("wins", 0)
                    cd = min(600, 120 + (max(0, consecutive) * 60))
                    self.cooldown_until[sym] = time.time() + cd
                    self.losing_symbols.add(sym)
                    logger.info(f"Loss cooldown {sym}: {cd}s (net losses: {consecutive})")
                else:
                    self.losing_symbols.discard(sym)

                logger.info(f"CLOSED | {sym} | {outcome} | PnL=${pnl:.4f}")

    # ── Scan & Trade ─────────────────────────────────────────────────────────

    async def scan_and_trade(self):
        ce = self.compound

        # ── FILTER 1: Override CONSERVATIVE mode (2W/20L = 9% WR) ──
        # Force NORMAL thresholds when engine wants CONSERVATIVE
        if ce.state.mode == "CONSERVATIVE":
            logger.info("Overriding CONSERVATIVE → NORMAL (CONSERVATIVE has 9% WR historically)")
            ce.state.mode = "NORMAL"

        # ── AGGRESSIVE MODE: Bypass circuit breakers ──
        if ce.state.mode == "AGGRESSIVE":
            # In aggressive mode, don't let circuit breakers stop trading
            logger.debug("AGGRESSIVE mode — circuit breakers bypassed")
        elif ce.check_circuit_breakers(
            self.balance,
            epoch_max_dd = float(gp("epoch_max_dd_pct", "30")) / 100,
            daily_max_dd = float(gp("daily_max_dd_pct", "20")) / 100,
        ):
            logger.warning(f"Circuit breaker active | mode={ce.state.mode}")
            return

        await self.refresh_balance()
        top_syms   = await self.scanner.scan(n=int(gp("vol_scan_n","14")))
        db_open    = {p["symbol"] for p in get_open_positions()}
        conf_floor = ce.compute_confidence_floor()
        max_conc   = ce.compute_max_concurrent()
        mode       = ce.state.mode

        # ── FILTER 3: Auto-blacklist symbols with poor win rates ──
        symbol_blacklist = set()
        for meta in top_syms:
            sym = meta["symbol"]
            ss = get_symbol_stats(sym)
            total = ss.get("wins", 0) + ss.get("losses", 0)
            if total >= 2 and ss.get("wins", 0) == 0:
                symbol_blacklist.add(sym)
                logger.info(f"Auto-blacklisted {sym}: 0 wins in {total} trades")

        results = []
        for meta in top_syms:
            sym = meta["symbol"]

            # Blacklist check
            if sym in symbol_blacklist:
                continue

            # Cooldown check
            if self.cooldown_until.get(sym, 0) > time.time(): continue

            # Losing symbol cooldown (longer cooldown for repeated losers)
            if sym in self.losing_symbols:
                if self.cooldown_until.get(sym, 0) > time.time():
                    continue

            data = await self.fetch_data(sym)
            if not data: continue

            sym_stats    = get_symbol_stats(sym)
            learned_bias = sym_stats.get("learned_bias", 0.0)

            analysis = self.signals.analyze(
                sym      = sym,
                k3m      = data["k3m"],
                k5m      = data["k5m"],
                k15m     = data["k15m"],
                k1h      = data["k1h"],
                k4h      = data["k4h"],
                funding  = data["funding"],
                ls       = data["ls"],
                oi_pct   = data["oi_pct"],
                ob_imb   = data["ob_imb"],
                liqs     = data["liqs"],
                learned_bias = learned_bias,
                mode     = mode,
            )

            signal = analysis["signal"]; conf = analysis["confidence"]

            # ── FILTER 4: Veto if lower timeframes disagree ──
            comps = analysis.get("components", {})
            if signal != "HOLD":
                macd5 = comps.get("macd_5m", 0)
                cross3 = comps.get("cross_3m", 0)
                st5 = comps.get("st_5m", 0)
                ob_score = comps.get("ob", 0)

                # Veto: macd_5m AND cross_3m both negative (33% of historical losses)
                if macd5 < 0 and cross3 < 0:
                    logger.info(f"VETO {sym}: macd_5m={macd5:.2f} AND cross_3m={cross3:.2f} both negative")
                    signal = "HOLD"
                    analysis["signal"] = "HOLD"
                    analysis["veto_reason"] = f"LTF_DISAGREE macd5={macd5:.2f} cross3={cross3:.2f}"

                # Veto: 5m SuperTrend disagrees with macro direction
                elif st5 < 0 and analysis.get("macro_direction", 0) != 0:
                    if ob_score <= 0.1:  # Only veto if orderbook also weak
                        logger.info(f"VETO {sym}: st_5m disagrees AND weak orderbook")
                        signal = "HOLD"
                        analysis["signal"] = "HOLD"
                        analysis["veto_reason"] = f"ST5M_DISAGREE_WEAK_OB st5={st5:.2f} ob={ob_score:.2f}"

            # ── FILTER 5: Fear & Greed sentiment gate ──
            if signal != "HOLD":
                fg_adj = self.market_feed.confidence_adjustment(signal)
                conf = max(0, conf - fg_adj)  # Raise bar at extremes
                analysis["confidence"] = conf
                if fg_adj != 0:
                    fg = self.market_feed.fear_greed_value
                    logger.debug(f"{sym} F&G={fg} ({self.market_feed.fear_greed_label}) adj={fg_adj} → conf={conf}")

            result = {
                "symbol": sym, "signal": signal, "confidence": conf,
                "composite": analysis["composite"],
                "veto_reason": analysis.get("veto_reason"),
                "macro_direction": analysis.get("macro_direction", 0),
                "leverage": ce.compute_leverage(meta.get("range_pct", 5)),
                "vol_score": meta.get("vol_score", 0),
                "range_pct": meta.get("range_pct", 0),
                "chg_24h":   meta.get("chg_24h", 0),
                "mode": mode, "ts": int(time.time()), "action": "HOLD"
            }

            already_open = sym in db_open
            can_open     = len(db_open) < max_conc and not already_open

            if signal != "HOLD" and conf >= conf_floor:
                if can_open:
                    ok = await self.execute(sym, analysis, meta)
                    if ok:
                        db_open.add(sym)
                        result["action"] = "OPENED"
                elif already_open:
                    result["action"] = "MONITORING"
                else:
                    result["action"] = "MAX_POSITIONS"
            results.append(result)
            await asyncio.sleep(0.25)

        self.scan_results = results
        self.last_scan_ts = int(time.time())

    # ── Epoch Boundary ───────────────────────────────────────────────────────

    async def check_epoch_boundary(self):
        """Check if epoch has elapsed and advance if so."""
        await self.refresh_balance()
        ce  = self.compound
        old = ce.state.epoch_num

        advanced = ce.advance_epoch(self.balance)
        if advanced:
            # Override target to TARGET_MULTIPLIER (200% = 3x)
            ce.state.epoch_target = ce.state.epoch_start_bal * TARGET_MULTIPLIER

            # Record completed epoch
            close_epoch_record(old, self.balance)
            new_epoch = ce.state.epoch_num

            # Persist new epoch state
            sp("current_epoch",   str(new_epoch))
            sp("epoch_start_ts",  str(ce.state.epoch_start_ts))
            sp("epoch_start_bal", str(round(ce.state.epoch_start_bal, 4)))

            # Create new epoch record
            open_epoch_record(new_epoch, ce.state.epoch_start_bal)

            await self.send_epoch_report(old, self.balance, ce.state.epoch_target)
            logger.info(f"New epoch {new_epoch} | stake=${ce.state.epoch_start_bal:.2f} | target=${ce.state.epoch_target:.2f}")

    # ── Hourly + Daily Tasks ─────────────────────────────────────────────────

    async def hourly_tasks(self):
        await self.refresh_balance()
        ce = self.compound
        s  = ce.state

        stats   = all_time_stats()
        h_trades= get_trades(hours=1)
        h_wins  = sum(1 for t in h_trades if t["outcome"] == "WIN")
        h_pnl   = sum(t["pnl_usdt"] or 0 for t in h_trades if t["outcome"] != "OPEN")

        snap_hour(s.epoch_num, self.balance, h_pnl,
                  len(h_trades), h_wins, stats["open_positions"], s.mode)
        log_capital(
            epoch        = s.epoch_num,
            day_in_epoch = s.day_in_epoch,
            balance      = self.balance,
            target_now   = ce.target_at_now(),
            target_eod   = ce.target_at_day_end(),
            ahead_pct    = ce.ahead_pct(self.balance),
            mode         = s.mode,
            open_pos     = stats["open_positions"],
        )
        await self.scanner.scan(force=True)

        # ── Hourly market intelligence updates ──
        await self.market_feed.update()
        await self.regime.update(self.client)
        top_syms = await self.scanner.scan(n=14)
        sym_list = [m["symbol"] for m in top_syms]
        await self.whales.update_all(sym_list, self.client)

        fg = self.market_feed.fear_greed_value
        logger.info(f"Hourly | ep={s.epoch_num} | bal=${self.balance:.2f} | mode={s.mode} | F&G={fg} | regime={self.regime.regime}")

    async def daily_tasks(self):
        await self.refresh_balance()
        # Reset daily 20% target based on current balance
        self.reset_daily_target()
        # Reset circuit breakers for the new day
        self.compound.advance_day(self.balance)
        self.compound.reset_daily_cb()
        # Clear losing symbols for fresh start
        self.losing_symbols.clear()
        self.cooldown_until.clear()
        # Send daily report email
        await self.send_daily_report()
        logger.info(f"DAILY RESET | balance=${self.balance:.2f} | new target=${self.daily_target:.2f} (+{self.daily_target_pct}%)")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def main_loop(self):
        self.running = True
        self.status  = "RUNNING"
        await self.refresh_balance()

        # Initialize daily 20% target from current balance
        saved_daily_start = float(gp("daily_start_bal", "0"))
        if saved_daily_start > 0:
            self.daily_start_balance = saved_daily_start
            self.daily_target = float(gp("daily_target", str(saved_daily_start * 1.2)))
        else:
            self.reset_daily_target()

        logger.info(f"CompoundEngine START | balance=${self.balance:.2f} | mode={self.compound.state.mode}")
        logger.info(f"Daily target: ${self.daily_target:.2f} (+{self.daily_target_pct}%) from ${self.daily_start_balance:.2f}")

        # Fetch market feeds on startup
        try:
            await self.market_feed.update()
            logger.info(f"Fear & Greed: {self.market_feed.fear_greed_value} ({self.market_feed.fear_greed_label})")
        except Exception as e:
            logger.warning(f"Market feed startup error (non-fatal): {e}")

        # Log email configuration status
        if SMTP_USER and SMTP_PASS:
            logger.info(f"Email reports ENABLED | to={REPORT_EMAIL} | from={SMTP_USER[:4]}*** | via={SMTP_HOST}:{SMTP_PORT}")
        else:
            logger.warning(f"Email reports DISABLED | SMTP_USER={'SET' if SMTP_USER else 'EMPTY'} | SMTP_PASS={'SET' if SMTP_PASS else 'EMPTY'}")

        # Start keep-alive background task (prevents Render free tier spin-down)
        asyncio.create_task(self._keep_alive())

        scan_secs      = int(gp("scan_interval_s", "40"))
        last_daily_hour= -1

        while self.running:
            t0 = time.time()
            try:
                await self.monitor()
                await self.check_epoch_boundary()
                await self.scan_and_trade()

                now = time.time()
                if now - self.last_hour_ts >= 3600:
                    await self.hourly_tasks()
                    self.last_hour_ts = now

                cur_hour = datetime.now(timezone.utc).hour
                if cur_hour == 0 and last_daily_hour != 0:
                    await self.daily_tasks()
                last_daily_hour = cur_hour

            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                self.errors.append(f"{datetime.now().isoformat()} | {e}")
                if len(self.errors) > 50:
                    self.errors = self.errors[-25:]  # Keep only last 25 errors
                await asyncio.sleep(5)

            await asyncio.sleep(max(0, scan_secs - (time.time() - t0)))

        self.status  = "STOPPED"; self.running = False
        if self.client: await self.client.close()
        await self.market_feed.close()

    async def stop(self):
        self.running = False

    # ── Reports ──────────────────────────────────────────────────────────────

    async def send_epoch_report(self, epoch_num: int, achieved: float, next_target: float):
        try:
            target = self.compound.state.epoch_start_bal  # before advance = old start
            pct    = (achieved - target) / target * 100 if target > 0 else 0
            body   = f"""
EPOCH {epoch_num} COMPLETE — BYBIT COMPOUND MCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch Start:   ${target:.2f} USDT
Achieved:      ${achieved:.2f} USDT
Target was:    ${target * TARGET_MULTIPLIER:.2f} USDT ({int((TARGET_MULTIPLIER-1)*100)}% gain)
Gain:          {pct:+.1f}%
Status:        {'✓ TARGET HIT' if achieved >= target * TARGET_MULTIPLIER else '✗ MISSED TARGET'}

Next Epoch {epoch_num + 1}:
  New Stake:   ${achieved:.2f} USDT
  New Target:  ${next_target:.2f} USDT
  Required:    {DAILY_REQUIRED_PCT:.2f}%/day for {EPOCH_DAYS} days

Projected Milestones (if targets met):
"""
            proj = self.compound.project_compounding(achieved, 8)
            for p in proj[:8]:
                body += f"  Epoch {p['epoch']}: ${p['target']:.2f}  (Day {p['days_elapsed']})\n"
            self._send_email(f"[Bybit Compound] Epoch {epoch_num} Done | ${achieved:.2f}", body)
        except Exception as e:
            logger.error(f"epoch report: {e}")

    async def send_daily_report(self):
        try:
            ce    = self.compound
            stats = all_time_stats()
            dp    = self.daily_progress()
            body  = f"""
BYBIT COMPOUND MCP — DAILY REPORT
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DAILY TARGET (+{self.daily_target_pct}%)
  Day Start:     ${dp.get('daily_start', 0):.2f}
  Day Target:    ${dp.get('daily_target', 0):.2f}
  Current:       ${self.balance:.2f}
  Progress:      {dp.get('progress_pct', 0):.1f}%
  Status:        {'✓ ON TRACK' if dp.get('on_track') else '✗ BEHIND'}
  Gained:        ${dp.get('gained', 0):.4f}
  Still Needed:  ${max(0, dp.get('needed', 0) - dp.get('gained', 0)):.4f}

MODE: {ce.state.mode} | Fear & Greed: {self.market_feed.fear_greed_value} ({self.market_feed.fear_greed_label})

PERFORMANCE
  Total Trades:  {stats['total_trades']}
  Win Rate:      {stats['win_rate']}%
  Total PnL:     ${stats['total_pnl']:.4f}
  Today PnL:     ${stats['today_pnl']:.4f}
  Open Pos:      {stats['open_positions']}
  Avg Win:       ${stats.get('avg_win', 0):.4f}
  Avg Loss:      ${stats.get('avg_loss', 0):.4f}
  Profit Factor: {abs(stats.get('avg_win', 0) / stats.get('avg_loss', 0.01)):.2f}x

SIZING
  Risk/Trade:    {ce.compute_risk_pct()*100:.2f}%
  Conf Floor:    {ce.compute_confidence_floor()}
  Max Positions: {ce.compute_max_concurrent()}

MARKET
  Regime:        {self.regime.regime}
  Whale Bias:    {self.whales.last_summary.get('market_bias', 'UNKNOWN')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next target resets at midnight UTC.
"""
            self._send_email(
                f"[ByBitDouble] Daily Report | ${self.balance:.2f} | +{self.daily_target_pct}% target | {ce.state.mode}",
                body
            )
        except Exception as e:
            logger.error(f"daily report: {e}")

    def _send_email(self, subject: str, body: str):
        if not SMTP_USER:
            logger.warning(f"[EMAIL SKIPPED] SMTP_USER not set. Subject: {subject}")
            return False
        if not SMTP_PASS:
            logger.warning(f"[EMAIL SKIPPED] SMTP_PASS not set. Subject: {subject}")
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SMTP_USER
            msg["To"]      = REPORT_EMAIL
            logger.info(f"Sending email to {REPORT_EMAIL} via {SMTP_HOST}:{SMTP_PORT}...")
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as sv:
                sv.starttls()
                sv.login(SMTP_USER, SMTP_PASS)
                sv.send_message(msg)
            logger.info(f"Email sent successfully: {subject}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"EMAIL AUTH FAILED — Check SMTP_USER/SMTP_PASS. Use Gmail App Password, not regular password. Error: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"EMAIL SMTP ERROR: {e}")
            return False
        except Exception as e:
            logger.error(f"EMAIL ERROR: {e}")
            return False

    def test_email(self) -> Dict:
        """Send a test email and return the result."""
        result = {
            "smtp_host": SMTP_HOST,
            "smtp_port": SMTP_PORT,
            "smtp_user": SMTP_USER[:4] + "***" if SMTP_USER else "NOT SET",
            "smtp_pass": "SET" if SMTP_PASS else "NOT SET",
            "report_email": REPORT_EMAIL,
        }

        if not SMTP_USER or not SMTP_PASS:
            result["status"] = "FAILED"
            result["error"] = f"Missing: {'SMTP_USER' if not SMTP_USER else ''} {'SMTP_PASS' if not SMTP_PASS else ''}".strip()
            return result

        success = self._send_email(
            "[ByBitDouble] Test Email — Connection Verified",
            f"""
BYBIT COMPOUND MCP — TEST EMAIL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Status:   Email delivery is working!
Balance:  ${self.balance:.2f} USDT
Engine:   {self.status}
Epoch:    {self.compound.state.epoch_num}
Mode:     {self.compound.state.mode}

If you received this email, daily reports will be delivered
at 00:00 UTC each day.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        )
        result["status"] = "SUCCESS" if success else "FAILED"
        if not success:
            result["error"] = "SMTP connection or authentication failed. Check Render logs for details."
        return result


engine = CompoundTradingEngine()
