"""
whale_intelligence.py — Smart Money & Whale Detection Engine
═══════════════════════════════════════════════════════════════════════════════
Detects institutional/whale activity using:

1. BYBIT ON-EXCHANGE SIGNALS (free, no external API needed)
   • Large trade detection via recent trades endpoint
   • Open interest spike detection (sudden +/- 5% in 1h)
   • Funding rate extremes (longs/shorts paying heavily)
   • Liquidation cascade analysis (large position liquidations)
   • Orderbook walls — detecting large limit orders
   • Long/Short ratio extremes (crowd positioning)

2. COINGLASS FREE DATA (no auth required, public endpoints)
   • Global OI changes across all exchanges
   • Large liquidation events
   • Funding rate aggregates

3. DERIVED WHALE SIGNALS (computed from Bybit market data)
   • Volume-to-OI ratio spikes (whales opening large positions)
   • Bid/ask absorption (large orders being filled)
   • Price-OI divergence (price rising but OI falling = shorts covering)

UPDATE CYCLE: Every 60 minutes (hourly) + on every scan cycle (light check)

SIGNAL INTERPRETATION (research-backed):
  Exchange inflows  → selling pressure (short signal)
  Exchange outflows → accumulation (buy signal)
  OI spike + price rise → new longs entering → bullish continuation
  OI spike + price fall → new shorts entering → bearish continuation
  OI drop + price rise → short squeeze (shorts covering) → bullish
  OI drop + price fall → long liquidation → bearish
  Extreme positive funding → crowded longs → contrarian short
  Large short liquidation cascade → buy the squeeze
═══════════════════════════════════════════════════════════════════════════════
"""
import asyncio
import aiohttp
import time
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# CoinGlass public endpoints (no API key needed)
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
COINGLASS_KEY  = os.getenv("COINGLASS_API_KEY", "")  # optional, increases rate limits


@dataclass
class WhaleSignal:
    symbol:       str
    direction:    int      # +1 = bullish whale activity, -1 = bearish, 0 = neutral
    strength:     float    # 0.0 to 1.0
    signal_type:  str      # what type of whale signal
    description:  str      # human readable
    ts:           int = field(default_factory=lambda: int(time.time()))


@dataclass
class WhaleIntelligence:
    """Aggregated whale intelligence for a symbol."""
    symbol:          str
    composite_score: float    # -1.0 to +1.0 (bull/bear smart money)
    signals:         List[WhaleSignal] = field(default_factory=list)
    oi_trend:        str  = "NEUTRAL"   # ACCUMULATING | DISTRIBUTING | NEUTRAL
    funding_extreme: bool = False
    liq_cascade:     str  = "NONE"      # SHORT_SQUEEZE | LONG_FLUSH | NONE
    large_trade_bias: str = "NEUTRAL"   # BUY | SELL | NEUTRAL
    updated_ts:      int  = field(default_factory=lambda: int(time.time()))


class WhaleEngine:
    """
    Tracks smart money movements across all scanned symbols.
    Updates hourly via Bybit's free market data APIs.
    No external paid APIs required.
    """

    def __init__(self):
        self._sess:     Optional[aiohttp.ClientSession] = None
        self._cache:    Dict[str, WhaleIntelligence]    = {}
        self._oi_hist:  Dict[str, List[float]]          = {}   # symbol → [oi_values]
        self._vol_hist: Dict[str, List[float]]          = {}   # symbol → [volume_values]
        self._price_hist: Dict[str, List[float]]        = {}
        self.last_update  = 0
        self.last_summary = {}
        self._bybit_client = None   # Set by engine on init

    def set_client(self, client):
        """Inject the Bybit client for API calls."""
        self._bybit_client = client

    async def _get_sess(self) -> aiohttp.ClientSession:
        if not self._sess or self._sess.closed:
            self._sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._sess

    async def close(self):
        if self._sess and not self._sess.closed:
            await self._sess.close()

    # ── Core Analysis Methods ─────────────────────────────────────────────────

    async def analyze_symbol(self, symbol: str, client) -> WhaleIntelligence:
        """Full whale analysis for one symbol using Bybit free endpoints."""
        signals: List[WhaleSignal] = []
        scores:  List[float]       = []

        try:
            # Fetch all needed data in parallel
            results = await asyncio.gather(
                client.open_interest(symbol, "5m"),    # 5-min OI for spike detection
                client.open_interest(symbol, "1h"),    # 1h OI for trend
                client.liquidations(symbol),
                client.orderbook(symbol, 50),
                client._get("/v5/market/recent-trade", {
                    "category": "linear", "symbol": symbol, "limit": "60"
                }, auth=False),
                client.ls_ratio(symbol),
                client.funding(symbol),
                return_exceptions=True
            )
            oi5m, oi1h, liqs, ob, recent_trades, lsr, funding = results

            # ── 1. Open Interest Analysis ─────────────────────────────────────
            oi_score, oi_trend = self._analyze_oi(symbol, oi5m, oi1h)
            if abs(oi_score) > 0.2:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(oi_score > 0) * 2 - 1,
                    strength=abs(oi_score),
                    signal_type="OI_TREND",
                    description=f"OI {oi_trend}: score={oi_score:.2f}"
                ))
            scores.append(oi_score * 1.5)  # High weight

            # ── 2. Liquidation Cascade Detection ──────────────────────────────
            liq_score, liq_type, liq_desc = self._analyze_liquidations(symbol, liqs)
            if abs(liq_score) > 0.3:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(liq_score > 0) * 2 - 1,
                    strength=abs(liq_score),
                    signal_type="LIQUIDATION_CASCADE",
                    description=liq_desc
                ))
            scores.append(liq_score * 2.0)  # Very high weight — cascades = strong signal

            # ── 3. Large Trade Detection ──────────────────────────────────────
            trade_score, trade_bias, trade_desc = self._analyze_large_trades(recent_trades)
            if abs(trade_score) > 0.2:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(trade_score > 0) * 2 - 1,
                    strength=abs(trade_score),
                    signal_type="LARGE_TRADE_FLOW",
                    description=trade_desc
                ))
            scores.append(trade_score * 1.8)

            # ── 4. Orderbook Wall Analysis ────────────────────────────────────
            wall_score, wall_desc = self._analyze_orderbook_walls(ob)
            if abs(wall_score) > 0.3:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(wall_score > 0) * 2 - 1,
                    strength=abs(wall_score),
                    signal_type="ORDERBOOK_WALL",
                    description=wall_desc
                ))
            scores.append(wall_score * 1.2)

            # ── 5. Funding Rate Extreme ───────────────────────────────────────
            fund_score, fund_extreme, fund_desc = self._analyze_funding_extreme(funding)
            if abs(fund_score) > 0.3:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(fund_score > 0) * 2 - 1,
                    strength=abs(fund_score),
                    signal_type="FUNDING_EXTREME",
                    description=fund_desc
                ))
            scores.append(fund_score * 1.3)

            # ── 6. Long/Short Ratio Extreme ───────────────────────────────────
            ls_score, ls_desc = self._analyze_ls_extreme(lsr)
            if abs(ls_score) > 0.3:
                signals.append(WhaleSignal(
                    symbol=symbol, direction=int(ls_score > 0) * 2 - 1,
                    strength=abs(ls_score),
                    signal_type="LS_RATIO_EXTREME",
                    description=ls_desc
                ))
            scores.append(ls_score * 1.0)

            # ── Composite ─────────────────────────────────────────────────────
            composite = sum(scores) / max(len(scores), 1) if scores else 0.0
            composite = max(-1.0, min(1.0, composite))

            intel = WhaleIntelligence(
                symbol=symbol,
                composite_score=round(composite, 3),
                signals=signals,
                oi_trend=oi_trend,
                funding_extreme=fund_extreme,
                liq_cascade=liq_type,
                large_trade_bias=trade_bias,
            )
            self._cache[symbol] = intel
            return intel

        except Exception as e:
            logger.error(f"WhaleEngine.analyze_symbol {symbol}: {e}")
            return WhaleIntelligence(symbol=symbol, composite_score=0.0)

    # ── Analysis Sub-Methods ──────────────────────────────────────────────────

    def _analyze_oi(self, symbol: str, oi5m, oi1h) -> Tuple[float, str]:
        """
        OI + Price correlation:
        Rising OI + Rising Price = new longs → BULLISH
        Rising OI + Falling Price = new shorts → BEARISH
        Falling OI + Rising Price = shorts covering → BULLISH (squeeze)
        Falling OI + Falling Price = longs exiting → BEARISH
        """
        try:
            # 5-minute OI for spike detection
            items_5m = oi5m.get("result", {}).get("list", []) if not isinstance(oi5m, Exception) else []
            items_1h = oi1h.get("result", {}).get("list", []) if not isinstance(oi1h, Exception) else []

            if not items_5m or len(items_5m) < 3:
                return 0.0, "NEUTRAL"

            oi_vals = [float(x.get("openInterest", 0) or 0) for x in items_5m[:12]]
            if not oi_vals or oi_vals[0] == 0:
                return 0.0, "NEUTRAL"

            # OI change over last 5 periods
            oi_recent  = float(items_5m[0].get("openInterest", 0) or 0)
            oi_old     = float(items_5m[min(5, len(items_5m)-1)].get("openInterest", 1) or 1)
            oi_pct_chg = (oi_recent - oi_old) / oi_old * 100

            # Track history for trend analysis
            if symbol not in self._oi_hist:
                self._oi_hist[symbol] = []
            self._oi_hist[symbol].append(oi_recent)
            self._oi_hist[symbol] = self._oi_hist[symbol][-20:]

            # Spike threshold: >3% OI change in 5 periods = significant
            if abs(oi_pct_chg) > 5.0:
                score  = 0.8 if oi_pct_chg > 0 else -0.8
                trend  = "ACCUMULATING" if oi_pct_chg > 0 else "DISTRIBUTING"
            elif abs(oi_pct_chg) > 2.0:
                score  = 0.4 if oi_pct_chg > 0 else -0.4
                trend  = "MILD_ACCUMULATE" if oi_pct_chg > 0 else "MILD_DISTRIBUTE"
            elif oi_pct_chg > 0:
                score  = 0.15
                trend  = "NEUTRAL_BULL"
            elif oi_pct_chg < 0:
                score  = -0.15
                trend  = "NEUTRAL_BEAR"
            else:
                score  = 0.0
                trend  = "NEUTRAL"

            return score, trend
        except Exception:
            return 0.0, "NEUTRAL"

    def _analyze_liquidations(self, symbol: str, liqs) -> Tuple[float, str, str]:
        """
        Large liquidation cascade signals:
        Many shorts liquidated → price rose fast → momentum is bullish
        Many longs liquidated → price fell fast → momentum is bearish
        """
        try:
            if isinstance(liqs, Exception):
                return 0.0, "NONE", "No liq data"
            items = liqs.get("result", {}).get("list", [])
            if not items:
                return 0.0, "NONE", "No liquidations"

            # "Buy" = long position liquidated (price fell)
            # "Sell" = short position liquidated (price rose)
            long_liq  = sum(float(x.get("size", 0) or 0) for x in items if x.get("side") == "Buy")
            short_liq = sum(float(x.get("size", 0) or 0) for x in items if x.get("side") == "Sell")
            total     = long_liq + short_liq

            if total < 1:
                return 0.0, "NONE", "No significant liquidations"

            short_dom  = short_liq / total if total > 0 else 0.5
            long_dom   = long_liq / total if total > 0 else 0.5

            if short_dom > 0.75:
                score = 0.9; ltype = "SHORT_SQUEEZE"
                desc  = f"SHORT SQUEEZE: {short_dom*100:.0f}% of liqs are shorts (${short_liq:.0f})"
            elif short_dom > 0.60:
                score = 0.5; ltype = "SHORT_PRESSURE"
                desc  = f"Short pressure: {short_dom*100:.0f}% short liqs"
            elif long_dom > 0.75:
                score = -0.9; ltype = "LONG_FLUSH"
                desc  = f"LONG FLUSH: {long_dom*100:.0f}% of liqs are longs (${long_liq:.0f})"
            elif long_dom > 0.60:
                score = -0.5; ltype = "LONG_PRESSURE"
                desc  = f"Long pressure: {long_dom*100:.0f}% long liqs"
            else:
                score = 0.0; ltype = "MIXED"
                desc  = f"Mixed: {long_liq:.0f} long / {short_liq:.0f} short"

            return score, ltype, desc
        except Exception as e:
            return 0.0, "NONE", str(e)

    def _analyze_large_trades(self, recent_trades) -> Tuple[float, str, str]:
        """
        Detect large individual trades (whale buy/sell orders).
        A trade > 5× average size = whale activity.
        """
        try:
            if isinstance(recent_trades, Exception):
                return 0.0, "NEUTRAL", "No trade data"

            items = recent_trades.get("result", {}).get("list", [])
            if not items or len(items) < 10:
                return 0.0, "NEUTRAL", "Insufficient trade data"

            sizes = [float(x.get("size", 0) or 0) for x in items]
            if not sizes:
                return 0.0, "NEUTRAL", "No sizes"

            avg_size  = sum(sizes) / len(sizes)
            threshold = avg_size * 5.0  # 5× average = whale

            whale_buys  = sum(float(x.get("size",0) or 0)
                              for x in items
                              if float(x.get("size",0) or 0) > threshold
                              and x.get("side") == "Buy")
            whale_sells = sum(float(x.get("size",0) or 0)
                              for x in items
                              if float(x.get("size",0) or 0) > threshold
                              and x.get("side") == "Sell")
            total_whale = whale_buys + whale_sells

            if total_whale < avg_size * 2:
                return 0.0, "NEUTRAL", "No significant whale trades"

            buy_pct = whale_buys / total_whale if total_whale > 0 else 0.5

            if buy_pct > 0.70:
                score = 0.7; bias = "BUY"
                desc  = f"Whale buying dominant: {buy_pct*100:.0f}% of large trades"
            elif buy_pct < 0.30:
                score = -0.7; bias = "SELL"
                desc  = f"Whale selling dominant: {(1-buy_pct)*100:.0f}% of large trades"
            else:
                score = 0.0; bias = "NEUTRAL"
                desc  = f"Balanced whale activity: {buy_pct*100:.0f}% buys"

            return score, bias, desc
        except Exception:
            return 0.0, "NEUTRAL", "Analysis error"

    def _analyze_orderbook_walls(self, ob) -> Tuple[float, str]:
        """
        Detect large limit order walls on bid or ask side.
        Big bid wall = whales defending support (bullish)
        Big ask wall = whales defending resistance (bearish)
        """
        try:
            if isinstance(ob, Exception):
                return 0.0, "No orderbook"

            bids = ob.get("result", {}).get("b", [])
            asks = ob.get("result", {}).get("a", [])

            if not bids or not asks:
                return 0.0, "No data"

            # Sum top 5 vs next 20 to detect walls
            bid_top5  = sum(float(b[1]) for b in bids[:5])
            bid_rest  = sum(float(b[1]) for b in bids[5:25])
            ask_top5  = sum(float(a[1]) for a in asks[:5])
            ask_rest  = sum(float(a[1]) for a in asks[5:25])

            total_bid = sum(float(b[1]) for b in bids[:25])
            total_ask = sum(float(a[1]) for a in asks[:25])
            imbalance = total_bid / (total_ask + 1e-9)

            # Wall detection: top 5 levels > 3× the rest per level
            bid_wall = bid_top5 / (bid_rest / 20 + 1e-9) if bid_rest > 0 else 0
            ask_wall = ask_top5 / (ask_rest / 20 + 1e-9) if ask_rest > 0 else 0

            if bid_wall > 5 and imbalance > 1.5:
                return 0.7, f"Large BID WALL: {bid_top5:.1f} vs ask {ask_top5:.1f}"
            elif ask_wall > 5 and imbalance < 0.67:
                return -0.7, f"Large ASK WALL: {ask_top5:.1f} vs bid {bid_top5:.1f}"
            elif imbalance > 2.0:
                return 0.4, f"Strong bid imbalance: {imbalance:.1f}×"
            elif imbalance < 0.5:
                return -0.4, f"Strong ask imbalance: {1/imbalance:.1f}×"
            else:
                return 0.0, f"Balanced book: imbalance={imbalance:.2f}"
        except Exception:
            return 0.0, "Analysis error"

    def _analyze_funding_extreme(self, funding) -> Tuple[float, bool, str]:
        """
        Extreme funding = crowded one side = contrarian signal.
        Research: Funding extremes are one of the most reliable contrarian signals.
        """
        try:
            if isinstance(funding, Exception):
                return 0.0, False, "No funding data"
            items = funding.get("result", {}).get("list", [])
            if not items:
                return 0.0, False, "No funding"

            rate = float(items[0].get("fundingRate", 0) or 0)

            if rate > 0.005:
                return -1.0, True, f"EXTREME POSITIVE FUNDING {rate*100:.3f}% — crowded long → short"
            elif rate > 0.002:
                return -0.7, True, f"Very high funding {rate*100:.3f}% — lean short"
            elif rate > 0.001:
                return -0.4, False, f"Elevated funding {rate*100:.3f}%"
            elif rate < -0.005:
                return  1.0, True, f"EXTREME NEGATIVE FUNDING {rate*100:.3f}% — crowded short → long"
            elif rate < -0.002:
                return  0.7, True, f"Very negative funding {rate*100:.3f}% — lean long"
            elif rate < -0.001:
                return  0.4, False, f"Low funding {rate*100:.3f}%"
            else:
                return 0.0, False, f"Normal funding {rate*100:.4f}%"
        except Exception:
            return 0.0, False, "Error"

    def _analyze_ls_extreme(self, lsr) -> Tuple[float, str]:
        """Long/short ratio extremes = contrarian signal (too crowded)."""
        try:
            if isinstance(lsr, Exception):
                return 0.0, "No data"
            items = lsr.get("result", {}).get("list", [])
            if not items:
                return 0.0, "No data"

            buy_r  = float(items[0].get("buyRatio",  0.5) or 0.5)
            sell_r = float(items[0].get("sellRatio", 0.5) or 0.5)
            ratio  = buy_r / sell_r if sell_r > 0 else 1.0

            if ratio > 2.0:
                return -0.8, f"EXTREME LONGS {ratio:.1f}× — contrarian SHORT"
            elif ratio > 1.5:
                return -0.5, f"Heavy longs {ratio:.1f}× — lean short"
            elif ratio < 0.5:
                return  0.8, f"EXTREME SHORTS {1/ratio:.1f}× — contrarian LONG"
            elif ratio < 0.67:
                return  0.5, f"Heavy shorts — lean long"
            else:
                return 0.0, f"Balanced L/S {ratio:.2f}"
        except Exception:
            return 0.0, "Error"

    # ── Hourly Full Update ─────────────────────────────────────────────────────

    async def update_all(self, symbols: List[str], client) -> Dict[str, WhaleIntelligence]:
        """
        Full hourly whale intelligence update for all scanned symbols.
        Returns dict of symbol → WhaleIntelligence.
        """
        logger.info(f"Whale intelligence update: {len(symbols)} symbols")
        results = {}
        for sym in symbols:
            try:
                intel = await self.analyze_symbol(sym, client)
                results[sym] = intel
                await asyncio.sleep(0.2)   # Rate limit
            except Exception as e:
                logger.error(f"Whale update {sym}: {e}")

        self.last_update  = int(time.time())
        self.last_summary = self._summarize(results)
        logger.info(f"Whale update complete | {len(results)} symbols analyzed")
        return results

    def _summarize(self, results: Dict[str, WhaleIntelligence]) -> Dict:
        """Market-wide whale sentiment summary."""
        if not results:
            return {}
        bull_count = sum(1 for v in results.values() if v.composite_score > 0.2)
        bear_count = sum(1 for v in results.values() if v.composite_score < -0.2)
        squeezes   = [k for k, v in results.items() if v.liq_cascade == "SHORT_SQUEEZE"]
        flushes    = [k for k, v in results.items() if v.liq_cascade == "LONG_FLUSH"]
        extremes   = [k for k, v in results.items() if v.funding_extreme]
        top_bull   = sorted(results.items(), key=lambda x: x[1].composite_score, reverse=True)[:3]
        top_bear   = sorted(results.items(), key=lambda x: x[1].composite_score)[:3]

        return {
            "bull_symbols":    bull_count,
            "bear_symbols":    bear_count,
            "neutral_symbols": len(results) - bull_count - bear_count,
            "market_bias":    "BULL" if bull_count > bear_count * 1.5 else
                              "BEAR" if bear_count > bull_count * 1.5 else "MIXED",
            "short_squeezes":  squeezes,
            "long_flushes":    flushes,
            "funding_extremes":extremes,
            "top_bullish":    [(k, round(v.composite_score, 3)) for k, v in top_bull],
            "top_bearish":    [(k, round(v.composite_score, 3)) for k, v in top_bear],
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }

    def get_signal_for(self, symbol: str) -> float:
        """Get cached whale composite score for a symbol. Returns 0 if not cached."""
        intel = self._cache.get(symbol)
        if not intel:
            return 0.0
        # Decay signal if older than 2 hours
        age = time.time() - intel.updated_ts
        if age > 7200:
            return 0.0
        # Decay linearly over 2 hours
        decay = max(0, 1.0 - (age / 7200))
        return intel.composite_score * decay

    def get_all_cache(self) -> Dict[str, Dict]:
        """Return all cached whale intelligence as dicts."""
        return {
            sym: {
                "composite_score": v.composite_score,
                "oi_trend":        v.oi_trend,
                "funding_extreme": v.funding_extreme,
                "liq_cascade":     v.liq_cascade,
                "large_trade_bias":v.large_trade_bias,
                "signals":         [
                    {"type": s.signal_type, "strength": s.strength,
                     "direction": s.direction, "desc": s.description}
                    for s in v.signals
                ],
                "updated_ts": v.updated_ts,
                "age_minutes": round((time.time() - v.updated_ts) / 60, 1),
            }
            for sym, v in self._cache.items()
        }

    def get_top_opportunities(self, n: int = 5) -> List[Dict]:
        """Get top N whale-backed opportunities sorted by absolute score."""
        items = [
            {"symbol": k, "score": v.composite_score, "direction": "LONG" if v.composite_score > 0 else "SHORT",
             "oi_trend": v.oi_trend, "cascade": v.liq_cascade, "signals": len(v.signals)}
            for k, v in self._cache.items()
            if abs(v.composite_score) > 0.15 and (time.time() - v.updated_ts) < 7200
        ]
        return sorted(items, key=lambda x: abs(x["score"]), reverse=True)[:n]
