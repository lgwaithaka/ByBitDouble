"""
whale_intelligence.py v2.0 — Smart Money & Whale Detection Engine
═══════════════════════════════════════════════════════════════════
Detects institutional/whale activity using Bybit on-exchange data:
  1. Open interest spike detection (sudden +/-5% in 1h)
  2. Funding rate extremes (crowded longs/shorts)
  3. Liquidation cascade analysis
  4. Orderbook wall detection (bid vs ask depth imbalance)
  5. Long/Short ratio extremes
  6. Volume spike detection (3x average = whale accumulation)

Outputs:
  - Per-symbol WhaleIntel with composite score [-1, +1]
  - Market-wide summary (bias, squeezes, counts)
  - Top opportunities ranked by absolute score
  - Full symbol cache for dashboard rendering
"""
import time, logging, asyncio
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WhaleIntel:
    """Per-symbol whale intelligence data."""
    def __init__(self, symbol: str):
        self.symbol        = symbol
        self.oi_spike      = 0.0     # OI change % in 1h
        self.oi_current    = 0.0     # current OI value (USDT)
        self.funding_extreme = False  # Extreme funding rate
        self.funding_rate  = 0.0
        self.liq_cascade   = False   # Recent large liquidations detected
        self.liq_net       = 0.0     # net long-short liquidation volume
        self.liq_total     = 0.0     # total liquidation volume
        self.ob_wall       = 0.0     # orderbook imbalance [-1, +1]
        self.ob_bid_vol    = 0.0     # total bid volume
        self.ob_ask_vol    = 0.0     # total ask volume
        self.ls_extreme    = False   # L/S ratio extreme
        self.ls_ratio      = 0.5     # long ratio (0-1)
        self.vol_spike     = False   # Volume > 3x average
        self.vol_ratio     = 0.0     # current vol / avg vol
        self.composite     = 0.0     # Overall whale score [-1, +1]
        self.bias          = "NEUTRAL"
        self.ts            = 0       # unix timestamp of analysis
        self.signals_hit   = 0       # how many of 6 signals fired

    def to_dict(self) -> Dict:
        """Full serialization for dashboard/API."""
        age = time.time() - self.ts if self.ts else 0
        return {
            "symbol":       self.symbol,
            "score":        round(self.composite, 3),
            "bias":         self.bias,
            "signals_hit":  self.signals_hit,
            "oi_spike":     round(self.oi_spike, 2),
            "oi_current":   round(self.oi_current, 0),
            "funding":      round(self.funding_rate * 100, 4),
            "funding_extreme": self.funding_extreme,
            "liq_cascade":  self.liq_cascade,
            "liq_net":      round(self.liq_net, 2),
            "liq_total":    round(self.liq_total, 2),
            "ob_imb":       round(self.ob_wall, 3),
            "ob_bid_vol":   round(self.ob_bid_vol, 0),
            "ob_ask_vol":   round(self.ob_ask_vol, 0),
            "ls_ratio":     round(self.ls_ratio, 3),
            "ls_extreme":   self.ls_extreme,
            "vol_spike":    self.vol_spike,
            "vol_ratio":    round(self.vol_ratio, 2),
            "ts":           self.ts,
            "age_sec":      int(age),
            "fresh":        age < 7200,
        }


class WhaleEngine:
    """Aggregates whale intelligence across all scanned symbols."""

    def __init__(self):
        self._cache: Dict[str, WhaleIntel] = {}
        self.last_summary: Dict = {}
        self._client = None
        self._update_count = 0
        self._last_update_ts = 0

    def set_client(self, client):
        self._client = client

    async def update_all(self, symbols: List[str], client=None) -> Dict:
        """Update whale data for all symbols. Returns summary."""
        if client:
            self._client = client
        if not self._client:
            return {}

        tasks = [self._analyze_symbol(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        squeezes = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_signals = 0
        strongest_bull = None
        strongest_bear = None

        for r in results:
            if isinstance(r, Exception):
                continue
            if r is None:
                continue
            self._cache[r.symbol] = r
            total_signals += r.signals_hit

            if r.composite > 0.3:
                bullish_count += 1
                if strongest_bull is None or r.composite > strongest_bull[1]:
                    strongest_bull = (r.symbol, r.composite)
            elif r.composite < -0.3:
                bearish_count += 1
                if strongest_bear is None or r.composite < strongest_bear[1]:
                    strongest_bear = (r.symbol, r.composite)
            else:
                neutral_count += 1

            # Short squeeze detection
            if r.funding_rate < -0.03 and r.oi_spike > 3 and r.ob_wall > 0.15:
                squeezes.append(r.symbol)

        total = bullish_count + bearish_count
        if total == 0:
            bias = "NEUTRAL"
        elif bullish_count > bearish_count * 1.5:
            bias = "BULLISH"
        elif bearish_count > bullish_count * 1.5:
            bias = "BEARISH"
        else:
            bias = "MIXED"

        self._update_count += 1
        self._last_update_ts = int(time.time())

        self.last_summary = {
            "market_bias":       bias,
            "bullish_symbols":   bullish_count,
            "bearish_symbols":   bearish_count,
            "neutral_symbols":   neutral_count,
            "total_symbols":     len(symbols),
            "symbols_tracked":   len(self._cache),
            "short_squeezes":    squeezes,
            "total_signals_hit": total_signals,
            "strongest_bull":    strongest_bull,
            "strongest_bear":    strongest_bear,
            "update_count":      self._update_count,
            "ts":                self._last_update_ts,
        }
        logger.info(
            f"WHALE UPDATE #{self._update_count} | bias={bias} "
            f"bull={bullish_count} bear={bearish_count} neutral={neutral_count} "
            f"squeezes={squeezes} signals={total_signals}"
        )
        return self.last_summary

    async def _analyze_symbol(self, symbol: str) -> Optional[WhaleIntel]:
        """Analyze whale activity for a single symbol."""
        try:
            intel = WhaleIntel(symbol)
            intel.ts = int(time.time())

            results = await asyncio.gather(
                self._client.open_interest(symbol, "1h"),
                self._client.funding(symbol),
                self._client.liquidations(symbol),
                self._client.orderbook(symbol, 50),
                self._client.ls_ratio(symbol),
                self._client.klines(symbol, "60", 25),
                return_exceptions=True,
            )
            oi_data, fr_data, liq_data, ob_data, ls_data, kl_data = results
            signals_fired = 0

            # 1. OI Spike
            if not isinstance(oi_data, Exception):
                lst = oi_data.get("result", {}).get("list", [])
                if len(lst) >= 2:
                    cur  = float(lst[0].get("openInterest", "0") or "0")
                    prev = float(lst[1].get("openInterest", "0") or "0")
                    intel.oi_current = cur
                    if prev > 0:
                        intel.oi_spike = (cur - prev) / prev * 100
                        if abs(intel.oi_spike) > 3:
                            signals_fired += 1

            # 2. Funding Rate
            if not isinstance(fr_data, Exception):
                lst = fr_data.get("result", {}).get("list", [])
                if lst:
                    rate = float(lst[0].get("fundingRate", "0") or "0")
                    intel.funding_rate    = rate
                    intel.funding_extreme = abs(rate) > 0.03
                    if intel.funding_extreme:
                        signals_fired += 1

            # 3. Liquidations
            if not isinstance(liq_data, Exception):
                lst = liq_data.get("result", {}).get("list", [])
                if lst:
                    long_vol = sum(float(x.get("qty", "0") or "0")
                                   for x in lst[:20] if x.get("side") == "Buy")
                    short_vol = sum(float(x.get("qty", "0") or "0")
                                    for x in lst[:20] if x.get("side") == "Sell")
                    intel.liq_net = long_vol - short_vol
                    intel.liq_total = long_vol + short_vol
                    intel.liq_cascade = intel.liq_total > 0
                    if intel.liq_cascade:
                        signals_fired += 1

            # 4. Orderbook Imbalance
            if not isinstance(ob_data, Exception):
                result = ob_data.get("result", {})
                bids = result.get("b", [])[:50]
                asks = result.get("a", [])[:50]
                bid_vol = sum(float(x[1]) for x in bids) if bids else 0
                ask_vol = sum(float(x[1]) for x in asks) if asks else 0
                intel.ob_bid_vol = bid_vol
                intel.ob_ask_vol = ask_vol
                total = bid_vol + ask_vol
                intel.ob_wall = (bid_vol - ask_vol) / total if total > 0 else 0
                if abs(intel.ob_wall) > 0.15:
                    signals_fired += 1

            # 5. Long/Short Ratio
            if not isinstance(ls_data, Exception):
                lst = ls_data.get("result", {}).get("list", [])
                if lst:
                    long_pct = float(lst[0].get("buyRatio", "0.5") or "0.5")
                    intel.ls_ratio   = long_pct
                    intel.ls_extreme = long_pct > 0.68 or long_pct < 0.32
                    if intel.ls_extreme:
                        signals_fired += 1

            # 6. Volume Spike
            if not isinstance(kl_data, Exception):
                lst = kl_data.get("result", {}).get("list", [])
                if lst and len(lst) >= 10:
                    volumes = [float(x[5]) for x in lst]
                    recent  = volumes[0] if volumes else 0
                    avg     = sum(volumes[1:]) / max(1, len(volumes) - 1)
                    intel.vol_ratio = recent / avg if avg > 0 else 0
                    intel.vol_spike = intel.vol_ratio > 3.0
                    if intel.vol_spike:
                        signals_fired += 1

            intel.signals_hit = signals_fired

            # ── Composite Score [-1, +1] ──
            score = 0.0
            if intel.oi_spike > 5:      score += 0.25
            elif intel.oi_spike < -5:   score -= 0.25

            if intel.funding_extreme:
                score += -0.3 if intel.funding_rate > 0 else 0.3

            if intel.liq_cascade:
                score += -0.15 if intel.liq_net > 0 else 0.15

            score += intel.ob_wall * 0.2

            if intel.ls_extreme:
                score += -0.15 if intel.ls_ratio > 0.65 else 0.15

            if intel.vol_spike:
                score += 0.1

            intel.composite = max(-1.0, min(1.0, score))
            intel.bias = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")

            return intel

        except Exception as e:
            logger.debug(f"Whale analysis {symbol}: {e}")
            return None

    def get_signal_for(self, symbol: str) -> float:
        """Get cached whale score for a symbol. Decays with age."""
        intel = self._cache.get(symbol)
        if not intel:
            return 0.0
        age = time.time() - intel.ts
        if age > 7200:
            return 0.0
        decay = max(0.0, 1.0 - age / 7200)
        return intel.composite * decay

    def get_top_opportunities(self, n: int = 10) -> List[Dict]:
        """Get top whale-backed opportunities sorted by abs score."""
        items = []
        for sym, intel in self._cache.items():
            if abs(intel.composite) < 0.15:
                continue
            age = time.time() - intel.ts
            if age > 7200:
                continue
            items.append(intel.to_dict())
        items.sort(key=lambda x: abs(x["score"]), reverse=True)
        return items[:n]

    def get_all_symbols(self) -> List[Dict]:
        """Get ALL tracked symbols data (unfiltered) for dashboard."""
        items = []
        for sym, intel in self._cache.items():
            items.append(intel.to_dict())
        items.sort(key=lambda x: abs(x["score"]), reverse=True)
        return items

    def get_cache_stats(self) -> Dict:
        """Cache diagnostics."""
        now = time.time()
        total = len(self._cache)
        fresh = sum(1 for i in self._cache.values() if (now - i.ts) < 7200)
        stale = total - fresh
        return {
            "total_cached": total,
            "fresh": fresh,
            "stale": stale,
            "update_count": self._update_count,
            "last_update_ts": self._last_update_ts,
            "last_update_ago_sec": int(now - self._last_update_ts) if self._last_update_ts else 0,
        }
