"""
whale_intelligence.py — Smart Money & Whale Detection Engine
═══════════════════════════════════════════════════════════════
Detects institutional/whale activity using Bybit on-exchange data:
  1. Open interest spike detection (sudden +/-5% in 1h)
  2. Funding rate extremes (crowded longs/shorts)
  3. Liquidation cascade analysis
  4. Orderbook wall detection
  5. Long/Short ratio extremes
  6. Volume spike detection (3x average = whale accumulation)
"""
import time, logging, asyncio
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WhaleIntel:
    """Per-symbol whale intelligence data."""
    def __init__(self, symbol: str):
        self.symbol        = symbol
        self.oi_spike      = 0.0     # OI change % in 1h
        self.funding_extreme = False  # Extreme funding rate
        self.funding_rate  = 0.0
        self.liq_cascade   = False   # Recent large liquidations
        self.liq_net       = 0.0     # net long-short liquidation volume
        self.ob_wall       = 0.0     # orderbook imbalance
        self.ls_extreme    = False   # L/S ratio extreme
        self.ls_ratio      = 0.5
        self.vol_spike     = False   # Volume > 3x average
        self.composite     = 0.0     # Overall whale score [-1, +1]
        self.bias          = "NEUTRAL"
        self.ts            = 0


class WhaleEngine:
    """Aggregates whale intelligence across all scanned symbols."""

    def __init__(self):
        self._cache: Dict[str, WhaleIntel] = {}
        self.last_summary: Dict = {}
        self._client = None

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

        for r in results:
            if isinstance(r, Exception):
                continue
            if r is None:
                continue
            self._cache[r.symbol] = r
            if r.composite > 0.3:
                bullish_count += 1
            elif r.composite < -0.3:
                bearish_count += 1
            # Short squeeze detection: extreme negative funding + OI spike + bullish OB
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
            bias = "NEUTRAL"

        self.last_summary = {
            "market_bias":     bias,
            "bullish_symbols": bullish_count,
            "bearish_symbols": bearish_count,
            "short_squeezes":  squeezes,
            "symbols_tracked": len(self._cache),
            "ts":              int(time.time()),
        }
        return self.last_summary

    async def _analyze_symbol(self, symbol: str) -> Optional[WhaleIntel]:
        """Analyze whale activity for a single symbol."""
        try:
            intel = WhaleIntel(symbol)
            intel.ts = int(time.time())

            # Gather all data concurrently
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

            # 1. OI Spike
            if not isinstance(oi_data, Exception):
                lst = oi_data.get("result", {}).get("list", [])
                if len(lst) >= 2:
                    cur  = float(lst[0].get("openInterest", "0") or "0")
                    prev = float(lst[1].get("openInterest", "0") or "0")
                    if prev > 0:
                        intel.oi_spike = (cur - prev) / prev * 100

            # 2. Funding Rate
            if not isinstance(fr_data, Exception):
                lst = fr_data.get("result", {}).get("list", [])
                if lst:
                    rate = float(lst[0].get("fundingRate", "0") or "0")
                    intel.funding_rate    = rate
                    intel.funding_extreme = abs(rate) > 0.03  # >0.03% = extreme

            # 3. Liquidations
            if not isinstance(liq_data, Exception):
                lst = liq_data.get("result", {}).get("list", [])
                if lst:
                    long_vol = sum(float(x.get("qty", "0") or "0")
                                   for x in lst[:20] if x.get("side") == "Buy")
                    short_vol = sum(float(x.get("qty", "0") or "0")
                                    for x in lst[:20] if x.get("side") == "Sell")
                    intel.liq_net = long_vol - short_vol
                    intel.liq_cascade = (long_vol + short_vol) > 0

            # 4. Orderbook Imbalance
            if not isinstance(ob_data, Exception):
                result = ob_data.get("result", {})
                bids = result.get("b", [])[:50]
                asks = result.get("a", [])[:50]
                bid_vol = sum(float(x[1]) for x in bids) if bids else 0
                ask_vol = sum(float(x[1]) for x in asks) if asks else 0
                total = bid_vol + ask_vol
                intel.ob_wall = (bid_vol - ask_vol) / total if total > 0 else 0

            # 5. Long/Short Ratio
            if not isinstance(ls_data, Exception):
                lst = ls_data.get("result", {}).get("list", [])
                if lst:
                    long_pct = float(lst[0].get("buyRatio", "0.5") or "0.5")
                    intel.ls_ratio   = long_pct
                    intel.ls_extreme = long_pct > 0.68 or long_pct < 0.32

            # 6. Volume Spike
            if not isinstance(kl_data, Exception):
                lst = kl_data.get("result", {}).get("list", [])
                if lst and len(lst) >= 10:
                    volumes = [float(x[5]) for x in lst]
                    recent  = volumes[0] if volumes else 0
                    avg     = sum(volumes[1:]) / max(1, len(volumes) - 1)
                    intel.vol_spike = recent > avg * 3.0

            # Composite Score [-1, +1]
            score = 0.0
            # OI spike: bullish if positive, bearish if negative
            if intel.oi_spike > 5:
                score += 0.25
            elif intel.oi_spike < -5:
                score -= 0.25

            # Funding extreme: negative funding = shorts crowded = bullish
            if intel.funding_extreme:
                score += -0.3 if intel.funding_rate > 0 else 0.3

            # Liquidations: net positive = longs liquidated = bearish pressure
            if intel.liq_cascade:
                if intel.liq_net > 0:
                    score -= 0.15
                else:
                    score += 0.15

            # Orderbook: positive = more bids = bullish
            score += intel.ob_wall * 0.2

            # L/S extreme: too many longs = bearish, too many shorts = bullish
            if intel.ls_extreme:
                if intel.ls_ratio > 0.65:
                    score -= 0.15  # crowded longs
                else:
                    score += 0.15  # crowded shorts

            # Volume spike
            if intel.vol_spike:
                score += 0.1  # accumulation signal

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
        if age > 7200:  # >2h stale
            return 0.0
        decay = max(0.0, 1.0 - age / 7200)
        return intel.composite * decay

    def get_top_opportunities(self, n: int = 5) -> List[Dict]:
        """Get top whale-backed opportunities."""
        items = []
        for sym, intel in self._cache.items():
            if abs(intel.composite) < 0.15:
                continue
            age = time.time() - intel.ts
            if age > 7200:
                continue
            items.append({
                "symbol":    sym,
                "score":     round(intel.composite, 3),
                "bias":      intel.bias,
                "oi_spike":  round(intel.oi_spike, 2),
                "funding":   round(intel.funding_rate * 100, 4),
                "ob_imb":    round(intel.ob_wall, 3),
                "ls_ratio":  round(intel.ls_ratio, 3),
                "vol_spike": intel.vol_spike,
            })
        items.sort(key=lambda x: abs(x["score"]), reverse=True)
        return items[:n]
