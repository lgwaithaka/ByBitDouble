"""
scanner.py — Volume-based market scanner for Bybit
Identifies top movers by volume spike, volatility, and 24h change.
Also exports core indicator functions used across the system.
"""
import asyncio, logging, time
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ── Symbol Pool ──────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "MATICUSDT",
    "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT",
    "UNIUSDT", "AAVEUSDT", "STRKUSDT", "LTCUSDT", "ETCUSDT",
]

class VScanner:
    """Volume scanner that identifies top movers by volume and volatility."""

    def __init__(self, client=None, symbols=None):
        self.client = client
        self.top: List[Dict] = []
        self.last_scan = 0
        self.symbols_pool = symbols or DEFAULT_SYMBOLS

    async def scan(self, n: int = 12, force: bool = False) -> List[Dict]:
        if not force and time.time() - self.last_scan < 55:
            return self.top[:n]

        results = []
        for sym in self.symbols_pool:
            try:
                data = await self._fetch_symbol_metrics(sym)
                if data:
                    results.append(data)
            except Exception as e:
                logger.debug(f"Scan {sym}: {e}")

        results.sort(key=lambda x: x.get("vol_score", 0), reverse=True)
        self.top = results
        self.last_scan = time.time()
        return results[:n]

    async def _fetch_symbol_metrics(self, symbol: str) -> Optional[Dict]:
        try:
            klines = await self.client.klines(symbol, "60", 25)
            if not klines:
                return None

            k = parse_klines(klines)
            if k is None or len(k) < 24:
                return None

            closes  = k[:, 4]
            highs   = k[:, 2]
            lows    = k[:, 3]
            volumes = k[:, 5]

            vol_recent = np.mean(volumes[-6:])
            vol_avg    = np.mean(volumes[:-6]) if len(volumes) > 6 else vol_recent
            vol_score  = vol_recent / (vol_avg + 1e-9)

            price     = float(closes[-1])
            range_pct = float((highs[-24:].max() - lows[-24:].min()) / price * 100) if price > 0 else 0
            chg_24h   = float((closes[-1] - closes[-24]) / closes[-24] * 100) if closes[-24] > 0 else 0

            return {
                "symbol":     symbol,
                "vol_score":  float(vol_score),
                "range_pct":  float(range_pct),
                "chg_24h":    chg_24h,
                "price":      price,
                "volume_24h": float(np.sum(volumes[-24:])),
            }
        except Exception as e:
            logger.debug(f"Metrics {symbol}: {e}")
            return None

    def get_status(self) -> Dict:
        return {
            "symbols_tracked": len(self.symbols_pool),
            "last_scan":       self.last_scan,
            "top_count":       len(self.top),
            "top_symbols":     [x["symbol"] for x in self.top[:5]],
        }

MarketScanner = VScanner

# ═══════════════════════════════════════════════════════════════════════════════
# CORE INDICATOR FUNCTIONS — Used by signals.py, engine.py
# ═══════════════════════════════════════════════════════════════════════════════

def parse_klines(klines) -> Optional[np.ndarray]:
    """Parse Bybit kline response into numpy array [ts, open, high, low, close, volume]."""
    if not klines or not isinstance(klines, dict):
        return None
    try:
        lst = klines.get("result", {}).get("list", [])
        if not lst:
            return None
        arr = np.array([
            [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
            for x in lst
        ])
        # Bybit returns newest first — reverse to chronological order
        return arr[::-1]
    except Exception:
        return None

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    if data is None or len(data) == 0:
        return np.array([])
    out = np.zeros_like(data, dtype=float)
    m   = 2.0 / (period + 1)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = (data[i] - out[i-1]) * m + out[i-1]
    return out

def atr(klines: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    if klines is None or len(klines) < period + 1:
        return 0.0
    highs  = klines[:, 2]
    lows   = klines[:, 3]
    closes = klines[:, 4]
    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr_val = float(np.mean(tr[1:period+1]))
    for i in range(period + 1, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
    return float(atr_val)

def rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    if closes is None or len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses_v = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses_v[:period]))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses_v[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))

def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """MACD line, signal line, histogram."""
    if closes is None or len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_f = ema(closes, fast)
    ema_s = ema(closes, slow)
    macd_line   = ema_f - ema_s
    signal_line = ema(macd_line, signal)
    hist = float(macd_line[-1] - signal_line[-1])
    return float(macd_line[-1]), float(signal_line[-1]), hist

def vwap(klines: np.ndarray, period: int = None) -> float:
    """Volume Weighted Average Price."""
    if klines is None or len(klines) == 0:
        return 0.0
    if period and len(klines) > period:
        klines = klines[-period:]
    tp  = (klines[:, 2] + klines[:, 3] + klines[:, 4]) / 3.0
    vol = klines[:, 5]
    s   = float(np.sum(vol))
    return float(np.sum(tp * vol) / s) if s > 0 else 0.0

def supertrend(klines: np.ndarray, period: int = 10, multiplier: float = 3.0) -> Tuple[float, str]:
    """SuperTrend indicator. Returns (value, direction)."""
    if klines is None or len(klines) < period + 2:
        return 0.0, "NEUTRAL"
    closes = klines[:, 4]
    highs  = klines[:, 2]
    lows   = klines[:, 3]

    atr_val = atr(klines, period)
    hl2 = (highs[-1] + lows[-1]) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    # Simple direction based on close vs bands
    if closes[-1] > upper:
        return float(lower), "BULL"
    elif closes[-1] < lower:
        return float(upper), "BEAR"
    elif closes[-1] > (upper + lower) / 2:
        return float(lower), "BULL"
    else:
        return float(upper), "BEAR"

# ── Data Parsers ─────────────────────────────────────────────────────────────

def p_funding(funding_data) -> float:
    try:
        if not funding_data or not isinstance(funding_data, dict):
            return 0.0
        rate = funding_data.get("result", {}).get("list", [{}])[0].get("fundingRate", "0")
        return float(rate) if rate else 0.0
    except Exception:
        return 0.0

def p_ls(ls_data) -> float:
    try:
        if not ls_data or not isinstance(ls_data, dict):
            return 0.5
        lst = ls_data.get("result", {}).get("list", [])
        if not lst:
            return 0.5
        return float(lst[0].get("buyRatio", "0.5") or "0.5")
    except Exception:
        return 0.5

def p_oi(oi_data) -> float:
    try:
        if not oi_data or not isinstance(oi_data, dict):
            return 0.0
        lst = oi_data.get("result", {}).get("list", [])
        if len(lst) < 2:
            return 0.0
        cur  = float(lst[0].get("openInterest", "0") or "0")
        prev = float(lst[1].get("openInterest", "0") or "0")
        if prev == 0:
            return 0.0
        return (cur - prev) / prev * 100
    except Exception:
        return 0.0

def p_ob(ob_data, depth: int = 50) -> float:
    try:
        if not ob_data or not isinstance(ob_data, dict):
            return 0.0
        result = ob_data.get("result", {})
        bids = result.get("b", [])[:depth]
        asks = result.get("a", [])[:depth]
        bid_vol = sum(float(x[1]) for x in bids) if bids else 0
        ask_vol = sum(float(x[1]) for x in asks) if asks else 0
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
    except Exception:
        return 0.0

def p_liq(liq_data) -> Dict:
    try:
        if not liq_data or not isinstance(liq_data, dict):
            return {"long_vol": 0, "short_vol": 0, "net": 0}
        lst = liq_data.get("result", {}).get("list", [])
        if not lst:
            return {"long_vol": 0, "short_vol": 0, "net": 0}
        long_vol = 0.0
        short_vol = 0.0
        for x in lst[:20]:
            if not isinstance(x, dict):
                continue
            side = x.get("side", "")
            try:
                vol = float(x.get("qty", "0") or "0")
            except Exception:
                vol = 0.0
            if side == "Buy":
                long_vol += vol
            elif side == "Sell":
                short_vol += vol
        return {"long_vol": long_vol, "short_vol": short_vol, "net": long_vol - short_vol}
    except Exception:
        return {"long_vol": 0, "short_vol": 0, "net": 0}

def price_precision(price: float) -> int:
    if price >= 1000:  return 2
    elif price >= 100: return 2
    elif price >= 10:  return 3
    elif price >= 1:   return 4
    elif price >= 0.1: return 5
    elif price >= 0.01: return 6
    else: return 8

__all__ = [
    "VScanner", "MarketScanner", "parse_klines",
    "ema", "atr", "rsi", "macd", "vwap", "supertrend",
    "p_funding", "p_ls", "p_oi", "p_ob", "p_liq", "price_precision",
]
