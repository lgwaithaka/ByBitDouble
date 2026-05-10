"""
scanner.py — Volume-based market scanner for ByBit
Scans for high-volume, high-volatility opportunities
"""
import asyncio, logging, time, math
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class VScanner:
    """Volume scanner that identifies top movers by volume and volatility."""
    
    def __init__(self, client):
        self.client = client
        self.top: List[Dict] = []
        self.last_scan = 0
        self.symbols_pool = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
            "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
            "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "PEPEUSDT",
            "SHIBUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT", "STRKUSDT",
        ]
    
    async def scan(self, n: int = 12, force: bool = False) -> List[Dict]:
        """
        Scan top symbols by volume and volatility.
        Returns list of dicts: {symbol, vol_score, range_pct, chg_24h}
        """
        if not force and time.time() - self.last_scan < 60:
            return self.top[:n]
        
        results = []
        for sym in self.symbols_pool:
            try:
                data = await self._fetch_symbol_metrics(sym)
                if data:
                    results.append(data)
            except Exception as e:
                logger.debug(f"Scan {sym}: {e}")
        
        # Sort by volume score (highest first)
        results.sort(key=lambda x: x.get("vol_score", 0), reverse=True)
        self.top = results
        self.last_scan = time.time()
        
        return results[:n]
    
    async def _fetch_symbol_metrics(self, symbol: str) -> Optional[Dict]:
        """Fetch volume, volatility, and 24h change for a symbol."""
        try:
            # Get klines
            klines = await self.client.klines(symbol, "60", 25)  # 1h, 25 candles
            if not klines:
                return None
            
            k = parse_klines(klines)
            if k is None or len(k) < 24:
                return None
            
            closes = k[:, 4]
            highs = k[:, 2]
            lows = k[:, 3]
            volumes = k[:, 5]
            
            # Volume score (recent vs average)
            vol_recent = np.mean(volumes[-6:])  # Last 6 hours
            vol_avg = np.mean(volumes[:-6]) if len(volumes) > 6 else vol_recent
            vol_score = vol_recent / (vol_avg + 1e-9)
            
            # Range/volatility
            price = float(closes[-1])
            range_pct = (highs[-24:].max() - lows[-24:].min()) / price if price > 0 else 0
            
            # 24h change
            chg_24h = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 and closes[-24] > 0 else 0
            
            return {
                "symbol": symbol,
                "vol_score": float(vol_score),
                "range_pct": float(range_pct),
                "chg_24h": float(chg_24h),
                "price": price,
                "volume_24h": float(np.sum(volumes[-24:]))
            }
        except Exception as e:
            logger.debug(f"Metrics {symbol}: {e}")
            return None


def parse_klines(klines) -> Optional[np.ndarray]:
    """
    Parse Bybit klines response into numpy array.
    Returns array with columns: [timestamp, open, high, low, close, volume]
    """
    if not klines or not isinstance(klines, dict):
        return None
    
    try:
        lst = klines.get("result", {}).get("list", [])
        if not lst:
            return None
        
        # Bybit format: [timestamp, open, high, low, close, volume, turnover]
        arr = np.array([
            [
                float(x[0]),  # timestamp
                float(x[1]),  # open
                float(x[2]),  # high
                float(x[3]),  # low
                float(x[4]),  # close
                float(x[5]),  # volume
            ]
            for x in lst
        ])
        return arr
    except Exception:
        return None


def atr(klines: np.ndarray, period: int = 14) -> float:
    """
    Calculate Average True Range.
    klines: array with columns [ts, open, high, low, close, volume]
    """
    if klines is None or len(klines) < period + 1:
        return 0.0
    
    highs = klines[:, 2]
    lows = klines[:, 3]
    closes = klines[:, 4]
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_closes = np.roll(closes, 1)
    tr1 = highs - lows
    tr2 = np.abs(highs - prev_closes)
    tr3 = np.abs(lows - prev_closes)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR = EMA of TR
    atr_vals = []
    atr_val = np.mean(tr[:period])
    atr_vals.append(atr_val)
    
    for i in range(period, len(tr)):
        atr_val = (atr_vals[-1] * (period - 1) + tr[i]) / period
        atr_vals.append(atr_val)
    
    return float(atr_vals[-1]) if atr_vals else 0.0


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    if data is None or len(data) == 0:
        return np.array([])
    
    ema_vals = np.zeros_like(data)
    multiplier = 2.0 / (period + 1)
    ema_vals[0] = data[0]
    
    for i in range(1, len(data)):
        ema_vals[i] = (data[i] - ema_vals[i-1]) * multiplier + ema_vals[i-1]
    
    return ema_vals


def p_funding(funding_data) -> float:
    """Parse funding rate from Bybit response."""
    try:
        if not funding_data:
            return 0.0
        rate = funding_data.get("result", {}).get("list", [{}])[0].get("fundingRate", "0")
        return float(rate) if rate else 0.0
    except Exception:
        return 0.0


def p_ls(ls_data) -> float:
    """Parse long/short ratio from Bybit response."""
    try:
        if not ls_data:
            return 1.0
        lst = ls_data.get("result", {}).get("list", [])
        if not lst:
            return 1.0
        # Return ratio of long accounts to total
        long_acc = float(lst[0].get("longAccount", "0.5") or "0.5")
        return long_acc
    except Exception:
        return 1.0


def p_oi(oi_data) -> float:
    """Parse open interest change from Bybit response."""
    try:
        if not oi_data:
            return 0.0
        lst = oi_data.get("result", {}).get("list", [])
        if len(lst) < 2:
            return 0.0
        oi_current = float(lst[0].get("openInterest", "0") or "0")
        oi_prev = float(lst[1].get("openInterest", "0") or "0")
        if oi_prev == 0:
            return 0.0
        return (oi_current - oi_prev) / oi_prev * 100
    except Exception:
        return 0.0


def p_ob(ob_data, depth: int = 50) -> float:
    """
    Parse order book imbalance.
    Returns: (bids - asks) / (bids + asks)
    Positive = bullish imbalance, Negative = bearish imbalance
    """
    try:
        if not ob_data:
            return 0.0
        result = ob_data.get("result", {})
        bids = result.get("b", [])[:depth]  # Top bids
        asks = result.get("a", [])[:depth]  # Top asks
        
        bid_vol = sum(float(x[1]) for x in bids) if bids else 0
        ask_vol = sum(float(x[1]) for x in asks) if asks else 0
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    except Exception:
        return 0.0


def p_liq(liq_data) -> Dict:
    """
    Parse liquidation data.
    Returns dict with long_liq_vol, short_liq_vol, net_liq
    """
    try:
        if not liq_data:
            return {"long_vol": 0, "short_vol": 0, "net": 0}
        
        lst = liq_data.get("result", {}).get("list", [])
        long_vol = 0.0
        short_vol = 0.0
        
        for x in lst[:20]:  # Last 20 liquidations
            side = x.get("side", "")
            vol = float(x.get("value", "0") or "0")
            if side == "Buy":  # Long liquidation (forced buy)
                long_vol += vol
            elif side == "Sell":  # Short liquidation (forced sell)
                short_vol += vol
        
        return {
            "long_vol": long_vol,
            "short_vol": short_vol,
            "net": long_vol - short_vol
        }
    except Exception:
        return {"long_vol": 0, "short_vol": 0, "net": 0}


def price_precision(price: float) -> int:
    """Determine price precision based on price level."""
    if price >= 1000:
        return 2
    elif price >= 100:
        return 2
    elif price >= 10:
        return 3
    elif price >= 1:
        return 4
    elif price >= 0.1:
        return 5
    elif price >= 0.01:
        return 6
    else:
        return 8


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """
    Calculate MACD indicator.
    Returns: (macd_line, signal_line, histogram)
    """
    if closes is None or len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA of MACD
    signal_line = ema(macd_line[-signal*2:], signal)  # Use last 2*signal periods
    
    histogram = macd_line[-1] - signal_line[-1] if len(signal_line) > 0 else 0.0
    
    return float(macd_line[-1]), float(signal_line[-1] if len(signal_line) > 0 else 0), float(histogram)


def rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if closes is None or len(closes) < period + 1:
        return 50.0
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    
    return float(rsi_val)


# Export all public functions
__all__ = [
    "VScanner",
    "parse_klines",
    "atr",
    "ema",
    "p_funding",
    "p_ls",
    "p_oi",
    "p_ob",
    "p_liq",
    "price_precision",
    "macd",
    "rsi",
]