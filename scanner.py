"""
scanner.py — Volatility Scanner + Market Data Parsers
Scans ALL Bybit linear perpetuals, scores by volatility composite.
"""
import time, logging, numpy as np
from typing import List, Dict, Optional
from bybit_client import BybitClient

logger   = logging.getLogger(__name__)
MIN_VOL  = 8_000_000      # $8M min daily turnover
BLACKLIST= {
    "USDCUSDT","USDTUSDT","BUSDUSDT","TUSDUSDT","FRAXUSDT","EURUSDT","GBPUSDT",
    # Exclude very high-price assets where micro-account qty < Bybit minimum
    # BTC at $65k: to hit $5.50 notional at 25x needs 0.0034 BTC - often below min lot
    # We keep ETH (workable at 25x), focus on mid/low price perps
}

# Price range that works well for micro accounts:
# Too high (>$5000): qty precision issues | Too low (<$0.00001): qty rounding issues
MICRO_PREFERRED_MAX_PRICE = 5000.0
MICRO_PREFERRED_MIN_PRICE = 0.0001
SCAN_TTL = 900             # 15-minute cache


class VScanner:
    def __init__(self, client: BybitClient):
        self.client       = client
        self.top: List[Dict] = []
        self.last_ts       = 0

    async def scan(self, n: int = 14, force: bool = False) -> List[Dict]:
        now = time.time()
        if not force and self.top and (now - self.last_ts) < SCAN_TTL:
            return self.top

        logger.info("Volatility scan running...")
        raw = await self.client.all_tickers()
        tickers = raw.get("result", {}).get("list", [])
        if not tickers:
            return self.top or _fallback()

        scored = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT") or sym in BLACKLIST: continue
            try:
                vol_24h = float(t.get("turnover24h","0") or 0)
                price   = float(t.get("lastPrice","0") or 0)
                chg     = float(t.get("price24hPcnt","0") or 0)   # decimal
                hi      = float(t.get("highPrice24h","0") or 0)
                lo      = float(t.get("lowPrice24h","0") or 0)
                if price <= 0 or vol_24h < MIN_VOL: continue
                # Skip dust tokens — qty rounding breaks minimum notional
                if price < 0.0001: continue

                rng  = (hi - lo) / price                           # 24h range %
                aabs = abs(chg)
                vf   = min(vol_24h / 1e8, 5.0)
                score= rng * 50 + aabs * 100 * 30 + vf * 20

                scored.append({
                    "symbol":    sym,
                    "price":     price,
                    "vol_score": round(score, 2),
                    "range_pct": round(rng * 100, 2),
                    "chg_24h":   round(chg * 100, 2),
                    "vol_24h_m": round(vol_24h / 1e6, 1),
                    "high_24h":  hi,
                    "low_24h":   lo,
                })
            except Exception: continue

        scored.sort(key=lambda x: x["vol_score"], reverse=True)
        self.top     = scored[:n]
        self.last_ts = now
        top5 = [f"{x['symbol']}({x['range_pct']}%)" for x in self.top[:5]]
        logger.info(f"Top volatile: {top5}")
        return self.top


def _fallback() -> List[Dict]:
    return [
        {"symbol":"BTCUSDT","price":65000,"vol_score":100,"range_pct":4.5,"chg_24h":2.1,"vol_24h_m":5000,"high_24h":66000,"low_24h":63000},
        {"symbol":"ETHUSDT","price":3200, "vol_score":95, "range_pct":5.2,"chg_24h":-3.1,"vol_24h_m":2000,"high_24h":3300,"low_24h":3100},
        {"symbol":"SOLUSDT","price":150,  "vol_score":90, "range_pct":7.1,"chg_24h":5.4,"vol_24h_m":1200,"high_24h":158,"low_24h":143},
        {"symbol":"DOGEUSDT","price":0.15,"vol_score":88, "range_pct":8.9,"chg_24h":-6.2,"vol_24h_m":900,"high_24h":0.16,"low_24h":0.14},
    ]


# ── Kline parsing ─────────────────────────────────────────────────────────────

def parse_klines(raw: Dict) -> Optional[np.ndarray]:
    try:
        items = raw.get("result", {}).get("list", [])
        if not items: return None
        # Bybit returns newest-first; reverse so index 0 = oldest
        arr = np.array([[float(x) for x in row] for row in reversed(items)])
        return arr   # cols: [ts, open, high, low, close, volume, turnover]
    except Exception as e:
        logger.error(f"parse_klines: {e}"); return None


def atr(k: np.ndarray, n: int = 14) -> float:
    if k is None or len(k) < n + 2: return 0.0
    h, lo, c = k[-n-1:,2], k[-n-1:,3], k[-n-1:,4]
    tr = np.maximum(h[1:]-lo[1:], np.maximum(abs(h[1:]-c[:-1]), abs(lo[1:]-c[:-1])))
    return float(tr.mean())


def rsi(closes: np.ndarray, n: int = 14) -> float:
    if len(closes) < n + 1: return 50.0
    d  = np.diff(closes[-(n+1):])
    g  = d[d > 0].mean() if (d > 0).any() else 0.0
    ls = abs(d[d < 0].mean()) if (d < 0).any() else 1e-9
    return 100.0 - 100.0 / (1.0 + g / ls)


def ema(prices: np.ndarray, n: int) -> np.ndarray:
    a   = 2.0 / (n + 1)
    out = np.zeros_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = a * prices[i] + (1 - a) * out[i-1]
    return out


def vwap(k: np.ndarray, n: int = 20) -> float:
    if k is None or len(k) < n: return 0.0
    sl = k[-n:]
    tp = (sl[:,2] + sl[:,3] + sl[:,4]) / 3
    vol= sl[:,5]
    return float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else 0.0


# ── Market structure parsers ──────────────────────────────────────────────────

def p_funding(raw): 
    try: return float((raw.get("result",{}).get("list",[{}])[0] or {}).get("fundingRate",0))
    except: return 0.0

def p_ls(raw):
    """buyRatio / sellRatio"""
    try:
        item = (raw.get("result",{}).get("list",[{}]) or [{}])[0]
        b, s = float(item.get("buyRatio",0.5)), float(item.get("sellRatio",0.5))
        return b / s if s > 0 else 1.0
    except: return 1.0

def p_oi(raw):
    try:
        items = raw.get("result",{}).get("list",[])
        if len(items) >= 2:
            n, o = float(items[0]["openInterest"]), float(items[-1]["openInterest"])
            return (n - o) / o * 100 if o > 0 else 0.0
        return 0.0
    except: return 0.0

def p_ob(raw):
    try:
        bids = raw.get("result",{}).get("b",[])
        asks = raw.get("result",{}).get("a",[])
        bv   = sum(float(b[1]) for b in bids[:15])
        av   = sum(float(a[1]) for a in asks[:15])
        return bv / av if av > 0 else 1.0
    except: return 1.0

def p_liq(raw):
    try:
        items = raw.get("result",{}).get("list",[])
        # "Buy" side = long position liquidated
        long_l  = sum(float(x.get("size",0)) for x in items if x.get("side")=="Buy")
        short_l = sum(float(x.get("size",0)) for x in items if x.get("side")=="Sell")
        return {"long_liq": long_l, "short_liq": short_l}
    except: return {"long_liq":0.0,"short_liq":0.0}


def price_precision(p: float) -> int:
    if p >= 10000: return 0
    if p >= 1000:  return 1
    if p >= 100:   return 2
    if p >= 10:    return 3
    if p >= 1:     return 4
    return 6
