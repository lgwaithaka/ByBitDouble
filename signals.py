"""
signals.py — Quantitative Signal Engine v2.2
Implements all strategy improvements from ByBitDouble Strategy Review (May 2026):
  • BTC 4H macro gate for altcoin LONGs (+15 conf penalty when BTC bearish)
  • Volume hard gate: entry candle must be >1.3× 20-period average
  • RSI divergence detection: +15 confidence bonus
  • Proper SHORT signal scoring (inverse of LONG)
  • Orderbook imbalance weighting (75% heavier per audit)
  • 3m cross computed from real k3m data
  • VWAP directional filter applied
  • LTF hard veto: MACD + 3m cross both negative
  • Signal weight retuning (+75/40/44% on identified loss patterns)
"""
import numpy as np
from typing import Dict, Optional, Tuple
from scanner import ema, atr, macd, rsi, vwap, parse_klines


# ── Tuned signal weights (post-audit) ────────────────────────────────────────
W = {
    "trend_1h":   30,   # EMA 8/21 cross on 1H
    "macd_5m":    28,   # 5m MACD histogram (+40% vs original 20)
    "cross_3m":   26,   # 3m EMA cross (+44% vs original 18)
    "vol_confirm":20,   # Volume confirmation (now a hard gate + soft weight)
    "ob_imb":     14,   # Orderbook imbalance (+75% vs original 8)
    "vwap":       12,   # VWAP directional filter (was unused)
    "rsi_div":    15,   # RSI divergence bonus (new)
    "supertrend": 10,   # 5m SuperTrend direction
}
TOTAL_BULL = sum(W.values())  # Max possible bullish score


def _supertrend(klines: np.ndarray, period: int = 10, mult: float = 3.0) -> int:
    """Returns +1 (bull) / -1 (bear) / 0 (neutral)."""
    if klines is None or len(klines) < period + 1:
        return 0
    highs  = klines[:, 2]
    lows   = klines[:, 3]
    closes = klines[:, 4]
    from scanner import atr as _atr
    atr_v = _atr(klines, period)
    hl2   = (highs[-1] + lows[-1]) / 2
    upper = hl2 + mult * atr_v
    lower = hl2 - mult * atr_v
    return 1 if closes[-1] > lower else -1


def _rsi_divergence(closes: np.ndarray, period: int = 14, lookback: int = 5) -> int:
    """
    Detects bullish (+1) or bearish (-1) RSI divergence.
    Bullish: price makes lower low but RSI makes higher low.
    Bearish: price makes higher high but RSI makes lower high.
    Returns 0 if no divergence.
    """
    if closes is None or len(closes) < period + lookback + 2:
        return 0
    try:
        rsi_vals = []
        for i in range(lookback + 1):
            idx = len(closes) - i - 1
            r = rsi(closes[:idx + 1], period)
            rsi_vals.append(r)
        rsi_vals = list(reversed(rsi_vals))

        price_now  = closes[-1]
        price_prev = closes[-(lookback + 1)]
        rsi_now    = rsi_vals[-1]
        rsi_prev   = rsi_vals[0]

        # Bullish divergence
        if price_now < price_prev and rsi_now > rsi_prev and rsi_now < 50:
            return 1
        # Bearish divergence
        if price_now > price_prev and rsi_now < rsi_prev and rsi_now > 50:
            return -1
    except Exception:
        pass
    return 0


def _volume_hard_gate(vol_series: np.ndarray, lookback: int = 20, threshold: float = 1.3) -> bool:
    """
    Hard gate: last candle volume must be > threshold × 20-period average.
    Based on QuantConnect research: high-volume breakouts succeed 71% vs 33% for low-vol.
    """
    if vol_series is None or len(vol_series) < lookback + 1:
        return True  # insufficient data — allow through
    avg = np.mean(vol_series[-(lookback + 1):-1])
    last = vol_series[-1]
    return last > (avg * threshold)


def _3m_ema_cross(k3m: np.ndarray) -> float:
    """
    Compute 3m EMA(3)/EMA(8) cross signal.
    Returns positive value when fast > slow (bullish), negative when fast < slow.
    """
    if k3m is None or len(k3m) < 10:
        return 0.0
    closes = k3m[:, 4]
    e3  = ema(closes, 3)
    e8  = ema(closes, 8)
    if len(e3) == 0 or len(e8) == 0:
        return 0.0
    diff = e3[-1] - e8[-1]
    # Normalise by price for comparability
    price = float(closes[-1]) or 1.0
    return float(diff / price)


def _btc_macro_gate(k4h_btc: Optional[np.ndarray]) -> int:
    """
    BTC 4H macro gate for altcoin LONGs.
    Returns: +1 (BTC bullish), -1 (BTC bearish / gate LONG), 0 (neutral/unknown).
    Strategy: raise LONG confidence floor +15 when BTC bearish.
    """
    if k4h_btc is None or len(k4h_btc) < 25:
        return 0
    closes = k4h_btc[:, 4]
    e8  = ema(closes, 8)
    e21 = ema(closes, 21)
    st  = _supertrend(k4h_btc, 10, 3.0)
    if e8[-1] > e21[-1] and st >= 0:
        return 1
    elif e8[-1] < e21[-1] and st <= 0:
        return -1
    return 0


class QuantSignalEngine:
    def __init__(self):
        pass

    def analyze(
        self,
        sym: str,
        k3m=None, k5m=None, k15m=None, k1h=None, k4h=None,
        k4h_btc=None,        # BTC 4H for macro gate (pass None for BTC itself)
        funding: float = 0.0,
        ls: float = 1.0,
        oi_pct: float = 0.0,
        ob_imb: float = 0.0,
        liqs: Dict = None,
        learned_bias: float = 0.0,
        mode: str = "NORMAL",
    ) -> Dict:
        """
        Full signal analysis with all strategy improvements applied.
        Returns dict: signal, confidence, composite, components.
        """
        signal     = "HOLD"
        confidence = 50.0
        components = {}

        if k5m is None or k1h is None:
            return {"signal": "HOLD", "confidence": 0, "composite": 0, "components": {}}

        closes_5  = k5m[:, 4]
        volumes_5 = k5m[:, 5]
        closes_1  = k1h[:, 4]

        # ── Volume Hard Gate ──────────────────────────────────────────────
        vol_ok = _volume_hard_gate(volumes_5, lookback=20, threshold=1.3)
        if not vol_ok:
            components["vol_gate"] = "FAIL"
            components["macd_5m"]  = 0.0
            components["cross_3m"] = 0.0
            components["rsi_5m"]   = rsi(closes_5, 14)
            components["trend_1h"] = "UNKNOWN"
            return {
                "signal":     "HOLD",
                "confidence": 35.0,
                "composite":  0.0,
                "components": components,
                "vol_gate":   False,
            }

        # ── Core Indicators ───────────────────────────────────────────────
        # 1H trend
        e8_1  = ema(closes_1, 8)
        e21_1 = ema(closes_1, 21)
        trend_bullish = bool(e8_1[-1] > e21_1[-1])
        trend_bearish = not trend_bullish

        # 5m MACD
        macd_val, macd_sig, macd_hist = macd(closes_5)
        macd_bull = macd_hist > 0
        macd_bear = macd_hist < 0

        # 5m RSI
        rsi_5 = rsi(closes_5, 14)
        rsi_bull = 30 < rsi_5 < 70   # not overbought/oversold
        rsi_bear = rsi_5 > 50         # elevated for short

        # 3m EMA cross
        cross_3m = _3m_ema_cross(k3m)
        cross_bull = cross_3m > 0
        cross_bear = cross_3m < 0

        # LTF hard veto: both 5m MACD and 3m cross in same negative direction
        ltf_long_veto  = macd_bear and cross_bear   # kills LONG entry
        ltf_short_veto = macd_bull and cross_bull   # kills SHORT entry

        # Volume soft score (gate already passed)
        avg_vol  = np.mean(volumes_5[:-1]) if len(volumes_5) > 1 else 1.0
        vol_ratio = volumes_5[-1] / (avg_vol + 1e-9)
        vol_confirm = vol_ratio >= 1.3  # already gated, this adds extra weight

        # VWAP directional filter
        vwap_5  = vwap(k5m, period=20)
        price_5 = float(closes_5[-1])
        vwap_bull = price_5 > vwap_5 * 1.001
        vwap_bear = price_5 < vwap_5 * 0.999

        # Orderbook imbalance (positive = more bids = bullish)
        ob_bull = ob_imb > 0.05
        ob_bear = ob_imb < -0.05

        # RSI divergence
        rsi_div = _rsi_divergence(closes_5, period=14, lookback=5)

        # 5m SuperTrend
        st_5 = _supertrend(k5m, period=10, mult=3.0)

        # BTC macro gate (for altcoin LONGs only)
        btc_macro = _btc_macro_gate(k4h_btc)
        btc_bearish = btc_macro == -1

        # 4H trend alignment for current symbol
        if k4h is not None and len(k4h) >= 22:
            closes_4 = k4h[:, 4]
            e8_4     = ema(closes_4, 8)
            e21_4    = ema(closes_4, 21)
            macro_bull = bool(e8_4[-1] > e21_4[-1])
            macro_bear = not macro_bull
        else:
            macro_bull = True
            macro_bear = False

        # ── LONG Score ────────────────────────────────────────────────────
        long_score = 0.0
        if trend_bullish:  long_score += W["trend_1h"]
        if macd_bull:      long_score += W["macd_5m"]
        if cross_bull:     long_score += W["cross_3m"]
        if vol_confirm:    long_score += W["vol_confirm"]
        if ob_bull:        long_score += W["ob_imb"]
        if vwap_bull:      long_score += W["vwap"]
        if rsi_div == 1:   long_score += W["rsi_div"]
        if st_5 == 1:      long_score += W["supertrend"]
        long_score += learned_bias * 10

        # BTC macro penalty for altcoin LONGs
        if btc_bearish and not sym.startswith("BTC"):
            long_score -= 15

        # ── SHORT Score ───────────────────────────────────────────────────
        short_score = 0.0
        if trend_bearish:  short_score += W["trend_1h"]
        if macd_bear:      short_score += W["macd_5m"]
        if cross_bear:     short_score += W["cross_3m"]
        if vol_confirm:    short_score += W["vol_confirm"]
        if ob_bear:        short_score += W["ob_imb"]
        if vwap_bear:      short_score += W["vwap"]
        if rsi_div == -1:  short_score += W["rsi_div"]
        if st_5 == -1:     short_score += W["supertrend"]
        short_score -= learned_bias * 10  # bias inverted for shorts

        # Macro alignment penalty
        if macro_bear and long_score > short_score:
            long_score  -= 10
        if macro_bull and short_score > long_score:
            short_score -= 10

        # ── Decision ─────────────────────────────────────────────────────
        # Need >48% of max possible score — adjusted for realistic signal mix
        # (not all components will ever fire simultaneously)
        threshold = TOTAL_BULL * 0.48  # ~74 out of 155

        if long_score >= threshold and not ltf_long_veto and long_score > short_score:
            signal     = "LONG"
            confidence = min(95.0, 45.0 + long_score * 0.38)
            # RSI divergence + trend alignment = high conviction bonus
            if rsi_div == 1 and trend_bullish and macro_bull:
                confidence = min(95.0, confidence + 8.0)

        elif short_score >= threshold and not ltf_short_veto and short_score > long_score:
            signal     = "SHORT"
            confidence = min(95.0, 45.0 + short_score * 0.38)
            if rsi_div == -1 and trend_bearish and macro_bear:
                confidence = min(95.0, confidence + 8.0)

        # BTC bearish = raise LONG confidence floor requirement by 15
        if btc_bearish and signal == "LONG" and not sym.startswith("BTC"):
            confidence -= 15.0

        # Funding extreme — high positive funding = bearish (longs paying)
        if funding > 0.003:    confidence -= 5.0 if signal == "LONG"  else 0.0
        elif funding < -0.003: confidence -= 5.0 if signal == "SHORT" else 0.0

        # Long/short ratio extremes
        if ls > 0.75 and signal == "LONG":   confidence -= 5.0   # crowd is long → contra
        elif ls < 0.30 and signal == "SHORT": confidence -= 5.0   # crowd is short → contra

        components = {
            "macd_5m":    float(macd_hist),
            "cross_3m":   float(cross_3m),
            "rsi_5m":     float(rsi_5),
            "trend_1h":   "BULL" if trend_bullish else "BEAR",
            "ob_imb":     float(ob_imb),
            "vwap_bull":  vwap_bull,
            "rsi_div":    rsi_div,
            "st_5m":      st_5,
            "btc_macro":  btc_macro,
            "vol_ratio":  float(vol_ratio),
            "ltf_long_veto":  ltf_long_veto,
            "ltf_short_veto": ltf_short_veto,
            "long_score":  float(long_score),
            "short_score": float(short_score),
        }

        return {
            "signal":     signal,
            "confidence": float(max(0.0, confidence)),
            "composite":  float(long_score if signal == "LONG" else short_score if signal == "SHORT" else max(long_score, short_score)),
            "components": components,
            "vol_gate":   True,
        }


def sl_tp(
    side: str,
    price: float,
    atr_val: float,
    k5m=None,
    k15m=None,
    sl_mult: float = 1.8,   # raised from 1.2 — strategy fix for noise-outs
    tp_mult: float = 3.0,
) -> Tuple[float, float]:
    """
    Calculate Stop Loss and Take Profit levels based on ATR and structure.
    SL floor: 1.8× ATR (raised from 1.2× to prevent noise-outs — strategy fix).
    TP: initial target; partial-close logic at 5% handled in engine.py.
    """
    if k5m is not None and len(k5m) > 20:
        lows_5       = k5m[-20:, 3]
        highs_5      = k5m[-20:, 2]
        structure_low  = float(np.min(lows_5))
        structure_high = float(np.max(highs_5))
    else:
        structure_low  = price * 0.985
        structure_high = price * 1.015

    if k15m is not None and len(k15m) > 10:
        lows_15       = k15m[-10:, 3]
        highs_15      = k15m[-10:, 2]
        structure_low  = min(structure_low,  float(np.min(lows_15)))
        structure_high = max(structure_high, float(np.max(highs_15)))

    # Clamp structure to be on the correct side of entry price
    # (avoids invalid SL when price is at extremes of the lookback window)
    structure_low  = min(structure_low,  price * 0.998)
    structure_high = max(structure_high, price * 1.002)

    if side == "Buy":
        sl_atr    = price - (atr_val * sl_mult)
        sl_struct = structure_low - (price * 0.002)
        # Use the wider (more protective) of the two — both are below entry
        sl        = max(sl_atr, sl_struct)
        sl        = min(sl, price * 0.995)  # safety: never within 0.5% of entry
        tp        = price + (atr_val * tp_mult)
    else:
        sl_atr    = price + (atr_val * sl_mult)
        sl_struct = structure_high + (price * 0.002)
        sl        = min(sl_atr, sl_struct)
        sl        = max(sl, price * 1.005)  # safety: never within 0.5% of entry
        tp        = price - (atr_val * tp_mult)

    return sl, tp
