"""
signals.py — Quantitative Signal Engine
Analyzes multi-timeframe data to generate trading signals and SL/TP levels.
"""
import numpy as np
from typing import Dict, Optional
from scanner import ema, atr, macd, rsi, vwap, parse_klines

class QuantSignalEngine:
    def __init__(self):
        pass

    def analyze(self, sym: str, k3m=None, k5m=None, k15m=None, k1h=None, k4h=None,
                funding: float = 0.0, ls: float = 1.0, oi_pct: float = 0.0, 
                ob_imb: float = 0.0, liqs: Dict = None,
                learned_bias: float = 0.0, mode: str = "NORMAL") -> Dict:
        
        # 1. Calculate Indicators for 5m (Entry TF) and 1h (Trend TF)
        tf_entry = k5m
        tf_trend = k1h
        
        signal = "HOLD"
        confidence = 50.0
        components = {}

        if tf_entry is None or tf_trend is None:
            return {"signal": "HOLD", "confidence": 0, "composite": 0, "components": {}}

        closes_5 = tf_entry[:, 4]
        closes_1 = tf_trend[:, 4]

        # Trend Bias (1H EMA Cross)
        e8_1 = ema(closes_1, 8)
        e21_1 = ema(closes_1, 21)
        trend_bullish = e8_1[-1] > e21_1[-1]
        
        # Momentum (5m RSI + MACD)
        rsi_5 = rsi(closes_5, 14)
        macd_val, macd_sig, macd_hist = macd(closes_5)
        momentum_bullish = rsi_5 < 70 and macd_hist > 0
        
        # Volume Confirmation
        vol_5 = tf_entry[:, 5]
        avg_vol = np.mean(vol_5[:-1])
        last_vol = vol_5[-1]
        vol_confirm = last_vol > avg_vol

        # Composite Score Logic
        score = 0.0
        if trend_bullish: score += 30
        if momentum_bullish: score += 30
        if vol_confirm: score += 20
        score += learned_bias * 10

        # Decision
        if score >= 70:
            signal = "LONG"
            confidence = min(95, 50 + score)
        elif score <= 20:
            signal = "SHORT" # Assuming short logic mirrors long logic inversely
            confidence = min(95, 50 + (100 - score))
        
        # 4H Macro Gate (Optional)
        if k4h is not None:
            closes_4 = k4h[:, 4]
            e8_4 = ema(closes_4, 8)
            e21_4 = ema(closes_4, 21)
            if signal == "LONG" and e8_4[-1] < e21_4[-1]:
                confidence -= 20 # Penalize if 4H is bearish

        components = {
            "macd_5m": macd_hist,
            "cross_3m": 0, # Placeholder if 3m data used
            "rsi_5m": rsi_5,
            "trend_1h": "BULL" if trend_bullish else "BEAR"
        }

        return {
            "signal": signal,
            "confidence": confidence,
            "composite": score,
            "components": components
        }


def sl_tp(side: str, price: float, atr_val: float, k5m=None, k15m=None, 
          sl_mult: float = 1.2, tp_mult: float = 2.4) -> tuple:
    """
    Calculate Stop Loss and Take Profit levels based on ATR and structure.
    """
    if k5m is not None and len(k5m) > 20:
        lows_5 = k5m[-20:, 3]
        highs_5 = k5m[-20:, 2]
        structure_low = np.min(lows_5)
        structure_high = np.max(highs_5)
    else:
        structure_low = price * 0.99
        structure_high = price * 1.01

    if side == "Buy":
        # SL: Max of (ATR based, Structure Low based)
        sl_atr = price - (atr_val * sl_mult)
        sl_struct = structure_low - (price * 0.002) # 0.2% below structure
        sl = max(sl_atr, sl_struct)
        
        # TP
        tp = price + (atr_val * tp_mult)
    else:
        # SL: Min of (ATR based, Structure High based)
        sl_atr = price + (atr_val * sl_mult)
        sl_struct = structure_high + (price * 0.002)
        sl = min(sl_atr, sl_struct)
        
        # TP
        tp = price - (atr_val * tp_mult)

    return sl, tp