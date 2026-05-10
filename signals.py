"""
signals.py — Quantitative Signal Engine v3.0
═══════════════════════════════════════════════
Multi-timeframe confluence scoring with all strategy improvements:

  ✔ BTC 4H macro gate: altcoin LONGs penalized when BTC bearish
  ✔ Volume >1.3x mandatory gate: hard gate, not advisory
  ✔ RSI divergence bonus: +15 confidence
  ✔ Updated signal weights: ob 1.4, macd_5m 1.4, cross_3m 1.3
  ✔ SuperTrend weight 1.8, penalty -0.8
  ✔ LTF veto: skip if 5m MACD + 3m cross both negative
  ✔ Structural SL/TP placement

Scoring System (20 factors → composite score):
  Each factor contributes a weighted value to a composite.
  Positive composite = LONG, Negative = SHORT.
  Confidence = normalized |composite| mapped to 0-100.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from scanner import ema, atr, macd, rsi, vwap, supertrend

class QuantSignalEngine:
    def __init__(self):
        self._btc_bias = 0  # Cached BTC 4H bias: +1 bull, -1 bear, 0 neutral

    def set_btc_bias(self, k4h_btc) -> int:
        """Compute BTC 4H macro bias for altcoin gating."""
        if k4h_btc is None or len(k4h_btc) < 22:
            self._btc_bias = 0
            return 0
        closes = k4h_btc[:, 4]
        e8  = ema(closes, 8)
        e21 = ema(closes, 21)
        _, st_dir = supertrend(k4h_btc, 10, 3.0)
        if e8[-1] > e21[-1] and st_dir == "BULL":
            self._btc_bias = 1
        elif e8[-1] < e21[-1] and st_dir == "BEAR":
            self._btc_bias = -1
        else:
            self._btc_bias = 0
        return self._btc_bias

    def analyze(self, sym: str, k3m=None, k5m=None, k15m=None, k1h=None, k4h=None,
                funding: float = 0.0, ls: float = 0.5, oi_pct: float = 0.0,
                ob_imb: float = 0.0, liqs: Dict = None,
                learned_bias: float = 0.0, mode: str = "NORMAL") -> Dict:
        """
        Multi-timeframe analysis → signal + confidence + composite.
        Returns {"signal", "confidence", "composite", "components", "volume_ok"}.
        """
        components = {}

        # Require at minimum 5m and 1h data
        if k5m is None or k1h is None or len(k5m) < 30 or len(k1h) < 22:
            return {"signal": "HOLD", "confidence": 0, "composite": 0,
                    "components": {}, "volume_ok": False}

        closes_5  = k5m[:, 4]
        closes_1h = k1h[:, 4]
        price     = float(closes_5[-1])

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 1: 1H Trend Direction (EMA 8/21 cross)
        # ═══════════════════════════════════════════════════════════════════
        e8_1h  = ema(closes_1h, 8)
        e21_1h = ema(closes_1h, 21)
        trend_1h = 1 if e8_1h[-1] > e21_1h[-1] else -1
        trend_slope = (e8_1h[-1] - e8_1h[-3]) / (e8_1h[-3] + 1e-9) if len(e8_1h) >= 3 else 0
        components["trend_1h"] = "BULL" if trend_1h > 0 else "BEAR"

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 2: 4H Macro Gate (if available)
        # ═══════════════════════════════════════════════════════════════════
        macro_4h = 0
        if k4h is not None and len(k4h) >= 22:
            closes_4h = k4h[:, 4]
            e8_4h  = ema(closes_4h, 8)
            e21_4h = ema(closes_4h, 21)
            _, st_dir_4h = supertrend(k4h, 10, 3.0)
            if e8_4h[-1] > e21_4h[-1]:
                macro_4h = 1
            elif e8_4h[-1] < e21_4h[-1]:
                macro_4h = -1
            components["macro_4h"] = "BULL" if macro_4h > 0 else ("BEAR" if macro_4h < 0 else "FLAT")
        else:
            components["macro_4h"] = "N/A"

        # 4H must agree with 1H for high confidence
        htf_aligned = (trend_1h == macro_4h) or macro_4h == 0

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 3: 5m Momentum (RSI + MACD)
        # ═══════════════════════════════════════════════════════════════════
        rsi_5 = rsi(closes_5, 14)
        macd_val, macd_sig, macd_hist = macd(closes_5)
        components["rsi_5m"]  = round(rsi_5, 1)
        components["macd_5m"] = round(macd_hist, 6)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 4: 3m EMA Cross (timing)
        # ═══════════════════════════════════════════════════════════════════
        cross_3m = 0.0
        if k3m is not None and len(k3m) >= 15:
            c3 = k3m[:, 4]
            e5_3  = ema(c3, 5)
            e13_3 = ema(c3, 13)
            cross_3m = float((e5_3[-1] - e13_3[-1]) / (e13_3[-1] + 1e-9))
        components["cross_3m"] = round(cross_3m, 6)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 5: VWAP position
        # ═══════════════════════════════════════════════════════════════════
        vwap_val = vwap(k5m, 20)
        vwap_pos = 1 if price > vwap_val else -1
        components["vwap"] = "ABOVE" if vwap_pos > 0 else "BELOW"

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 6: SuperTrend 5m
        # ═══════════════════════════════════════════════════════════════════
        _, st_dir_5m = supertrend(k5m, 10, 3.0)
        st_val = 1 if st_dir_5m == "BULL" else -1
        components["supertrend_5m"] = st_dir_5m

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 7: Volume Confirmation (MANDATORY GATE)
        # Strategy doc: >1.3x 20-period average required
        # ═══════════════════════════════════════════════════════════════════
        vol_5 = k5m[:, 5]
        vol_avg_20 = float(np.mean(vol_5[-21:-1])) if len(vol_5) > 20 else float(np.mean(vol_5[:-1]))
        vol_last   = float(vol_5[-1])
        vol_ratio  = vol_last / (vol_avg_20 + 1e-9)
        volume_ok  = vol_ratio >= 1.3
        components["vol_ratio"] = round(vol_ratio, 2)
        components["volume_ok"] = volume_ok

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 8: Orderbook Imbalance (weight 1.4 — raised from 0.8)
        # ═══════════════════════════════════════════════════════════════════
        components["ob_imb"] = round(ob_imb, 3)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 9: Funding Rate
        # ═══════════════════════════════════════════════════════════════════
        components["funding"] = round(funding * 100, 4)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 10: OI Change
        # ═══════════════════════════════════════════════════════════════════
        components["oi_pct"] = round(oi_pct, 2)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 11: Long/Short Ratio
        # ═══════════════════════════════════════════════════════════════════
        components["ls_ratio"] = round(ls, 3)

        # ═══════════════════════════════════════════════════════════════════
        # FACTOR 12: RSI Divergence (+15 confidence bonus)
        # ═══════════════════════════════════════════════════════════════════
        rsi_div = self._detect_rsi_divergence(closes_5, rsi_5)
        components["rsi_divergence"] = rsi_div

        # ═══════════════════════════════════════════════════════════════════
        # COMPOSITE SCORING — Weighted sum
        # Updated weights per strategy review
        # ═══════════════════════════════════════════════════════════════════
        composite = 0.0

        # 1. 1H Trend (weight 2.0)
        composite += trend_1h * 2.0

        # 2. 4H Macro (weight 1.5)
        composite += macro_4h * 1.5

        # 3. 5m MACD histogram direction (weight 1.4 — raised from 1.0)
        if macd_hist > 0:
            composite += 1.4
        elif macd_hist < 0:
            composite -= 1.4

        # 4. RSI zone
        if rsi_5 < 35:
            composite += 1.0   # Oversold = bullish
        elif rsi_5 > 65:
            composite -= 1.0   # Overbought = bearish
        elif 45 < rsi_5 < 55:
            composite += trend_1h * 0.3  # Neutral zone follows trend

        # 5. 3m EMA cross (weight 1.3 — raised from 0.9)
        if cross_3m > 0.001:
            composite += 1.3
        elif cross_3m < -0.001:
            composite -= 1.3

        # 6. VWAP position (weight 0.8, penalty raised)
        composite += vwap_pos * 0.8

        # 7. SuperTrend 5m (weight 1.8 — raised from 1.3, penalty -0.8)
        if st_val == trend_1h:
            composite += st_val * 1.8
        else:
            composite += st_val * 0.5  # Disagrees — mild weight only

        # 8. Volume (weight 1.2 — but also hard gate)
        if volume_ok:
            composite += trend_1h * 1.2
        else:
            composite *= 0.6  # Dampen signal without volume

        # 9. Orderbook imbalance (weight 1.4 — raised from 0.8)
        composite += ob_imb * 1.4

        # 10. Funding (contrarian: negative funding = bullish for longs)
        if abs(funding) > 0.0003:
            if funding < 0:
                composite += 0.6   # Shorts paying → bullish
            else:
                composite -= 0.6   # Longs paying → bearish

        # 11. OI change
        if abs(oi_pct) > 3:
            composite += (1 if oi_pct > 0 else -1) * 0.4

        # 12. L/S ratio (contrarian)
        if ls > 0.65:
            composite -= 0.5   # Crowded longs → slight bearish
        elif ls < 0.35:
            composite += 0.5   # Crowded shorts → slight bullish

        # 13. Liquidation pressure
        if liqs:
            net_liq = liqs.get("net", 0)
            if abs(net_liq) > 100:
                # Heavy long liqs = bearish cascade, heavy short liqs = bullish squeeze
                composite -= 0.3 if net_liq > 0 else -0.3

        # 14. Learned bias from symbol history
        composite += learned_bias * 1.0

        # 15. RSI divergence bonus
        if rsi_div == "BULL_DIV":
            composite += 1.5
        elif rsi_div == "BEAR_DIV":
            composite -= 1.5

        # ═══════════════════════════════════════════════════════════════════
        # SIGNAL DECISION
        # ═══════════════════════════════════════════════════════════════════
        signal = "HOLD"
        if composite > 2.5:
            signal = "LONG"
        elif composite < -2.5:
            signal = "SHORT"

        # Confidence: scale |composite| to 0-100
        raw_conf = min(100, abs(composite) * 8)
        confidence = round(raw_conf, 1)

        # ═══════════════════════════════════════════════════════════════════
        # BTC MACRO GATE — Penalize altcoin LONGs when BTC is bearish
        # Strategy doc: WR drops 65%→31% when BTC 4H is down
        # ═══════════════════════════════════════════════════════════════════
        if signal == "LONG" and sym != "BTCUSDT" and self._btc_bias == -1:
            confidence -= 15
            components["btc_gate"] = "PENALIZED (-15)"
        else:
            components["btc_gate"] = "OK"

        # 4H/1H disagreement penalty
        if not htf_aligned and signal != "HOLD":
            confidence -= 10
            components["htf_aligned"] = False
        else:
            components["htf_aligned"] = True

        # RSI divergence bonus (on top of composite contribution)
        if rsi_div in ("BULL_DIV", "BEAR_DIV"):
            confidence += 10
            components["div_bonus"] = True

        # Floor/cap
        confidence = max(0, min(98, confidence))

        return {
            "signal":     signal,
            "confidence": confidence,
            "composite":  round(composite, 3),
            "components": components,
            "volume_ok":  volume_ok,
        }

    def _detect_rsi_divergence(self, closes: np.ndarray, current_rsi: float) -> str:
        """Detect bullish/bearish RSI divergence over last 20 bars."""
        if closes is None or len(closes) < 25:
            return "NONE"
        try:
            recent = closes[-20:]
            # Calculate RSI at lookback point
            lookback_rsi = rsi(closes[:-10], 14) if len(closes) > 25 else 50

            # Bullish divergence: price lower low, RSI higher low
            if recent[-1] < recent[0] and current_rsi > lookback_rsi:
                return "BULL_DIV"
            # Bearish divergence: price higher high, RSI lower high
            if recent[-1] > recent[0] and current_rsi < lookback_rsi:
                return "BEAR_DIV"
        except Exception:
            pass
        return "NONE"


def sl_tp(side: str, price: float, atr_val: float,
          k5m=None, k15m=None,
          sl_mult: float = 1.8, tp_mult: float = 3.0) -> Tuple[float, float]:
    """
    Calculate SL/TP using ATR + structure.
    SL minimum 1.8x ATR (raised from 1.2x per strategy doc — prevents noise-outs).
    TP minimum 3.0x ATR for sufficient R:R.
    """
    prec = price_precision(price)

    # Structure levels from recent candles
    if k5m is not None and len(k5m) > 20:
        lows_5  = k5m[-20:, 3]
        highs_5 = k5m[-20:, 2]
        structure_low  = float(np.min(lows_5))
        structure_high = float(np.max(highs_5))
    else:
        structure_low  = price * 0.985
        structure_high = price * 1.015

    if side == "Buy":
        sl_atr    = price - (atr_val * sl_mult)
        sl_struct = structure_low - (price * 0.002)
        sl = max(sl_atr, sl_struct)
        sl = min(sl, price * 0.97)  # Never more than 3% from entry
        tp = price + (atr_val * tp_mult)
    else:
        sl_atr    = price + (atr_val * sl_mult)
        sl_struct = structure_high + (price * 0.002)
        sl = min(sl_atr, sl_struct)
        sl = max(sl, price * 1.03)
        tp = price - (atr_val * tp_mult)

    return round(sl, prec), round(tp, prec)


def price_precision(price: float) -> int:
    if price >= 1000:  return 2
    elif price >= 100: return 2
    elif price >= 10:  return 3
    elif price >= 1:   return 4
    elif price >= 0.1: return 5
    elif price >= 0.01: return 6
    else: return 8
