"""
signals.py — 20-Factor Signal Engine
Multi-timeframe: 3m scalp | 5m entry | 15m trend | 1h macro bias
Compound-mode aware: more aggressive in TURBO/AGGRESSIVE modes.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from scanner import rsi, ema, vwap, price_precision
import logging

logger = logging.getLogger(__name__)


class SignalEngine:

    def analyze(
        self,
        sym: str,
        k3m: Optional[np.ndarray],
        k5m: Optional[np.ndarray],
        k15m: Optional[np.ndarray],
        k1h:  Optional[np.ndarray],
        funding: float,
        ls:      float,
        oi_pct:  float,
        ob_imb:  float,
        liqs:    Dict,
        learned_bias: float = 0.0,
        mode:         str   = "NORMAL",
    ) -> Dict:

        scores: Dict[str, float] = {}
        c5  = k5m[:, 4]  if k5m  is not None and len(k5m)  > 55 else None
        c3  = k3m[:, 4]  if k3m  is not None and len(k3m)  > 30 else None
        c15 = k15m[:, 4] if k15m is not None and len(k15m) > 30 else None
        c1h = k1h[:, 4]  if k1h  is not None and len(k1h)  > 20 else None

        # ── 1h Macro Bias ────────────────────────────────────────────────────
        if c1h is not None:
            rsi1h  = rsi(c1h, 14)
            ema21_1h = ema(c1h, 21)
            if   c1h[-1] > ema21_1h[-1] and rsi1h > 50: scores["macro"] =  0.7
            elif c1h[-1] < ema21_1h[-1] and rsi1h < 50: scores["macro"] = -0.7
            else:                                         scores["macro"] =  0.0

        # ── 15m Trend Filter ──────────────────────────────────────────────────
        if c15 is not None:
            ema21_15 = ema(c15, 21)
            ema50_15 = ema(c15, 50)
            rsi15    = rsi(c15, 14)
            p15      = c15[-1]

            if   p15 > ema21_15[-1] > ema50_15[-1]: scores["trend_15"] =  0.9
            elif p15 < ema21_15[-1] < ema50_15[-1]: scores["trend_15"] = -0.9
            else:
                scores["trend_15"] = 0.4 if p15 > ema21_15[-1] else -0.4

            if   rsi15 < 35: scores["rsi_15"] =  0.65
            elif rsi15 > 65: scores["rsi_15"] = -0.65
            else:            scores["rsi_15"] =  0.0

            # 15m momentum (5-candle slope of ema21)
            if len(ema21_15) > 5:
                slope15 = (ema21_15[-1] - ema21_15[-6]) / (ema21_15[-6] + 1e-9)
                scores["momentum_15"] = min(max(slope15 * 300, -0.7), 0.7)

        # ── 5m Core Signals ───────────────────────────────────────────────────
        if c5 is not None:
            e8   = ema(c5, 8);  e21 = ema(c5, 21);  e55 = ema(c5, 55)
            rsi5 = rsi(c5, 14)
            p5   = c5[-1]

            # Triple EMA alignment (strongest signal)
            bull_align = e8[-1] > e21[-1] > e55[-1]
            bear_align = e8[-1] < e21[-1] < e55[-1]
            scores["ema_align"] = 1.0 if bull_align else (-1.0 if bear_align else
                                   (0.4 if e8[-1] > e21[-1] else -0.4))

            # EMA8 slope (momentum quality)
            if len(e8) > 3:
                slp8 = (e8[-1] - e8[-4]) / (e8[-4] + 1e-9)
                scores["ema_slope"] = min(max(slp8 * 600, -0.9), 0.9)

            # RSI
            if   rsi5 < 22: scores["rsi5"] =  1.0
            elif rsi5 < 32: scores["rsi5"] =  0.65
            elif rsi5 < 42: scores["rsi5"] =  0.25
            elif rsi5 > 78: scores["rsi5"] = -1.0
            elif rsi5 > 68: scores["rsi5"] = -0.65
            elif rsi5 > 58: scores["rsi5"] = -0.25
            else:           scores["rsi5"] =  0.0

            # MACD (12/26/9) — crossover is the key event
            e12  = ema(c5, 12); e26 = ema(c5, 26)
            macd = e12 - e26;   sig_line = ema(macd, 9)
            hist = macd[-1] - sig_line[-1]
            if len(macd) > 2:
                prev = macd[-2] - sig_line[-2]
                if   hist > 0 and prev <= 0: scores["macd"] =  0.95   # fresh cross up
                elif hist < 0 and prev >= 0: scores["macd"] = -0.95   # fresh cross dn
                else:                        scores["macd"] =  0.45 if hist > 0 else -0.45
            else:
                scores["macd"] = 0.45 if hist > 0 else -0.45

            # Bollinger Bands (20,2)
            if len(c5) >= 20:
                bb_mid = c5[-20:].mean(); bb_std = c5[-20:].std()
                bb_up  = bb_mid + 2 * bb_std; bb_dn = bb_mid - 2 * bb_std
                if   p5 < bb_dn:          scores["bb"] =  0.85
                elif p5 > bb_up:          scores["bb"] = -0.85
                else:
                    pos = (p5 - bb_dn) / (bb_up - bb_dn + 1e-9) - 0.5
                    scores["bb"] = -pos * 0.35

            # Volume spike (Z-score)
            if k5m.shape[1] > 5 and len(k5m) >= 20:
                vols   = k5m[-20:, 5]
                vz     = (vols[-1] - vols[:-1].mean()) / (vols[:-1].std() + 1e-9)
                tdir   = 1 if e8[-1] > e21[-1] else -1
                scores["vol_spike"] = min(max(vz * 0.22 * tdir, -0.85), 0.85)

            # VWAP
            vw = vwap(k5m, 20)
            if vw > 0:
                dev = (p5 - vw) / vw
                if   dev < -0.006: scores["vwap"] =  0.55
                elif dev > 0.006:  scores["vwap"] = -0.35
                else:              scores["vwap"] =  0.0

            # Price structure: HH+HL vs LL+LH (last 6 candles)
            if len(k5m) >= 6:
                hs = k5m[-6:, 2]; ls2 = k5m[-6:, 3]
                hh = hs[-1] > hs[-3] > hs[-5]
                hl = ls2[-1]> ls2[-3]> ls2[-5]
                ll = ls2[-1]< ls2[-3]< ls2[-5]
                lh = hs[-1] < hs[-3] < hs[-5]
                if   hh and hl:  scores["structure"] =  0.75
                elif ll and lh:  scores["structure"] = -0.75
                else:            scores["structure"] =  0.0

            # Stochastic RSI approximation
            if len(c5) >= 30:
                rsi_vals = np.array([rsi(c5[:i], 14) for i in range(16, len(c5) + 1)])
                if len(rsi_vals) >= 14:
                    rmin, rmax = rsi_vals[-14:].min(), rsi_vals[-14:].max()
                    stoch_k = (rsi_vals[-1] - rmin) / (rmax - rmin + 1e-9) * 100
                    if   stoch_k < 20: scores["stoch_rsi"] =  0.6
                    elif stoch_k > 80: scores["stoch_rsi"] = -0.6
                    else:              scores["stoch_rsi"] =  0.0

        # ── 3m Scalp Confirmation ─────────────────────────────────────────────
        if c3 is not None:
            rsi3   = rsi(c3, 9)
            e5_3m  = ema(c3, 5); e13_3m = ema(c3, 13)

            if   rsi3 < 28: scores["rsi3"] =  0.8
            elif rsi3 > 72: scores["rsi3"] = -0.8
            else:           scores["rsi3"] =  0.0

            # 3m EMA cross
            if len(c3) > 1:
                cross_up   = e5_3m[-1] > e13_3m[-1] and e5_3m[-2] <= e13_3m[-2]
                cross_dn   = e5_3m[-1] < e13_3m[-1] and e5_3m[-2] >= e13_3m[-2]
                if   cross_up: scores["cross_3m"] =  0.95
                elif cross_dn: scores["cross_3m"] = -0.95
                else:          scores["cross_3m"] =  0.35 if e5_3m[-1] > e13_3m[-1] else -0.35

        # ── Microstructure ────────────────────────────────────────────────────

        # Funding rate (contrarian at extremes)
        if   funding >  0.0010: scores["funding"] = -0.85
        elif funding >  0.0003: scores["funding"] = -0.40
        elif funding < -0.0010: scores["funding"] =  0.85
        elif funding < -0.0003: scores["funding"] =  0.40
        else:                   scores["funding"] =  0.0

        # L/S ratio — extreme readings contrarian
        if   ls > 1.6:  scores["ls_ratio"] = -0.75
        elif ls > 1.25: scores["ls_ratio"] = -0.35
        elif ls < 0.62: scores["ls_ratio"] =  0.75
        elif ls < 0.80: scores["ls_ratio"] =  0.35
        else:           scores["ls_ratio"] =  0.0

        # Open interest change
        if   oi_pct > 3.5:  scores["oi"] =  0.55
        elif oi_pct < -3.5: scores["oi"] = -0.55
        else:               scores["oi"] =  oi_pct / 6.5

        # Orderbook imbalance
        if   ob_imb > 2.2:  scores["ob"] =  0.85
        elif ob_imb > 1.45: scores["ob"] =  0.40
        elif ob_imb < 0.45: scores["ob"] = -0.85
        elif ob_imb < 0.69: scores["ob"] = -0.40
        else:               scores["ob"] =  0.0

        # Liquidation cascade
        ll  = liqs.get("long_liq", 0); sl2 = liqs.get("short_liq", 0)
        tot = ll + sl2
        if tot > 0:
            sdm = sl2 / tot
            if   sdm > 0.78: scores["liq"] =  0.95
            elif sdm > 0.60: scores["liq"] =  0.50
            elif sdm < 0.22: scores["liq"] = -0.95
            elif sdm < 0.40: scores["liq"] = -0.50
            else:            scores["liq"] =  0.0
        else:
            scores["liq"] = 0.0

        # Per-symbol learned directional bias
        scores["learned"] = max(-0.55, min(0.55, learned_bias))

        # ── Weighted Composite ─────────────────────────────────────────────────
        W = {
            "ema_align":   2.8,
            "trend_15":    2.2,
            "macro":       1.8,
            "cross_3m":    1.9,
            "ema_slope":   1.6,
            "macd":        1.6,
            "learned":     1.6,
            "rsi5":        1.3,
            "momentum_15": 1.2,
            "structure":   1.2,
            "rsi_15":      1.1,
            "stoch_rsi":   1.0,
            "rsi3":        1.0,
            "liq":         1.4,
            "bb":          0.9,
            "vol_spike":   0.9,
            "ob":          1.0,
            "vwap":        0.7,
            "funding":     0.9,
            "ls_ratio":    0.8,
            "oi":          0.6,
        }
        tw   = sum(W.get(k, 1.0) for k in scores)
        comp = sum(scores[k] * W.get(k, 1.0) for k in scores) / (tw + 1e-9)

        # Confidence = magnitude × signal agreement
        vals = list(scores.values())
        agree = abs(np.mean(vals)) / (np.std(vals) + 0.05)
        conf  = int(min(100, abs(comp) * 90 + agree * 15))

        # Mode-adjusted threshold (compound engine passes this)
        thresh = {"CONSERVATIVE": 0.15, "NORMAL": 0.11, "AGGRESSIVE": 0.09, "TURBO": 0.07}.get(mode, 0.11)

        if   comp >  thresh and conf >= 38: signal = "LONG"
        elif comp < -thresh and conf >= 38: signal = "SHORT"
        else:                               signal = "HOLD"

        return {
            "signal":     signal,
            "confidence": conf,
            "composite":  round(comp, 4),
            "components": {k: round(v, 3) for k, v in scores.items()},
            "rsi_5m":     rsi(c5, 14) if c5 is not None else 50.0,
            "funding":    funding,
            "ls_ratio":   ls,
        }


def sl_tp(side: str, entry: float, atr_val: float, sl_m: float, tp_m: float) -> Tuple[float, float]:
    if atr_val <= 0: atr_val = entry * 0.009
    dp = price_precision(entry)
    sd = atr_val * sl_m; td = atr_val * tp_m
    if side == "Buy":
        return round(entry - sd, dp), round(entry + td, dp)
    else:
        return round(entry + sd, dp), round(entry - td, dp)
