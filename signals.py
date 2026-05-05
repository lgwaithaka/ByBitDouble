"""
signals.py — Research-Backed Quant Signal Engine v3
Built on: TrendRider (67.9% WR), QuantPedia D1/H1 study, Medium AI bot (60%+ WR)
Core: Fewer trades, better trades. Hard MTF gates before any signal fires.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from scanner import rsi, ema, vwap, price_precision
import logging

logger = logging.getLogger(__name__)


def calc_adx(k, period=14):
    if k is None or len(k) < period * 2: return 0.0
    try:
        h=k[:,2]; lo=k[:,3]; c=k[:,4]
        tr=np.maximum(h[1:]-lo[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(lo[1:]-c[:-1])))
        dmp=np.where((h[1:]-h[:-1])>(lo[:-1]-lo[1:]),np.maximum(h[1:]-h[:-1],0),0.0)
        dmn=np.where((lo[:-1]-lo[1:])>(h[1:]-h[:-1]),np.maximum(lo[:-1]-lo[1:],0),0.0)
        def ws(a,n):
            o=np.zeros(len(a)); o[n-1]=a[:n].sum()
            for i in range(n,len(a)): o[i]=o[i-1]-o[i-1]/n+a[i]
            return o/n
        atr14=ws(tr,period); dmp14=ws(dmp,period); dmn14=ws(dmn,period)
        dip=100*dmp14/(atr14+1e-9); din=100*dmn14/(atr14+1e-9)
        dx=100*np.abs(dip-din)/(dip+din+1e-9)
        return float(ws(dx,period)[-1])
    except: return 0.0


def calc_supertrend(k, period=10, mult=3.0):
    if k is None or len(k) < period+2: return 0
    try:
        h=k[:,2]; lo=k[:,3]; c=k[:,4]; hl2=(h+lo)/2
        tr=np.maximum(h[1:]-lo[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(lo[1:]-c[:-1])))
        atr=np.zeros(len(c)); atr[1]=tr[0]
        for i in range(2,len(tr)+1): atr[i]=(atr[i-1]*(period-1)+tr[i-1])/period
        upper=hl2+mult*atr; lower=hl2-mult*atr
        d=np.ones(len(c))
        for i in range(1,len(c)):
            if c[i]>upper[i-1]: d[i]=1
            elif c[i]<lower[i-1]: d[i]=-1
            else: d[i]=d[i-1]
        return int(d[-1])
    except: return 0


def calc_macd(closes):
    e12=ema(closes,12); e26=ema(closes,26); m=e12-e26; s=ema(m,9)
    return float(m[-1]),float(s[-1]),float(m[-1]-s[-1])


def funding_signal(rate):
    """Correct: positive funding = crowded long = SHORT edge. Negative = SHORT crowded = LONG edge."""
    if   rate >  0.003: return -1.0
    elif rate >  0.001: return -0.6
    elif rate >  0.0003: return -0.25
    elif rate < -0.003: return  1.0
    elif rate < -0.001: return  0.6
    elif rate < -0.0003: return  0.25
    return 0.0


def detect_swing_structure(k):
    if k is None or len(k)<20: return -0.02, 0.02
    try:
        price=float(k[-1,4]); highs=k[-30:,2]; lows=k[-30:,3]
        res=[highs[i] for i in range(2,len(highs)-2) if highs[i]>highs[i-1] and highs[i]>highs[i+1]]
        sup=[lows[i]  for i in range(2,len(lows)-2)  if lows[i] <lows[i-1]  and lows[i] <lows[i+1]]
        nr=min((r for r in res if r>price),default=price*1.02)
        ns=max((s for s in sup if s<price),default=price*0.98)
        return (ns-price)/price, (nr-price)/price
    except: return -0.02, 0.02


def calc_volume_confirm(k, n=8):
    if k is None or len(k)<n+2: return 0.5
    try:
        v=k[-n:,5]; c=k[-n:,4]
        slope=(v[-1]-v[0])/(v[0]+1e-9)
        bull=c[-1]>c[0]
        if (bull and slope>0.1) or (not bull and slope>0.1): return 0.8
        if slope<-0.2: return 0.3
        return 0.5
    except: return 0.5


class SignalEngine:
    """
    Hard multi-timeframe gate system.
    Gate 1: 4H macro trend (veto if ranging or disagrees)
    Gate 2: 1H structure must align with 4H (veto if not)
    Gate 3+: 15m/5m/3m scored with high thresholds
    Min confidence: 65. Min composite: 0.35. Result: fewer trades, much higher win rate.
    """

    def analyze(self, sym, k3m, k5m, k15m, k1h, k4h,
                funding, ls, oi_pct, ob_imb, liqs,
                learned_bias=0.0, mode="NORMAL"):

        hold = {"signal":"HOLD","confidence":0,"composite":0.0,
                "components":{},"veto_reason":None,"macro_direction":0}

        # ── GATE 1: 4H MACRO TREND ───────────────────────────────────────
        macro = 0
        adx_4h_val = 0.0

        if k4h is not None and len(k4h) >= 50:
            c4h = k4h[:,4]
            e21_4h = ema(c4h,21); e50_4h = ema(c4h,50)
            st4h   = calc_supertrend(k4h,10,3.0)
            _,_,hist4h = calc_macd(c4h)
            adx_4h_val = calc_adx(k4h,14)
            p4h = float(c4h[-1])

            # ADX < 15 = truly ranging, no directional trades (was 18 — too strict)
            if adx_4h_val < 15:
                hold["veto_reason"] = f"RANGING_ADX4H={adx_4h_val:.1f}<15"
                return hold

            bull4h = p4h > e21_4h[-1] and e21_4h[-1] > e50_4h[-1] and st4h == 1
            bear4h = p4h < e21_4h[-1] and e21_4h[-1] < e50_4h[-1] and st4h == -1

            if   bull4h: macro =  1
            elif bear4h: macro = -1
            elif adx_4h_val > 25:
                macro =  1 if hist4h > 0 else -1
            else:
                hold["veto_reason"] = f"4H_UNCLEAR_ADX={adx_4h_val:.1f}"
                return hold

        elif k1h is not None and len(k1h) >= 50:
            # Fallback: use 1H as macro (less reliable)
            c1h_tmp = k1h[:,4]
            e50_tmp = ema(c1h_tmp,50)
            adx_1h_tmp = calc_adx(k1h,14)
            if adx_1h_tmp < 16:
                hold["veto_reason"] = f"RANGING_ADX1H={adx_1h_tmp:.1f}<16"
                return hold
            macro = 1 if c1h_tmp[-1] > e50_tmp[-1] else -1
        else:
            hold["veto_reason"] = "NO_HIGHER_TF"
            return hold

        # ── GATE 2: 1H STRUCTURE MUST AGREE ─────────────────────────────
        h1_score = 0.0
        if k1h is not None and len(k1h) >= 50:
            c1h = k1h[:,4]
            e21_1h=ema(c1h,21); e50_1h=ema(c1h,50)
            st1h=calc_supertrend(k1h,10,3.0)
            _,_,hist1h=calc_macd(c1h)
            adx1h=calc_adx(k1h,14)
            p1h=float(c1h[-1])

            bull1h = p1h > e21_1h[-1] and st1h == 1 and hist1h > 0
            bear1h = p1h < e21_1h[-1] and st1h == -1 and hist1h < 0

            if macro==1 and bull1h:
                h1_score = 1.0 if (p1h > e50_1h[-1] and adx1h > 25) else 0.7
            elif macro==-1 and bear1h:
                h1_score = 1.0 if (p1h < e50_1h[-1] and adx1h > 25) else 0.7
            else:
                hold["veto_reason"] = f"1H_DISAGREES_4H macro={macro} bull1h={bull1h} bear1h={bear1h}"
                return hold
        else:
            hold["veto_reason"] = "NO_1H_DATA"
            return hold

        # ── GATE 3+: SCORE LOWER TIMEFRAMES ──────────────────────────────
        scores = {}
        scores["htf_alignment"] = float(macro) * h1_score * 2.5

        # 15m
        if k15m is not None and len(k15m) >= 50:
            c15=k15m[:,4]
            e8_15=ema(c15,8); e21_15=ema(c15,21)
            rsi15=rsi(c15,14); st15=calc_supertrend(k15m,10,3.0)
            _,_,hist15=calc_macd(c15)
            adx15=calc_adx(k15m,14); p15=float(c15[-1])

            # EMA
            if macro==1:
                scores["ema_15m"] = 0.9 if (e8_15[-1]>e21_15[-1] and p15>e21_15[-1]) else (0.4 if p15>e21_15[-1] else -0.4)
            else:
                scores["ema_15m"] = 0.9 if (e8_15[-1]<e21_15[-1] and p15<e21_15[-1]) else (0.4 if p15<e21_15[-1] else -0.4)

            # RSI — pullback entry preferred
            if macro==1:
                scores["rsi_15m"] = 1.0 if rsi15<35 else (0.5 if rsi15<50 else (-0.4 if rsi15>70 else 0.2))
            else:
                scores["rsi_15m"] = 1.0 if rsi15>65 else (0.5 if rsi15>50 else (-0.4 if rsi15<30 else 0.2))

            # MACD cross
            if macro==1:
                scores["macd_15m"] = 1.0 if hist15>0 else -0.3
            else:
                scores["macd_15m"] = 1.0 if hist15<0 else -0.3

            scores["st_15m"]  = 0.8 if st15==macro else -0.6
            scores["adx_15m"] = 0.6 if adx15>30 else (0.3 if adx15>20 else -0.3)

        # 5m
        if k5m is not None and len(k5m) >= 55:
            c5=k5m[:,4]
            e8_5=ema(c5,8); e21_5=ema(c5,21); e55_5=ema(c5,55)
            rsi5=rsi(c5,14); st5=calc_supertrend(k5m,7,2.5)
            _,_,hist5=calc_macd(c5); p5=float(c5[-1])

            if macro==1:
                if e8_5[-1]>e21_5[-1]>e55_5[-1]: scores["ema_5m"]=0.9
                elif e8_5[-1]>e21_5[-1]:           scores["ema_5m"]=0.4
                else:                               scores["ema_5m"]=-0.3
                scores["rsi_5m"] = 1.0 if rsi5<35 else (0.5 if rsi5<48 else (-0.5 if rsi5>68 else 0.2))
            else:
                if e8_5[-1]<e21_5[-1]<e55_5[-1]: scores["ema_5m"]=0.9
                elif e8_5[-1]<e21_5[-1]:           scores["ema_5m"]=0.4
                else:                               scores["ema_5m"]=-0.3
                scores["rsi_5m"] = 1.0 if rsi5>65 else (0.5 if rsi5>52 else (-0.5 if rsi5<32 else 0.2))

            scores["st_5m"]  = 0.7 if st5==macro else -0.8  # raised per report
            scores["macd_5m"] = 0.6 if (hist5>0 and macro==1) or (hist5<0 and macro==-1) else -0.5  # raised per report

            # Bollinger — pullback-to-band entries preferred
            if len(c5)>=20:
                bm=c5[-20:].mean(); bs=c5[-20:].std()
                bu=bm+2*bs; bd=bm-2*bs
                if macro==1:
                    scores["bb"] = 1.0 if p5<bd else (0.5 if p5<bm else (-0.5 if p5>bu else 0.2))
                else:
                    scores["bb"] = 1.0 if p5>bu else (0.5 if p5>bm else (-0.5 if p5<bd else 0.2))

            scores["volume"] = calc_volume_confirm(k5m,8)*0.9 - 0.25

            # VWAP
            vw=vwap(k5m,20)
            if vw>0:
                dev=(p5-vw)/vw
                if macro==1:
                    scores["vwap"] = 0.5 if dev<-0.003 else (-0.3 if dev>0.012 else 0.15)
                else:
                    scores["vwap"] = 0.5 if dev>0.003 else (-0.3 if dev<-0.012 else 0.15)

        # 3m timing
        if k3m is not None and len(k3m)>=30:
            c3=k3m[:,4]; rsi3=rsi(c3,9)
            e5_3=ema(c3,5); e13_3=ema(c3,13)
            cross_up = e5_3[-1]>e13_3[-1] and len(e5_3)>1 and e5_3[-2]<=e13_3[-2]
            cross_dn = e5_3[-1]<e13_3[-1] and len(e5_3)>1 and e5_3[-2]>=e13_3[-2]
            if macro==1:
                scores["cross_3m"] = 0.8 if cross_up else (0.3 if e5_3[-1]>e13_3[-1] else -0.5)  # raised per report
                scores["rsi_3m"]   = 0.5 if rsi3<45 else (-0.3 if rsi3>72 else 0.1)
            else:
                scores["cross_3m"] = 0.8 if cross_dn else (0.3 if e5_3[-1]<e13_3[-1] else -0.5)  # raised per report
                scores["rsi_3m"]   = 0.5 if rsi3>55 else (-0.3 if rsi3<28 else 0.1)

        # Microstructure
        fs = funding_signal(funding)
        scores["funding"] = (abs(fs)*0.8 if (macro==1 and fs>0) or (macro==-1 and fs<0)
                             else (fs*0.4 if abs(fs)>0.5 else 0.0))

        if macro==1:
            scores["ls"]  = 0.6 if ls<0.7 else (0.3 if ls<0.85 else (-0.5 if ls>1.4 else 0.1))
        else:
            scores["ls"]  = 0.6 if ls>1.4 else (0.3 if ls>1.15 else (-0.5 if ls<0.7 else 0.1))

        scores["oi"] = min(max(oi_pct/5.0,-1.0),1.0)*0.4

        if macro==1:
            scores["ob"] = 0.7 if ob_imb>2.0 else (0.35 if ob_imb>1.3 else (-0.5 if ob_imb<0.6 else 0.1))
        else:
            scores["ob"] = 0.7 if ob_imb<0.5 else (0.35 if ob_imb<0.77 else (-0.5 if ob_imb>1.7 else 0.1))

       

        # Check if liqs is None before trying to use .get()
        if liqs is None:
            scores["liq"] = 0.0
        else:
            ll = liqs.get("long_liq", 0)
            sl2 = liqs.get("short_liq", 0)
            tot = ll + sl2
            if tot > 0:
                sdm = sl2 / tot
                if macro == 1:
                    scores["liq"] = 1.0 if sdm > 0.75 else (0.5 if sdm > 0.55 else (-0.7 if sdm < 0.25 else 0.0))
                else:
                    scores["liq"] = 1.0 if sdm < 0.25 else (0.5 if sdm < 0.45 else (-0.7 if sdm > 0.75 else 0.0))
            else:
                scores["liq"] = 0.0
            scores["learned"] = max(-0.5, min(0.5, learned_bias * float(macro)))
                    
        # Weights
        # Weights retuned per weekly report analysis of 89 trades:
        # Orderbook +75%, 5m ST +38%, 5m MACD +40%, 3m cross +44%
        # HTF alignment remains most critical (only score that ALL winners share)
        W = {
            "htf_alignment":3.0,"ema_15m":2.0,"macd_15m":1.8,"st_15m":1.7,
            "rsi_15m":1.5,"adx_15m":1.2,"ema_5m":1.6,
            "st_5m":1.8,       # +38% (was 1.3) — report: 5m ST = 0.7 on all wins
            "rsi_5m":1.1,
            "macd_5m":1.4,     # +40% (was 1.0) — loss-correlated indicator
            "bb":0.9,"volume":1.0,"vwap":0.7,
            "cross_3m":1.3,    # +44% (was 0.9) — loss-correlated
            "rsi_3m":0.6,
            "liq":1.3,"funding":1.0,
            "ob":1.4,          # +75% (was 0.8) — report identified as key loss predictor
            "ls":0.7,"oi":0.5,"learned":1.2,
        }
        tw   = sum(W.get(k,1.0) for k in scores)
        comp = sum(scores[k]*W.get(k,1.0) for k in scores)/(tw+1e-9)

        pos=sum(1 for v in scores.values() if v>0.1)
        neg=sum(1 for v in scores.values() if v<-0.1)
        agreement = max(pos,neg)/len(scores) if scores else 0.5
        confidence = int(min(100, abs(comp)*80 + agreement*25))

        # Thresholds — no CONSERVATIVE mode
        # NORMAL is the baseline: more frequent than before, still quality-filtered
        # Confidence floors raised per weekly report:
        thresholds  = {"NORMAL":0.25, "AGGRESSIVE":0.18, "TURBO":0.14}
        conf_floors = {"NORMAL":68,   "AGGRESSIVE":62,   "TURBO":55}  # NORMAL:65→68, AGG:58→62
        thresh      = thresholds.get(mode, 0.25)
        conf_floor  = conf_floors.get(mode, 52)

        if macro==1 and comp>=thresh and confidence>=conf_floor:
            signal="LONG"
        elif macro==-1 and comp<=-thresh and confidence>=conf_floor:
            signal="SHORT"
        else:
            signal="HOLD"

        logger.debug(f"{sym} macro={macro} comp={comp:.3f} conf={confidence} signal={signal}")

        return {
            "signal":signal,"confidence":confidence,"composite":round(comp,4),
            "macro_direction":macro,"agreement_pct":round(agreement*100,1),
            "components":{k:round(v,3) for k,v in scores.items()},
            "veto_reason": None if signal!="HOLD" else
                f"comp={comp:.3f}<{thresh:.2f} or conf={confidence}<{conf_floor}",
        }


def sl_tp(side, entry, atr_val, k5m=None, k15m=None, sl_mult=1.8, tp_mult=2.8):
    """
    ATR SL/TP — balanced for frequency + profitability.
    SL: 1.8x ATR — tight enough to protect, loose enough to breathe
    TP: 2.8x ATR — 1:1.56 R:R, hits more often = more frequent wins
    """
    if atr_val<=0: atr_val=entry*0.01
    dp=price_precision(entry)
    sl_d=atr_val*sl_mult; tp_d=atr_val*tp_mult

    if k15m is not None:
        sup_pct, res_pct = detect_swing_structure(k15m)
        if side=="Buy":
            sup_price = entry*(1+sup_pct)
            if sup_price < entry and sup_price > entry*0.90:
                sl_d = max(entry-sup_price+atr_val*0.3, atr_val*1.5)
        else:
            res_price = entry*(1+res_pct)
            if res_price > entry and res_price < entry*1.10:
                sl_d = max(res_price-entry+atr_val*0.3, atr_val*1.5)

    if side=="Buy":
        return round(entry-sl_d,dp), round(entry+tp_d,dp)
    else:
        return round(entry+sl_d,dp), round(entry-tp_d,dp)
