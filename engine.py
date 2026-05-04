"""
engine.py v2 — Perpetual Compounding Trading Engine
New features:
  • 3%+ unrealised profit → close immediately (take profits)
  • Trailing stop: move SL to breakeven at 2%, trail at 1.5x ATR above 5%
  • Diversification: max 1 position per asset class, never add to a loser
  • Hourly market regime: TRENDING | RANGING | VOLATILE → adjusts strategy live
  • No upward profit cap — trailing stop lets winners run forever
"""
import asyncio, json, os, time, logging, smtplib, threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from bybit_client    import BybitClient
from scanner         import VScanner, parse_klines, atr, p_funding, p_ls, p_oi, p_ob, p_liq, price_precision
from signals         import QuantSignalEngine, sl_tp
from whale_intelligence import WhaleEngine
from compound_engine import CompoundEngine, DAILY_REQUIRED_PCT, EPOCH_DAYS
from db import (
    init_db, gp, sp, open_trade, close_trade, update_pos_price,
    get_open_positions, get_trades, all_time_stats, log_capital,
    get_symbol_stats, snap_hour, close_epoch_record, open_epoch_record,
    get_all_epochs
)

logger = logging.getLogger(__name__)
REPORT_EMAIL = os.getenv("REPORT_EMAIL", "lgwaithaka@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST",    "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT","587"))
SMTP_USER    = os.getenv("SMTP_USER",    "")
SMTP_PASS    = os.getenv("SMTP_PASS",    "")

# ── Asset class groupings ─────────────────────────────────────────────────────
ASSET_CLASSES = {
    "BTC":   ["BTCUSDT"],
    "ETH":   ["ETHUSDT","ETHFIUSDT","WETHUSDT"],
    "SOL":   ["SOLUSDT"],
    "BNB":   ["BNBUSDT"],
    "DOGE":  ["DOGEUSDT"],
    "XRP":   ["XRPUSDT"],
    "AVAX":  ["AVAXUSDT"],
    "LINK":  ["LINKUSDT"],
    "ADA":   ["ADAUSDT"],
    "MEME":  ["PEPEUSDT","SHIBUSDT","FLOKIUSDT","WIFUSDT","BONKUSDT","POPCATUSDT"],
    "LAYER2":["ARBUSDT","OPUSDT","MATICUSDT","STRKUSDT"],
    "DEFI":  ["UNIUSDT","AAVEUSDT","CRVUSDT","MKRUSDT"],
    "OTHER": [],
}

def get_asset_class(symbol: str) -> str:
    for cls, syms in ASSET_CLASSES.items():
        if symbol in syms: return cls
    for cls in ["BTC","ETH","SOL","BNB","DOGE","XRP","AVAX","LINK","ADA"]:
        if symbol.startswith(cls): return cls
    return "OTHER"


class MarketRegime:
    TRENDING = "TRENDING"
    RANGING  = "RANGING"
    VOLATILE = "VOLATILE"

    def __init__(self):
        self.regime         = self.RANGING
        self.trend_strength = 0.0
        self.volatility_pct = 0.0
        self.last_updated   = 0

    def update(self, klines_dict: Dict[str, np.ndarray]):
        from scanner import ema as _ema
        trend_vals, range_vals = [], []
        for sym, k in klines_dict.items():
            if k is None or len(k) < 22: continue
            closes = k[:, 4]; highs = k[:, 2]; lows = k[:, 3]
            price  = float(closes[-1])
            if price <= 0: continue
            e8  = _ema(closes, 8)
            e21 = _ema(closes, 21)
            slope = abs((e8[-1] - e8[-5]) / (e8[-5] + 1e-9))
            trend_vals.append(slope)
            rng = (highs[-10:].max() - lows[-10:].min()) / price
            range_vals.append(float(rng))

        if not trend_vals: return
        avg_trend = float(np.mean(trend_vals))
        avg_range = float(np.mean(range_vals))
        self.trend_strength = avg_trend
        self.volatility_pct = avg_range * 100

        if avg_range > 0.065:     self.regime = self.VOLATILE
        elif avg_trend > 0.0025:  self.regime = self.TRENDING
        else:                     self.regime = self.RANGING
        self.last_updated = int(time.time())
        logger.info(f"Regime: {self.regime} | trend={avg_trend:.5f} | range={avg_range*100:.2f}%")

    def adjustments(self) -> Dict:
        if self.regime == self.TRENDING:
            return {"conf_bonus":-3, "tp_bonus":0.6, "sl_bonus":-0.1, "lev_bonus":2,
                    "label":"TRENDING — momentum entries, let winners run"}
        elif self.regime == self.VOLATILE:
            return {"conf_bonus":+5, "tp_bonus":-0.4, "sl_bonus":0.3, "lev_bonus":-3,
                    "label":"VOLATILE — selective entries, wider stops, quick TP"}
        else:
            return {"conf_bonus":+2, "tp_bonus":-0.2, "sl_bonus":0.0, "lev_bonus":0,
                    "label":"RANGING — mean-reversion, tighter targets"}


class CompoundTradingEngine:
    def __init__(self):
        self.client:   Optional[BybitClient]    = None
        self.scanner:  Optional[VScanner]       = None
        self.signals   = QuantSignalEngine()
        self.compound  = CompoundEngine()
        self.regime    = MarketRegime()
        self.whales    = WhaleEngine()        # Smart money / whale tracker

        self.running        = False
        self.status         = "IDLE"
        self.balance               = 0.0
        self.capital_auto_detected = False   # True until real balance confirmed from Bybit
        self.live_positions: Dict[str, Dict]   = {}
        self.cooldown_until: Dict[str, float]  = {}
        self.losing_symbols: Set[str]          = set()
        self._klines_cache: Dict[str, np.ndarray] = {}

        self.scan_results: List[Dict] = []
        self.last_scan_ts  = 0
        self.last_hour_ts  = 0
        self.errors: List[str] = []

    def init(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client  = BybitClient(api_key, api_secret, testnet)
        self.scanner = VScanner(self.client)
        init_db()

        epoch_num = int(gp("current_epoch",  "1"))
        epoch_ts  = int(gp("epoch_start_ts", str(int(time.time()))))

        # Determine if this is a fresh first run (no capital seeded yet)
        saved_initial = gp("initial_capital", None)
        saved_epoch   = gp("epoch_start_bal", None)
        first_run     = (saved_initial is None or saved_initial == "10.0"
                         or saved_epoch  is None or saved_epoch  == "10.0")

        if first_run:
            # Will be updated in main_loop after async balance fetch
            logger.info("Fresh start — will auto-detect balance from Bybit on first loop")
            self.balance = 0.0
            self.capital_auto_detected = True
        else:
            self.balance = float(saved_epoch)
            self.capital_auto_detected = False

        epoch_bal = float(saved_epoch or "0.0")
        initial   = float(saved_initial or "0.0")
        self.compound.initialise(initial or epoch_bal or 10.0,
                                 epoch_num, epoch_ts,
                                 epoch_bal or 10.0)
        self.whales.set_client(self.client)
        logger.info(f"Engine init | Epoch {epoch_num} | saved_bal=${epoch_bal:.4f} | auto_detect={self.capital_auto_detected}")

    # ── Balance ──────────────────────────────────────────────────────────────

    async def refresh_balance(self) -> float:
        try:
            for at in ["UNIFIED", "CONTRACT"]:
                resp = await self.client.wallet(at)
                if resp.get("retCode", -1) != 0: continue
                for acc in resp.get("result", {}).get("list", []):
                    v = float(acc.get("totalWalletBalance", "0") or "0")
                    if v > 0:
                        self.balance = v; return v
                    for coin in acc.get("coin", []):
                        if coin.get("coin") == "USDT":
                            cv = float(coin.get("walletBalance","0") or "0")
                            if cv > 0:
                                self.balance = cv; return cv
        except Exception as e:
            logger.error(f"refresh_balance: {e}")
        return self.balance

    async def refresh_live_positions(self):
        try:
            resp = await self.client.positions()
            self.live_positions = {
                p["symbol"]: p
                for p in resp.get("result",{}).get("list",[])
                if float(p.get("size","0")) > 0
            }
        except Exception as e:
            logger.error(f"refresh_live_positions: {e}")

    # ── Fetch Market Data ─────────────────────────────────────────────────────


    async def fetch_data(self, symbol: str) -> Optional[Dict]:
        try:
            results = await asyncio.gather(
                self.client.klines(symbol, "3",   80),   # 3m  — timing
                self.client.klines(symbol, "5",  130),   # 5m  — entry
                self.client.klines(symbol, "15",  80),   # 15m — setup
                self.client.klines(symbol, "60",  60),   # 1h  — structure
                self.client.klines(symbol, "240", 60),   # 4H  — MACRO GATE
                self.client.funding(symbol),
                self.client.ls_ratio(symbol),
                self.client.open_interest(symbol, "1h"),
                self.client.orderbook(symbol, 50),
                self.client.liquidations(symbol),
                return_exceptions=True
            )
            k3,k5,k15,k1h,k4h,fr,lsr,oi,ob,liq = results
            k5p = parse_klines(k5) if not isinstance(k5, Exception) else None
            if k5p is not None:
                self._klines_cache[symbol] = k5p
            return {
                "k3m":  parse_klines(k3)   if not isinstance(k3,  Exception) else None,
                "k5m":  k5p,
                "k15m": parse_klines(k15)  if not isinstance(k15, Exception) else None,
                "k1h":  parse_klines(k1h)  if not isinstance(k1h, Exception) else None,
                "k4h":  parse_klines(k4h)  if not isinstance(k4h, Exception) else None,
                "funding": p_funding(fr)   if not isinstance(fr,  Exception) else 0.0,
                "ls":      p_ls(lsr)       if not isinstance(lsr, Exception) else 1.0,
                "oi_pct":  p_oi(oi)        if not isinstance(oi,  Exception) else 0.0,
                "ob_imb":  p_ob(ob)        if not isinstance(ob,  Exception) else 1.0,
                "liqs":    p_liq(liq)      if not isinstance(liq, Exception) else {},
            }
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}"); return None

    # ── 3% Profit Taking + Trailing Stop (No Upper Limit) ────────────────────

    async def check_profit_targets(self):
        db_open = get_open_positions()
        initial = float(gp("epoch_start_bal","0.0") or "0.0") or self.balance or 10.0

        for pos in db_open:
            sym      = pos["symbol"]
            entry    = float(pos.get("entry_price", 0) or 0)
            qty      = float(pos.get("qty",         0) or 0)
            side     = pos.get("side", "Buy")
            trade_id = pos.get("trade_id")
            unr      = float(pos.get("unrealised_pnl", 0) or 0)
            cur      = float(pos.get("current_price",  0) or 0)

            if entry <= 0 or qty <= 0 or cur <= 0: continue
            notional = entry * qty
            unr_pct  = (unr / notional * 100) if notional > 0 else 0

            # ─ Take profit at 3%+ ─────────────────────────────────────────
            if unr_pct >= 3.0:
                logger.info(f"TAKE PROFIT | {sym} +{unr_pct:.2f}% | ${unr:.4f}")
                try:
                    resp = await self.client.close_pos(sym, side, str(qty))
                    if resp.get("retCode",-1) == 0:
                        await asyncio.sleep(0.5)
                        pr  = await self.client.closed_pnl(sym, 5)
                        pls = pr.get("result",{}).get("list",[])
                        pnl    = float(pls[0].get("closedPnl",   "0") or "0") if pls else unr
                        exit_p = float(pls[0].get("avgExitPrice","0") or "0") if pls else cur
                        close_trade(trade_id, exit_p, pnl, initial)
                        self.compound.record_outcome(pnl > 0, pnl, self.balance)
                        self.cooldown_until[sym] = time.time() + 30
                        logger.info(f"PROFIT BANKED | {sym} | ${pnl:.4f}")
                except Exception as e:
                    logger.error(f"Profit close error {sym}: {e}")
                continue

            # ─ Trailing stop: no upward limit ─────────────────────────────
            if unr_pct >= 5.0:
                try:
                    raw5   = await self.client.klines(sym, "5", 20)
                    k5     = parse_klines(raw5)
                    atr_v  = atr(k5, 14) if k5 is not None else entry * 0.01
                    trail  = atr_v * 1.5
                    prec   = price_precision(cur)
                    if side == "Buy":
                        new_sl = round(cur - trail, prec)
                        old_sl = float(pos.get("sl_price", 0) or 0)
                        if new_sl > old_sl + (atr_v * 0.2):
                            await self.client.set_tpsl(sym, sl=str(new_sl))
                            logger.info(f"TRAIL ↑ | {sym} sl={new_sl:.4f} (+{unr_pct:.1f}%)")
                    else:
                        new_sl = round(cur + trail, prec)
                        old_sl = float(pos.get("sl_price", 0) or 0)
                        if old_sl == 0 or new_sl < old_sl - (atr_v * 0.2):
                            await self.client.set_tpsl(sym, sl=str(new_sl))
                            logger.info(f"TRAIL ↓ | {sym} sl={new_sl:.4f} (+{unr_pct:.1f}%)")
                except Exception as e:
                    logger.debug(f"Trail SL {sym}: {e}")

            # ─ Breakeven at 2% ────────────────────────────────────────────
            elif unr_pct >= 2.0:
                try:
                    prec   = price_precision(entry)
                    buf    = entry * 0.0008
                    old_sl = float(pos.get("sl_price", 0) or 0)
                    if side == "Buy":
                        be = round(entry + buf, prec)
                        if old_sl < be - buf:
                            await self.client.set_tpsl(sym, sl=str(be))
                            logger.info(f"BREAKEVEN | {sym} sl={be:.4f}")
                    else:
                        be = round(entry - buf, prec)
                        if old_sl == 0 or old_sl > be + buf:
                            await self.client.set_tpsl(sym, sl=str(be))
                            logger.info(f"BREAKEVEN | {sym} sl={be:.4f}")
                except Exception as e:
                    logger.debug(f"Breakeven {sym}: {e}")

    # ── Diversification Gate ──────────────────────────────────────────────────

    def _can_open(self, symbol: str, signal: str) -> Tuple[bool, str]:
        db_open   = get_open_positions()
        open_syms = {p["symbol"] for p in db_open}

        if symbol in open_syms:
            return False, "Already open"
        if symbol in self.losing_symbols:
            return False, "Currently losing — cooling off"

        my_class = get_asset_class(symbol)
        for p in db_open:
            if get_asset_class(p["symbol"]) == my_class:
                return False, f"Class {my_class} occupied by {p['symbol']}"

        if len(open_syms) >= self.compound.compute_max_concurrent():
            return False, "Max concurrent reached"

        return True, "OK"

    # ── Trade Execution ───────────────────────────────────────────────────────

    async def execute(self, symbol: str, analysis: Dict, meta: Dict) -> bool:
        ce     = self.compound
        signal = analysis["signal"]
        conf   = analysis["confidence"]
        comp   = analysis["composite"]
        mode   = ce.state.mode
        adj    = self.regime.adjustments()

        raw5  = await self.client.klines(symbol, "5", 50)
        k5    = parse_klines(raw5)
        if k5 is None or len(k5) == 0: return False

        price   = float(k5[-1, 4])
        atr_val = atr(k5, 14)

        ss      = get_symbol_stats(symbol)
        sl_mult = max(0.8, min(ss.get("sl_mult", 1.2) + adj["sl_bonus"], 2.5))
        tp_mult = max(1.8, min(ss.get("tp_mult", 2.4) + adj["tp_bonus"], 5.0))
        leverage= max(10, min(ce.compute_leverage(meta.get("range_pct", 5.0)) + adj["lev_bonus"], 25))

        risk_pct     = ce.compute_risk_pct()
        risk_usd     = self.balance * risk_pct
        min_notional = float(gp("min_notional_usdt","5.5"))
        qty          = (risk_usd * leverage) / price

        if qty * price < min_notional:
            qty = (min_notional * 1.05) / price
        max_n = self.balance * 0.40 * leverage
        if qty * price > max_n:
            qty = max_n / price

        if price >= 1000: qty = round(qty, 3)
        elif price >= 1:  qty = round(qty, 2)
        else:             qty = max(round(qty, 0), 1)

        if self.balance < (min_notional / leverage) * 1.5:
            logger.warning(f"Balance too low for {symbol}"); return False

        side = "Buy" if signal == "LONG" else "Sell"
        # Pass kline data for structural SL placement
        k5_cached  = self._klines_cache.get(symbol)
        # Fetch fresh k15m for structure (from last fetch_data result)
        sl, tp = sl_tp(
        side, price, atr_val,
        k5m=k5_cached,
        k15m=None,      # structural from 15m handled inside sl_tp
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        )

        try: await self.client.set_leverage(symbol, leverage)
        except Exception: pass

        resp = await self.client.place_order(symbol, side, qty, sl=sl, tp=tp)
        if resp.get("retCode", -1) != 0:
            err = resp.get("retMsg","?")
            logger.error(f"Order fail {symbol}: {err}")
            self.errors.append(f"{datetime.now().isoformat()} | {symbol} | {err}")
            return False

        order_id = resp.get("result",{}).get("orderId","")
        epoch    = ce.state.epoch_num
        open_trade(
            epoch=epoch, symbol=symbol, side=side, signal=signal,
            confidence=conf, composite=comp,
            entry_price=price, qty=qty, leverage=leverage,
            sl=sl, tp=tp, order_id=order_id,
            tag=f"EP{epoch}_L{leverage}_{self.regime.regime[:3]}_{mode[:3]}",
            mode=mode, components=analysis.get("components",{}),
            vol_score=meta.get("vol_score",0), range_pct=meta.get("range_pct",0),
        )
        logger.info(
            f"OPEN [{mode}/{self.regime.regime}] {symbol} {signal} "
            f"qty={qty}@${price:.4f} SL={sl:.4f} TP={tp:.4f} L={leverage}x"
        )
        return True

    # ── Position Monitor ──────────────────────────────────────────────────────

    async def monitor(self):
        await self.refresh_live_positions()
        db_open = get_open_positions()
        initial = float(gp("epoch_start_bal","0.0") or "0.0") or self.balance or 10.0

        self.losing_symbols = set()
        for pos in db_open:
            if (pos.get("unrealised_pnl") or 0) < -0.05:
                self.losing_symbols.add(pos["symbol"])

        for pos in db_open:
            sym = pos["symbol"]; tid = pos["trade_id"]
            if sym in self.live_positions:
                live = self.live_positions[sym]
                try:
                    unr  = float(live.get("unrealisedPnl","0") or "0")
                    cprc = float(live.get("markPrice",    "0") or "0")
                    update_pos_price(sym, cprc, unr)
                except Exception: pass
            else:
                pnl = 0.0; exit_p = 0.0
                try:
                    pr  = await self.client.closed_pnl(sym, 10)
                    pls = pr.get("result",{}).get("list",[])
                    if pls:
                        pnl    = float(pls[0].get("closedPnl",   "0") or "0")
                        exit_p = float(pls[0].get("avgExitPrice","0") or "0")
                except Exception: pass
                outcome = close_trade(tid, exit_p, pnl, initial)
                win     = outcome == "WIN"
                self.compound.record_outcome(win, pnl, self.balance)
                self.cooldown_until[sym] = time.time() + (30 if win else 180)
                if not win: self.losing_symbols.add(sym)
                logger.info(f"CLOSED | {sym} | {outcome} | ${pnl:.4f}")

    # ── Scan & Trade ──────────────────────────────────────────────────────────

    async def scan_and_trade(self):
        ce = self.compound
        if ce.check_circuit_breakers(
            self.balance,
            epoch_max_dd=float(gp("epoch_max_dd_pct","35"))/100,
            daily_max_dd=float(gp("daily_max_dd_pct","25"))/100,
        ):
            return

        await self.refresh_balance()
        top_syms   = await self.scanner.scan(n=int(gp("vol_scan_n","10")))
        adj        = self.regime.adjustments()
        conf_floor = max(35, ce.compute_confidence_floor() + adj["conf_bonus"])
        mode       = ce.state.mode

        results = []
        for meta in top_syms:
            sym = meta["symbol"]
            if self.cooldown_until.get(sym, 0) > time.time(): continue

            data = await self.fetch_data(sym)
            if not data: continue

            ss   = get_symbol_stats(sym)
            bias = ss.get("learned_bias", 0.0)

            # Incorporate whale intelligence score
            # Cached from last hourly update — decays with age
            whale_score = self.whales.get_signal_for(sym)

            an   = self.signals.analyze(
                sym=sym, k3m=data["k3m"], k5m=data["k5m"],
                k15m=data["k15m"], k1h=data["k1h"],
                k4h=data.get("k4h"),              # 4H macro gate
                funding=data["funding"], ls=data["ls"],
                oi_pct=data["oi_pct"], ob_imb=data["ob_imb"],
                liqs=data["liqs"],
                learned_bias=bias + (whale_score * 0.3),  # Blend whale signal into bias
                mode=mode,
            )
            # Attach whale metadata to result
            an["whale_score"]  = round(whale_score, 3)
            an["whale_intel"]  = self.whales._cache.get(sym)

            result = {
                "symbol":sym, "signal":an["signal"], "confidence":an["confidence"],
                "composite":an["composite"], "leverage":ce.compute_leverage(meta.get("range_pct",5)),
                "vol_score":meta.get("vol_score",0), "range_pct":meta.get("range_pct",0),
                "chg_24h":meta.get("chg_24h",0),
                "regime":self.regime.regime, "mode":mode,
                "ts":int(time.time()), "action":"HOLD"
            }

            if an["signal"] != "HOLD" and an["confidence"] >= conf_floor:
                can, reason = self._can_open(sym, an["signal"])
                if can:
                    ok = await self.execute(sym, an, meta)
                    result["action"] = "OPENED" if ok else "EXEC_FAIL"
                else:
                    result["action"] = f"BLOCKED:{reason}"

            results.append(result)
            await asyncio.sleep(0.25)

        self.scan_results = results
        self.last_scan_ts = int(time.time())

    # ── Hourly Strategy Adjustment ────────────────────────────────────────────

    async def hourly_strategy_update(self):
        await self.refresh_balance()
        ce = self.compound; s = ce.state

        if self._klines_cache:
            self.regime.update(self._klines_cache)

        h_trades = get_trades(hours=1)
        h_closed = [t for t in h_trades if t["outcome"] not in ("OPEN",None)]
        h_wins   = sum(1 for t in h_closed if t["outcome"]=="WIN")
        h_pnl    = sum(t["pnl_usdt"] or 0 for t in h_closed)
        h_wr     = h_wins/len(h_closed)*100 if h_closed else 0
        stats    = all_time_stats()

        snap_hour(s.epoch_num, self.balance, h_pnl, len(h_closed), h_wins,
                  stats["open_positions"], s.mode)
        log_capital(epoch=s.epoch_num, day_in_epoch=s.day_in_epoch,
                    balance=self.balance, target_now=ce.target_at_now(),
                    target_eod=ce.target_at_day_end(),
                    ahead_pct=ce.ahead_pct(self.balance),
                    mode=s.mode, open_pos=stats["open_positions"])

        sp("market_regime",    self.regime.regime)
        sp("regime_label",     self.regime.adjustments()["label"])
        sp("hourly_win_rate",  str(round(h_wr,1)))
        sp("hourly_pnl",       str(round(h_pnl,4)))
        sp("last_strategy_ts", str(int(time.time())))

        await self.scanner.scan(force=True)

        # Hourly whale intelligence update
        try:
            top_syms = [x["symbol"] for x in self.scanner.top]
            if top_syms and self.client:
                whale_data = await self.whales.update_all(top_syms, self.client)
                top_ops    = self.whales.get_top_opportunities(5)
                sp("whale_summary",        json.dumps(self.whales.last_summary))
                sp("whale_opportunities",  json.dumps(top_ops))
                sp("whale_last_update",    str(int(time.time())))
                logger.info(f"Whale update | market_bias={self.whales.last_summary.get('market_bias','?')} | "
                            f"squeezes={self.whales.last_summary.get('short_squeezes',[])} | "
                            f"top={self.whales.get_top_opportunities(3)}")
        except Exception as e:
            logger.error(f"Hourly whale update error: {e}")

        logger.info(f"Hourly | regime={self.regime.regime} | pnl=${h_pnl:.4f} | wr={h_wr:.1f}%")

    # ── Epoch Boundary ────────────────────────────────────────────────────────

    async def check_epoch_boundary(self):
        await self.refresh_balance()
        ce = self.compound; old = ce.state.epoch_num
        if ce.advance_epoch(self.balance):
            close_epoch_record(old, self.balance)
            sp("current_epoch",   str(ce.state.epoch_num))
            sp("epoch_start_ts",  str(ce.state.epoch_start_ts))
            sp("epoch_start_bal", str(round(ce.state.epoch_start_bal,4)))
            open_epoch_record(ce.state.epoch_num, ce.state.epoch_start_bal)
            self.losing_symbols.clear()
            await self.send_epoch_report(old, self.balance, ce.state.epoch_target)

    async def daily_tasks(self):
        await self.refresh_balance()
        self.compound.advance_day(self.balance)
        self.compound.reset_daily_cb()
        self.losing_symbols.clear()
        await self.send_daily_report()

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def main_loop(self):
        self.running = True; self.status = "RUNNING"

        # ── Auto-detect balance from Bybit on startup ─────────────────────────
        live_bal = await self.refresh_balance()
        logger.info(f"Live Bybit balance on startup: ${live_bal:.4f} USDT")

        if live_bal > 0 and self.capital_auto_detected:
            # First ever run — seed all capital params from real balance
            logger.info(f"Auto-seeding capital from Bybit balance: ${live_bal:.4f}")
            sp("initial_capital", str(round(live_bal, 4)))
            sp("epoch_start_bal", str(round(live_bal, 4)))
            sp("epoch_start_ts",  str(int(time.time())))
            sp("current_epoch",   "1")

            # Update compound engine with real balance
            self.compound.initialise(
                start_balance   = live_bal,
                epoch_num       = 1,
                epoch_start_ts  = int(time.time()),
                epoch_start_bal = live_bal,
            )

            # Update epoch 1 DB record with real start balance
            open_epoch_record(1, live_bal)

            # Adjust min notional and concurrent based on balance size
            if live_bal < 20:
                sp("min_notional_usdt", "5.5")
                sp("max_concurrent",    "3")
                sp("vol_scan_n",        "8")
            elif live_bal < 100:
                sp("min_notional_usdt", "5.5")
                sp("max_concurrent",    "5")
                sp("vol_scan_n",        "10")
            elif live_bal < 500:
                sp("min_notional_usdt", "10.0")
                sp("max_concurrent",    "8")
                sp("vol_scan_n",        "12")
            else:
                sp("min_notional_usdt", "20.0")
                sp("max_concurrent",    "12")
                sp("vol_scan_n",        "14")

            self.capital_auto_detected = False
            logger.info(
                f"Capital auto-configured | initial=${live_bal:.4f} "
                f"| target=${self.compound.state.epoch_target:.4f} "
                f"| epoch_1"
            )

        elif live_bal > 0:
            # Subsequent run — update current balance but keep epoch intact
            self.balance = live_bal
            logger.info(f"Engine resumed | Epoch {self.compound.state.epoch_num} | ${live_bal:.4f}")

        else:
            logger.warning(
                "Could not read Bybit balance (0.0). "
                "Check API keys and that funds are in Unified Trading Account. "
                "Engine will retry on each scan cycle."
            )

        logger.info(f"Engine START | ${self.balance:.4f} | mode={self.compound.state.mode} | target=${self.compound.state.epoch_target:.4f}")
        scan_secs = int(gp("scan_interval_s","45"))
        last_daily_hour = -1

        while self.running:
            t0 = time.time()
            try:
                await self.monitor()
                await self.check_profit_targets()
                await self.check_epoch_boundary()
                await self.scan_and_trade()
                now = time.time()
                if now - self.last_hour_ts >= 3600:
                    await self.hourly_strategy_update()
                    self.last_hour_ts = now
                cur_hour = datetime.now(timezone.utc).hour
                if cur_hour == 0 and last_daily_hour != 0:
                    await self.daily_tasks()
                last_daily_hour = cur_hour
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Loop: {e}", exc_info=True)
                self.errors.append(f"{datetime.now().isoformat()} | {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(max(0, scan_secs - (time.time() - t0)))

        self.status = "STOPPED"; self.running = False
        if self.client: await self.client.close()

    async def stop(self): self.running = False

    # ── Reports ───────────────────────────────────────────────────────────────

    async def send_epoch_report(self, epoch_num, achieved, next_target):
        try:
            stake = self.compound.state.epoch_start_bal
            pct   = (achieved - stake) / stake * 100 if stake else 0
            body  = (f"EPOCH {epoch_num} | ${stake:.2f}→${achieved:.2f} "
                     f"({pct:+.1f}%) | {'✓ DOUBLED' if achieved>=stake*2 else '✗ MISSED'}\n"
                     f"Next epoch: ${achieved:.2f}→${next_target:.2f}\n")
            self._send_email(f"[Bybit] Ep{epoch_num} ${achieved:.2f}", body)
        except Exception as e: logger.error(f"epoch report: {e}")

    async def send_daily_report(self):
        try:
            ce = self.compound; stats = all_time_stats()
            body = (f"Day {ce.state.day_in_epoch}/5 Ep{ce.state.epoch_num} | "
                    f"${self.balance:.4f} | Mode:{ce.state.mode} | "
                    f"Regime:{self.regime.regime}\n"
                    f"WR:{stats['win_rate']}% PnL:${stats['total_pnl']:.4f}\n")
            self._send_email(
                f"[Bybit] Day{ce.state.day_in_epoch} ${self.balance:.2f}", body)
        except Exception as e: logger.error(f"daily report: {e}")

    def _send_email(self, subject, body):
        if not SMTP_USER or not SMTP_PASS: return
        try:
            msg = MIMEText(body); msg["Subject"]=subject
            msg["From"]=SMTP_USER; msg["To"]=REPORT_EMAIL
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as sv:
                sv.starttls(); sv.login(SMTP_USER, SMTP_PASS); sv.send_message(msg)
        except Exception as e: logger.error(f"email: {e}")


engine = CompoundTradingEngine()
