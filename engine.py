"""
engine.py v4.0 — Perpetual Compounding Trading Engine
═══════════════════════════════════════════════════════
Strategy: Hold positions longer, close in profit tranches.

  ✔ NO exchange-side TP — positions managed programmatically
  ✔ TRANCHE CLOSE: 40% at 15%, 30% at 20%, 30% at 25% (return on margin)
  ✔ BREAKEVEN SL: move SL to entry once in ANY profit territory
  ✔ PROGRESSIVE TRAILING SL: tighten as profit grows
  ✔ CONSERVATIVE mode REMOVED (9.1% WR → 83% of losses)
  ✔ Dead-zone block: only trades 19:00-00:59 UTC
  ✔ BTC 4H macro gate for altcoin LONGs
  ✔ Volume >1.3x mandatory gate
  ✔ Session-weighted sizing: +25% in 19:00-20:59 UTC
  ✔ Pyramid trigger at +2.5%
  ✔ RSI divergence bonus in signals
  ✔ Updated signal weights per loss analysis
  ✔ LTF veto: skip if 5m MACD + 3m cross both negative
  ✔ Fear & Greed macro gate
  ✔ Auto-blacklist 0% WR symbols (≥5 trades)
  ✔ Whale intelligence integration
"""
import asyncio, json, os, time, logging, smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from bybit_client       import BybitClient
from scanner             import VScanner, parse_klines, atr, ema, p_funding, p_ls, p_oi, p_ob, p_liq, price_precision
from signals             import QuantSignalEngine, sl_tp
from whale_intelligence  import WhaleEngine
from compound_engine     import CompoundEngine, DAILY_REQUIRED_PCT, EPOCH_DAYS
from db import (
    init_db, gp, sp, open_trade, close_trade, update_pos_price,
    get_open_positions, get_trades, all_time_stats, log_capital,
    get_symbol_stats, all_symbol_stats, snap_hour,
    close_epoch_record, open_epoch_record, get_all_epochs,
)

logger = logging.getLogger(__name__)

REPORT_EMAIL = os.getenv("REPORT_EMAIL", "lgwaithaka@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST",    "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER",    "")
SMTP_PASS    = os.getenv("SMTP_PASS",    "")

# ── Trading Window (from strategy review: 19:00-00:59 UTC) ──────────────────
TRADING_HOURS_START = 19   # UTC hour
TRADING_HOURS_END   = 1    # UTC hour (wraps midnight: 19,20,21,22,23,0)
DEAD_ZONE_START     = 20   # UTC — confirmed 0% WR zone
DEAD_ZONE_END       = 24   # UTC

# ── Asset Classes ────────────────────────────────────────────────────────────
ASSET_CLASSES = {
    "BTC":   ["BTCUSDT"],
    "ETH":   ["ETHUSDT"],
    "SOL":   ["SOLUSDT"],
    "BNB":   ["BNBUSDT"],
    "DOGE":  ["DOGEUSDT"],
    "XRP":   ["XRPUSDT"],
    "AVAX":  ["AVAXUSDT"],
    "LINK":  ["LINKUSDT"],
    "ADA":   ["ADAUSDT"],
    "MEME":  ["PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT"],
    "LAYER2":["ARBUSDT", "OPUSDT", "MATICUSDT", "STRKUSDT"],
    "DEFI":  ["UNIUSDT", "AAVEUSDT"],
}


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
        trend_vals, range_vals = [], []
        for sym, k in klines_dict.items():
            if k is None or len(k) < 22:
                continue
            closes = k[:, 4]; highs = k[:, 2]; lows = k[:, 3]
            price  = float(closes[-1])
            if price <= 0:
                continue
            e8  = ema(closes, 8)
            e21 = ema(closes, 21)
            slope = abs((e8[-1] - e8[-5]) / (e8[-5] + 1e-9)) if len(e8) >= 5 else 0
            trend_vals.append(slope)
            rng = float((highs[-10:].max() - lows[-10:].min()) / price)
            range_vals.append(rng)
        if not trend_vals:
            return
        avg_trend = float(np.mean(trend_vals))
        avg_range = float(np.mean(range_vals))
        self.trend_strength = avg_trend
        self.volatility_pct = avg_range * 100
        if avg_range > 0.065:
            self.regime = self.VOLATILE
        elif avg_trend > 0.0025:
            self.regime = self.TRENDING
        else:
            self.regime = self.RANGING
        self.last_updated = int(time.time())

    def adjustments(self) -> Dict:
        if self.regime == self.TRENDING:
            return {"conf_bonus": -3, "tp_bonus": 0.6, "sl_bonus": -0.1, "lev_bonus": 2,
                    "label": "TRENDING — momentum entries, let winners run"}
        elif self.regime == self.VOLATILE:
            return {"conf_bonus": +5, "tp_bonus": -0.4, "sl_bonus": 0.3, "lev_bonus": -3,
                    "label": "VOLATILE — selective entries, wider stops"}
        else:
            return {"conf_bonus": +2, "tp_bonus": -0.2, "sl_bonus": 0.0, "lev_bonus": 0,
                    "label": "RANGING — mean-reversion, tighter targets"}


class CompoundTradingEngine:
    def __init__(self):
        self.client:   Optional[BybitClient]  = None
        self.scanner:  Optional[VScanner]     = None
        self.signals   = QuantSignalEngine()
        self.compound  = CompoundEngine()
        self.regime    = MarketRegime()
        self.whales    = WhaleEngine()
        self.running        = False
        self.status         = "IDLE"
        self.start_time     = int(time.time())
        self.balance        = 0.0
        self.capital_auto_detected = True
        self.live_positions: Dict[str, Dict]   = {}
        self.cooldown_until: Dict[str, float]  = {}
        self.losing_symbols: Set[str]          = set()
        self._klines_cache: Dict[str, np.ndarray] = {}
        self.scan_results: List[Dict]  = []
        self.last_scan_ts  = 0
        self.last_hour_ts  = 0
        self.errors: List[str] = []

    # ── Init ─────────────────────────────────────────────────────────────

    def init(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client  = BybitClient(api_key, api_secret, testnet)
        self.scanner = VScanner(self.client)
        init_db()

        epoch_num = int(gp("current_epoch",  "1"))
        epoch_ts  = int(gp("epoch_start_ts",  str(int(time.time()))))
        saved_initial = gp("initial_capital", None)
        saved_epoch   = gp("epoch_start_bal", None)
        first_run = (saved_initial is None or saved_initial == "0.0"
                     or saved_epoch is None or saved_epoch == "0.0")

        if first_run:
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
        logger.info(f"Engine init | Epoch {epoch_num} | bal=${epoch_bal:.4f}")

    # ── Balance ──────────────────────────────────────────────────────────

    async def refresh_balance(self) -> float:
        try:
            for at in ["UNIFIED", "CONTRACT"]:
                resp = await self.client.wallet(at)
                if resp.get("retCode", -1) != 0:
                    continue
                for acc in resp.get("result", {}).get("list", []):
                    v = float(acc.get("totalWalletBalance", "0") or "0")
                    if v > 0:
                        self.balance = v
                        return v
                    for coin in acc.get("coin", []):
                        if coin.get("coin") == "USDT":
                            cv = float(coin.get("walletBalance", "0") or "0")
                            if cv > 0:
                                self.balance = cv
                                return cv
        except Exception as e:
            logger.error(f"refresh_balance: {e}")
        return self.balance

    async def refresh_live_positions(self):
        try:
            resp = await self.client.positions()
            self.live_positions = {
                p["symbol"]: p
                for p in resp.get("result", {}).get("list", [])
                if float(p.get("size", "0")) > 0
            }
        except Exception as e:
            logger.error(f"refresh_live_positions: {e}")

    # ── Trading Window Check ─────────────────────────────────────────────

    def _in_trading_window(self) -> Tuple[bool, str]:
        """
        Strategy doc: Only trade 19:00-00:59 UTC.
        Dead zone 20:00-23:00 UTC had 0% win rate.
        Configurable via params.
        """
        h = datetime.now(timezone.utc).hour
        use_window = gp("use_trading_window", "true")
        if use_window and use_window.lower() == "false":
            return True, "24/7 mode"

        # Trading window: 19:00-00:59 UTC
        if h >= TRADING_HOURS_START or h < TRADING_HOURS_END:
            # Inside window — check dead zone
            if DEAD_ZONE_START <= h < DEAD_ZONE_END:
                return False, f"Dead zone ({h}:00 UTC — 0% WR confirmed)"
            return True, f"Active window ({h}:00 UTC)"
        return False, f"Outside trading window ({h}:00 UTC)"

    def _session_size_multiplier(self) -> float:
        """
        Strategy doc: +25% sizing in 19:00-20:59 UTC (peak liquidity).
        """
        h = datetime.now(timezone.utc).hour
        if h in (19,):  # US close = 85.2% WR
            return 1.25
        elif h == 0:     # Asia open = 88.2% WR
            return 1.20
        return 1.0

    # ── Fetch Market Data ────────────────────────────────────────────────

    async def fetch_data(self, symbol: str) -> Optional[Dict]:
        try:
            results = await asyncio.gather(
                self.client.klines(symbol, "3",   80),
                self.client.klines(symbol, "5",  130),
                self.client.klines(symbol, "15",  80),
                self.client.klines(symbol, "60",  60),
                self.client.klines(symbol, "240", 60),
                self.client.funding(symbol),
                self.client.ls_ratio(symbol),
                self.client.open_interest(symbol, "1h"),
                self.client.orderbook(symbol, 50),
                self.client.liquidations(symbol),
                return_exceptions=True,
            )
            k3, k5, k15, k1h, k4h, fr, lsr, oi, ob, liq = results
            k5p = parse_klines(k5) if not isinstance(k5, Exception) else None
            if k5p is not None:
                self._klines_cache[symbol] = k5p
            return {
                "k3m":     parse_klines(k3)   if not isinstance(k3,  Exception) else None,
                "k5m":     k5p,
                "k15m":    parse_klines(k15)  if not isinstance(k15, Exception) else None,
                "k1h":     parse_klines(k1h)  if not isinstance(k1h, Exception) else None,
                "k4h":     parse_klines(k4h)  if not isinstance(k4h, Exception) else None,
                "funding": p_funding(fr)   if not isinstance(fr,  Exception) else 0.0,
                "ls":      p_ls(lsr)       if not isinstance(lsr, Exception) else 0.5,
                "oi_pct":  p_oi(oi)        if not isinstance(oi,  Exception) else 0.0,
                "ob_imb":  p_ob(ob)        if not isinstance(ob,  Exception) else 0.0,
                "liqs":    p_liq(liq)      if not isinstance(liq, Exception) else {},
            }
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}")
            return None

    # ── Profit Targets + Partial Close ───────────────────────────────────

    async def check_profit_targets(self):
        """
        TRANCHE PROFIT-TAKING — hold positions longer, close in 3 stages:
          Tranche 1: Close 40% at  15% return on margin
          Tranche 2: Close 30% at  20% return on margin
          Tranche 3: Close 30% at  25% return on margin

        BREAKEVEN PROTECTION — once position is in ANY profit territory,
        move SL to entry price (above entry for LONG, below for SHORT).
        This locks in zero-loss once we're positive.
        """
        db_open = get_open_positions()
        for pos in db_open:
            sym      = pos["symbol"]
            entry    = float(pos.get("entry_price", 0) or 0)
            qty      = float(pos.get("qty", 0) or 0)
            side     = pos.get("side", "Buy")
            trade_id = pos.get("trade_id")
            unr      = float(pos.get("unrealised_pnl", 0) or 0)
            cur      = float(pos.get("current_price", 0) or 0)
            leverage = int(pos.get("leverage", 20) or 20)

            if entry <= 0 or qty <= 0 or cur <= 0:
                continue

            # Calculate return on MARGIN (leverage-adjusted) — this is what matters
            notional  = entry * qty
            margin    = notional / leverage if leverage > 0 else notional
            margin_pct = (unr / margin * 100) if margin > 0 else 0   # % return on invested margin

            prec   = price_precision(cur)
            old_sl = float(pos.get("sl_price", 0) or 0)
            tag    = pos.get("tag") or ""

            # ── BREAKEVEN SL — once in ANY profit, move SL to entry ──────
            if margin_pct > 0.5:   # Small buffer to avoid noise
                if side == "Buy":
                    be_sl = round(entry * 1.001, prec)   # Just above entry (0.1% buffer for fees)
                    if old_sl == 0 or old_sl < be_sl:
                        try:
                            await self.client.set_tpsl(sym, sl=str(be_sl))
                            if "BE" not in tag:
                                logger.info(f"BREAKEVEN SL | {sym} sl→{be_sl:.4f} (entry={entry:.4f}) +{margin_pct:.1f}%")
                                sp(f"be_{trade_id}", "1")
                        except Exception as e:
                            logger.debug(f"BE SL {sym}: {e}")
                else:  # Sell/Short
                    be_sl = round(entry * 0.999, prec)   # Just below entry
                    if old_sl == 0 or old_sl > be_sl:
                        try:
                            await self.client.set_tpsl(sym, sl=str(be_sl))
                            if "BE" not in tag:
                                logger.info(f"BREAKEVEN SL | {sym} sl→{be_sl:.4f} (entry={entry:.4f}) +{margin_pct:.1f}%")
                                sp(f"be_{trade_id}", "1")
                        except Exception as e:
                            logger.debug(f"BE SL {sym}: {e}")

            # ── TRANCHE 1: Close 40% at 15% return on margin ────────────
            t1_done = gp(f"t1_{trade_id}", None)
            if margin_pct >= 15.0 and not t1_done:
                try:
                    close_qty = round(qty * 0.40, 3)
                    if close_qty > 0 and close_qty * cur >= 1.0:
                        await self.client.close_pos(sym, side, close_qty)
                        sp(f"t1_{trade_id}", "1")
                        logger.info(f"TRANCHE 1 | {sym} 40% ({close_qty}) @ +{margin_pct:.1f}% margin return")
                except Exception as e:
                    logger.debug(f"Tranche 1 {sym}: {e}")

            # ── TRANCHE 2: Close 30% at 20% return on margin ────────────
            t2_done = gp(f"t2_{trade_id}", None)
            if margin_pct >= 20.0 and t1_done and not t2_done:
                try:
                    # 30% of ORIGINAL qty (not remaining)
                    close_qty = round(qty * 0.30, 3)
                    if close_qty > 0 and close_qty * cur >= 1.0:
                        await self.client.close_pos(sym, side, close_qty)
                        sp(f"t2_{trade_id}", "1")
                        logger.info(f"TRANCHE 2 | {sym} 30% ({close_qty}) @ +{margin_pct:.1f}% margin return")
                        # Tighten SL to lock in gains
                        if side == "Buy":
                            lock_sl = round(entry * 1.005, prec)  # 0.5% above entry
                            await self.client.set_tpsl(sym, sl=str(lock_sl))
                        else:
                            lock_sl = round(entry * 0.995, prec)
                            await self.client.set_tpsl(sym, sl=str(lock_sl))
                except Exception as e:
                    logger.debug(f"Tranche 2 {sym}: {e}")

            # ── TRANCHE 3: Close remaining 30% at 25% return on margin ──
            t3_done = gp(f"t3_{trade_id}", None)
            if margin_pct >= 25.0 and t1_done and t2_done and not t3_done:
                try:
                    # Close all remaining (30% of original)
                    close_qty = round(qty * 0.30, 3)
                    if close_qty > 0 and close_qty * cur >= 1.0:
                        await self.client.close_pos(sym, side, close_qty)
                        sp(f"t3_{trade_id}", "1")
                        logger.info(f"TRANCHE 3 | {sym} final 30% ({close_qty}) @ +{margin_pct:.1f}% margin return")
                except Exception as e:
                    logger.debug(f"Tranche 3 {sym}: {e}")

            # ── TRAILING STOP for remaining position after all tranches ──
            # After T3 the position should be fully closed. But if any qty remains
            # (rounding, partial fills), trail it tightly.
            if t1_done and t2_done and t3_done:
                try:
                    # Check if position still exists on exchange
                    if sym in self.live_positions:
                        live = self.live_positions[sym]
                        remaining = float(live.get("size", "0") or "0")
                        if remaining > 0:
                            await self.client.close_pos(sym, side, remaining)
                            logger.info(f"CLEANUP | {sym} closed remaining {remaining}")
                except Exception:
                    pass

            # ── PROGRESSIVE TRAILING SL for positions above 15% ──────────
            if margin_pct >= 15.0 and t1_done:
                atr_v = entry * 0.01
                try:
                    raw5 = await self.client.klines(sym, "5", 30)
                    k5   = parse_klines(raw5)
                    if k5 is not None:
                        atr_v = atr(k5, 14)
                except Exception:
                    pass

                # Trail SL tighter as profit grows
                if side == "Buy":
                    if   margin_pct >= 25.0: trail_sl = round(cur - atr_v * 0.5, prec)
                    elif margin_pct >= 20.0: trail_sl = round(cur - atr_v * 1.0, prec)
                    else:                    trail_sl = round(cur - atr_v * 1.5, prec)
                    if trail_sl > old_sl + atr_v * 0.05:
                        try:
                            await self.client.set_tpsl(sym, sl=str(trail_sl))
                            logger.info(f"TRAIL SL | {sym} +{margin_pct:.1f}% sl→{trail_sl:.4f}")
                        except Exception:
                            pass
                else:
                    if   margin_pct >= 25.0: trail_sl = round(cur + atr_v * 0.5, prec)
                    elif margin_pct >= 20.0: trail_sl = round(cur + atr_v * 1.0, prec)
                    else:                    trail_sl = round(cur + atr_v * 1.5, prec)
                    if old_sl == 0 or trail_sl < old_sl - atr_v * 0.05:
                        try:
                            await self.client.set_tpsl(sym, sl=str(trail_sl))
                            logger.info(f"TRAIL SL | {sym} +{margin_pct:.1f}% sl→{trail_sl:.4f}")
                        except Exception:
                            pass

    # ── Can Open Gate ────────────────────────────────────────────────────

    def _can_open(self, symbol: str, signal: str) -> Tuple[bool, str]:
        db_open   = get_open_positions()
        open_syms = {p["symbol"] for p in db_open}

        if symbol in open_syms:
            return False, "Already open"
        if symbol in self.losing_symbols:
            return False, "Cooling off — last trade lost"
        if self.cooldown_until.get(symbol, 0) > time.time():
            return False, "Cooldown active"
        if len(open_syms) >= self.compound.compute_max_concurrent():
            return False, "Max concurrent reached"
        return True, "OK"

    # ── Trade Execution ──────────────────────────────────────────────────

    async def execute(self, symbol: str, analysis: Dict, meta: Dict) -> bool:
        ce     = self.compound
        signal = analysis["signal"]
        conf   = analysis["confidence"]
        comp   = analysis["composite"]
        mode   = ce.state.mode
        adj    = self.regime.adjustments()

        raw5 = await self.client.klines(symbol, "5", 50)
        k5   = parse_klines(raw5)
        if k5 is None or len(k5) == 0:
            return False

        price   = float(k5[-1, 4])
        atr_val = atr(k5, 14)
        ss      = get_symbol_stats(symbol)
        sl_mult = max(1.5, min(ss.get("sl_mult", 1.8) + adj["sl_bonus"], 2.8))
        tp_mult = max(2.5, min(ss.get("tp_mult", 3.0) + adj["tp_bonus"], 6.0))
        leverage = max(8, min(ce.compute_leverage(meta.get("range_pct", 5.0)) + adj["lev_bonus"], 25))

        # Session-weighted sizing
        session_mult = self._session_size_multiplier()
        risk_pct     = ce.compute_risk_pct() * session_mult
        risk_usd     = self.balance * risk_pct
        min_notional = float(gp("min_notional_usdt", "5.5"))
        qty          = (risk_usd * leverage) / price

        if qty * price < min_notional:
            qty = (min_notional * 1.05) / price
        max_n = self.balance * 0.40 * leverage
        if qty * price > max_n:
            qty = max_n / price

        if price >= 1000:  qty = round(qty, 3)
        elif price >= 1:   qty = round(qty, 2)
        else:              qty = max(round(qty, 0), 1)

        if self.balance < (min_notional / leverage) * 1.5:
            logger.warning(f"Balance too low for {symbol}")
            return False

        side = "Buy" if signal == "LONG" else "Sell"
        k5_cached = self._klines_cache.get(symbol)
        sl, _tp_unused = sl_tp(side, price, atr_val, k5m=k5_cached, sl_mult=sl_mult, tp_mult=tp_mult)

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception:
            pass

        # NO exchange-side TP — positions are managed by check_profit_targets()
        # which closes in tranches: 40% at 15%, 30% at 20%, 30% at 25%
        resp = await self.client.place_order(symbol, side, qty, sl=sl, tp=None)
        if resp.get("retCode", -1) != 0:
            err = resp.get("retMsg", "?")
            logger.error(f"Order fail {symbol}: {err}")
            self.errors.append(f"{datetime.now().isoformat()} | {symbol} | {err}")
            return False

        order_id = resp.get("result", {}).get("orderId", "")
        epoch    = ce.state.epoch_num
        open_trade(
            epoch=epoch, symbol=symbol, side=side, signal=signal,
            confidence=conf, composite=comp,
            entry_price=price, qty=qty, leverage=leverage,
            sl=sl, tp=0, order_id=order_id,
            tag=f"EP{epoch}_L{leverage}_{self.regime.regime[:3]}_{mode[:3]}",
            mode=mode, components=analysis.get("components", {}),
            vol_score=meta.get("vol_score", 0), range_pct=meta.get("range_pct", 0),
        )
        logger.info(f"OPEN [{mode}/{self.regime.regime}] {symbol} {signal} "
                     f"qty={qty}@${price:.4f} SL={sl:.4f} L={leverage}x | TP managed programmatically")
        return True

    # ── Position Monitor ─────────────────────────────────────────────────

    async def monitor(self):
        await self.refresh_live_positions()
        db_open = get_open_positions()
        initial = float(gp("epoch_start_bal", "0.0") or "0.0") or self.balance or 10.0

        self.losing_symbols = set()
        for pos in db_open:
            if (pos.get("unrealised_pnl") or 0) < -0.05:
                self.losing_symbols.add(pos["symbol"])

        for pos in db_open:
            sym = pos["symbol"]
            tid = pos["trade_id"]
            if sym in self.live_positions:
                live = self.live_positions[sym]
                try:
                    unr  = float(live.get("unrealisedPnl", "0") or "0")
                    cprc = float(live.get("markPrice",     "0") or "0")
                    update_pos_price(sym, cprc, unr)
                except Exception:
                    pass
            else:
                pnl = 0.0
                exit_p = 0.0
                try:
                    pr  = await self.client.closed_pnl(sym, 10)
                    pls = pr.get("result", {}).get("list", [])
                    if pls:
                        pnl    = float(pls[0].get("closedPnl",    "0") or "0")
                        exit_p = float(pls[0].get("avgExitPrice", "0") or "0")
                except Exception:
                    pass

                if exit_p == 0:
                    continue  # Still live on exchange

                outcome = close_trade(tid, exit_p, pnl, initial)
                win     = outcome == "WIN"
                self.compound.record_outcome(win, pnl, self.balance)
                if win:
                    self.cooldown_until[sym] = time.time() + 5
                else:
                    ss_cd = get_symbol_stats(sym)
                    net_losses = max(0, ss_cd.get("losses", 0) - ss_cd.get("wins", 0))
                    cd_secs = min(600, 120 + (net_losses * 60))
                    self.cooldown_until[sym] = time.time() + cd_secs
                if not win:
                    self.losing_symbols.add(sym)
                logger.info(f"CLOSED | {sym} | {outcome} | ${pnl:.4f}")

        # Pyramid winners
        await self._pyramid_winners()

    async def _pyramid_winners(self):
        """Pyramid at +2.5% (raised from +0.8% per strategy doc)."""
        mode = self.compound.state.mode
        if mode not in ("AGGRESSIVE", "TURBO"):
            return

        db_open = get_open_positions()
        for pos in db_open:
            sym   = pos["symbol"]
            entry = pos.get("entry_price") or 0
            side  = pos.get("side") or ""
            qty   = pos.get("qty") or 0
            tag   = pos.get("tag") or ""

            if entry <= 0 or qty <= 0 or "PYR" in tag:
                continue

            live_p = self.live_positions.get(sym, {})
            cur = float(live_p.get("markPrice", "0") or 0)
            if cur <= 0:
                continue

            # Raised to +2.5% (from +0.8%)
            min_profit = 0.025
            if side == "Buy" and (cur - entry) / entry < min_profit:
                continue
            if side == "Sell" and (entry - cur) / entry < min_profit:
                continue

            pyramid_qty = round(qty * 0.50, 3)
            min_notional = float(gp("min_notional_usdt", "5.5"))
            if pyramid_qty * cur < min_notional:
                continue

            try:
                resp = await self.client.place_order(sym, side, str(pyramid_qty))
                if resp.get("retCode", 0) == 0:
                    logger.info(f"PYRAMID | {sym} +{pyramid_qty} @ ${cur:.4f}")
                    open_trade(
                        epoch=pos.get("epoch", 1), symbol=sym, side=side,
                        signal=pos.get("signal", ""), confidence=pos.get("confidence", 50),
                        composite=0.0, entry_price=cur, qty=pyramid_qty,
                        leverage=pos.get("leverage", 10),
                        sl=pos.get("sl_price", 0), tp=pos.get("tp_price", 0),
                        order_id=resp.get("result", {}).get("orderId", ""),
                        tag=f"PYR_{tag}", mode=mode, components={},
                    )
            except Exception as e:
                logger.debug(f"Pyramid {sym}: {e}")

    # ── Scan & Trade ─────────────────────────────────────────────────────

    async def scan_and_trade(self):
        ce = self.compound

        # Circuit breakers
        if ce.check_circuit_breakers(
            self.balance,
            epoch_max_dd=float(gp("epoch_max_dd_pct", "35")) / 100,
            daily_max_dd=float(gp("daily_max_dd_pct", "25")) / 100,
        ):
            return

        # Trading window check
        in_window, window_reason = self._in_trading_window()
        if not in_window:
            logger.debug(f"Skipping scan: {window_reason}")
            sp("trading_window_status", window_reason)
            return

        sp("trading_window_status", window_reason)
        await self.refresh_balance()

        # Fear & Greed
        fg = {}
        fg_val = 50
        fg_conf_adj = 0
        try:
            import aiohttp
            async with aiohttp.ClientSession() as _sess:
                async with _sess.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as _r:
                    if _r.status == 200:
                        _fng = await _r.json()
                        fg    = _fng.get("data", [{}])[0]
                        fg_val = int(fg.get("value", 50))
                        if   fg_val <= 20: fg_conf_adj = -3
                        elif fg_val <= 35: fg_conf_adj = -1
                        elif fg_val >= 80: fg_conf_adj = +3
                        elif fg_val >= 65: fg_conf_adj = +1
        except Exception:
            pass

        # Auto-blacklist: 0% WR over ≥5 trades
        auto_blacklist: set = set()
        try:
            for _ss in all_symbol_stats():
                if _ss.get("wins", 0) + _ss.get("losses", 0) >= 5 and _ss.get("wins", 0) == 0:
                    auto_blacklist.add(_ss["symbol"])
        except Exception:
            pass

        # BTC 4H macro gate — update before scanning
        try:
            btc_4h = await self.client.klines("BTCUSDT", "240", 60)
            btc_4h_parsed = parse_klines(btc_4h)
            self.signals.set_btc_bias(btc_4h_parsed)
        except Exception:
            pass

        mode   = ce.state.mode
        n_syms = int(gp("vol_scan_n", "12"))
        if mode == "TURBO":
            n_syms = min(n_syms + 4, 18)
        elif mode == "AGGRESSIVE":
            n_syms = min(n_syms + 2, 16)

        top_syms   = await self.scanner.scan(n=n_syms)
        adj        = self.regime.adjustments()
        conf_floor = max(35, ce.compute_confidence_floor() + adj["conf_bonus"] + fg_conf_adj)

        results = []
        for meta in top_syms:
            sym = meta["symbol"]
            if self.cooldown_until.get(sym, 0) > time.time():
                continue
            if sym in auto_blacklist:
                continue
            if sym in self.losing_symbols and mode == "NORMAL":
                continue

            data = await self.fetch_data(sym)
            if not data:
                continue

            ss   = get_symbol_stats(sym)
            bias = ss.get("learned_bias", 0.0)
            whale_score = self.whales.get_signal_for(sym)

            an = self.signals.analyze(
                sym=sym, k3m=data["k3m"], k5m=data["k5m"],
                k15m=data["k15m"], k1h=data["k1h"], k4h=data.get("k4h"),
                funding=data["funding"], ls=data["ls"],
                oi_pct=data["oi_pct"], ob_imb=data["ob_imb"],
                liqs=data["liqs"],
                learned_bias=bias + (whale_score * 0.3),
                mode=mode,
            )
            an["whale_score"] = round(whale_score, 3)

            result = {
                "symbol": sym, "signal": an["signal"],
                "confidence": an["confidence"], "composite": an["composite"],
                "leverage": ce.compute_leverage(meta.get("range_pct", 5)),
                "vol_score": meta.get("vol_score", 0),
                "range_pct": meta.get("range_pct", 0),
                "chg_24h": meta.get("chg_24h", 0),
                "regime": self.regime.regime, "mode": mode,
                "ts": int(time.time()), "action": "HOLD",
                "fear_greed": fg_val,
                "fg_label": fg.get("value_classification", "Neutral"),
                "whale_score": an.get("whale_score", 0),
                "volume_ok": an.get("volume_ok", False),
            }

            # LTF veto
            comps = an.get("components", {})
            macd_5m_val = comps.get("macd_5m", 0)
            cross_3m_val = comps.get("cross_3m", 0)
            if isinstance(macd_5m_val, (int, float)) and isinstance(cross_3m_val, (int, float)):
                if macd_5m_val < -0.00001 and cross_3m_val < -0.00001:
                    result["action"] = "LTF_VETO"
                    results.append(result)
                    continue

            # Volume mandatory gate
            if not an.get("volume_ok", False):
                result["action"] = "VOL_GATE"
                results.append(result)
                continue

            if an["signal"] != "HOLD" and an["confidence"] >= conf_floor:
                can, reason = self._can_open(sym, an["signal"])
                if can:
                    ok = await self.execute(sym, an, meta)
                    result["action"] = "OPENED" if ok else "EXEC_FAIL"
                else:
                    result["action"] = f"SKIP:{reason}"

            results.append(result)
            await asyncio.sleep(0.25)

        self.scan_results = results
        self.last_scan_ts = int(time.time())

    # ── Hourly Update ────────────────────────────────────────────────────

    async def hourly_strategy_update(self):
        await self.refresh_balance()
        ce = self.compound
        s  = ce.state

        if self._klines_cache:
            self.regime.update(self._klines_cache)

        h_trades = get_trades(hours=1)
        h_closed = [t for t in h_trades if t["outcome"] not in ("OPEN", None)]
        h_wins   = sum(1 for t in h_closed if t["outcome"] == "WIN")
        h_pnl    = sum(t["pnl_usdt"] or 0 for t in h_closed)
        h_wr     = h_wins / len(h_closed) * 100 if h_closed else 0
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
        sp("hourly_win_rate",  str(round(h_wr, 1)))
        sp("hourly_pnl",       str(round(h_pnl, 4)))

        await self.scanner.scan(force=True)

        # Whale update
        try:
            top_syms = [x["symbol"] for x in self.scanner.top]
            if top_syms and self.client:
                await self.whales.update_all(top_syms, self.client)
                top_ops = self.whales.get_top_opportunities(5)
                sp("whale_summary",       json.dumps(self.whales.last_summary))
                sp("whale_opportunities", json.dumps(top_ops))
                sp("whale_last_update",   str(int(time.time())))
        except Exception as e:
            logger.error(f"Hourly whale update: {e}")

        logger.info(f"Hourly | regime={self.regime.regime} | pnl=${h_pnl:.4f} | wr={h_wr:.1f}%")

    # ── Epoch & Daily ────────────────────────────────────────────────────

    async def check_epoch_boundary(self):
        await self.refresh_balance()
        ce  = self.compound
        old = ce.state.epoch_num
        if ce.advance_epoch(self.balance):
            close_epoch_record(old, self.balance)
            sp("current_epoch",   str(ce.state.epoch_num))
            sp("epoch_start_ts",  str(ce.state.epoch_start_ts))
            sp("epoch_start_bal", str(round(ce.state.epoch_start_bal, 4)))
            open_epoch_record(ce.state.epoch_num, ce.state.epoch_start_bal)
            self.losing_symbols.clear()
            await self.send_epoch_report(old, self.balance, ce.state.epoch_target)

    async def daily_tasks(self):
        await self.refresh_balance()
        self.compound.advance_day(self.balance)
        self.compound.reset_daily_cb()
        self.losing_symbols.clear()
        await self.send_daily_report()

    # ── Main Loop ────────────────────────────────────────────────────────

    async def main_loop(self):
        self.running = True
        self.status  = "RUNNING"

        # Auto-detect balance
        live_bal = await self.refresh_balance()
        logger.info(f"Live Bybit balance: ${live_bal:.4f} USDT")

        if live_bal > 0 and self.capital_auto_detected:
            sp("initial_capital", str(round(live_bal, 4)))
            sp("epoch_start_bal", str(round(live_bal, 4)))
            sp("epoch_start_ts",  str(int(time.time())))
            sp("current_epoch",   "1")
            self.compound.initialise(live_bal, 1, int(time.time()), live_bal)
            open_epoch_record(1, live_bal)

            if   live_bal < 20:  sp("min_notional_usdt", "5.5");  sp("max_concurrent", "3")
            elif live_bal < 100: sp("min_notional_usdt", "5.5");  sp("max_concurrent", "5")
            elif live_bal < 500: sp("min_notional_usdt", "10.0"); sp("max_concurrent", "8")
            else:                sp("min_notional_usdt", "20.0"); sp("max_concurrent", "12")

            self.capital_auto_detected = False
            logger.info(f"Capital auto-set: ${live_bal:.4f}")
        elif live_bal > 0:
            self.balance = live_bal
        else:
            logger.warning("Cannot read Bybit balance — check API keys")

        scan_secs = int(gp("scan_interval_s", "30"))
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
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop: {e}", exc_info=True)
                self.errors.append(f"{datetime.now().isoformat()} | {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(max(0, scan_secs - (time.time() - t0)))

        self.status = "STOPPED"
        self.running = False
        if self.client:
            await self.client.close()

    async def stop(self):
        self.running = False

    # ── Reports ──────────────────────────────────────────────────────────

    async def send_epoch_report(self, epoch_num, achieved, next_target):
        try:
            stake = self.compound.state.epoch_start_bal
            pct   = (achieved - stake) / stake * 100 if stake else 0
            body  = (f"EPOCH {epoch_num} | ${stake:.2f}→${achieved:.2f} "
                     f"({pct:+.1f}%) | {'✓ DOUBLED' if achieved >= stake * 2 else '✗ MISSED'}\n"
                     f"Next epoch: ${achieved:.2f}→${next_target:.2f}\n")
            self._send_email(f"[Bybit] Ep{epoch_num} ${achieved:.2f}", body)
        except Exception as e:
            logger.error(f"epoch report: {e}")

    async def send_daily_report(self):
        try:
            ce    = self.compound
            stats = all_time_stats()
            body  = (f"Day {ce.state.day_in_epoch}/{EPOCH_DAYS} Ep{ce.state.epoch_num} | "
                     f"${self.balance:.4f} | Mode:{ce.state.mode} | "
                     f"Regime:{self.regime.regime}\n"
                     f"WR:{stats['win_rate']}% PnL:${stats['total_pnl']:.4f}\n")
            self._send_email(f"[Bybit] Day{ce.state.day_in_epoch} ${self.balance:.2f}", body)
        except Exception as e:
            logger.error(f"daily report: {e}")

    def _send_email(self, subject, body):
        if not SMTP_USER or not SMTP_PASS:
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SMTP_USER
            msg["To"]      = REPORT_EMAIL
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as sv:
                sv.starttls()
                sv.login(SMTP_USER, SMTP_PASS)
                sv.send_message(msg)
        except Exception as e:
            logger.error(f"email: {e}")


# Global instance
engine = CompoundTradingEngine()
