"""
engine.py v3.0 — ByBitDouble Compound Trading Engine
Strategy improvements from May 2026 review:
  • Dead-zone block: NO new entries 01:00–18:59 UTC (recovered $3.06/week in testing)
  • Partial close: 50% at +5%, remainder trails to +10%+ (raises avg win +35%)
  • Session sizing: +25% notional during 19:00–20:59 UTC peak window
  • Pyramid trigger: raised from +0.8% → +2.5% to avoid noise adds
  • BTC 4H macro gate passed to signals engine
  • CONSERVATIVE mode removed — minimum is NORMAL
  • SL minimum: 1.8× ATR (was 1.2× — prevents noise-outs)
  • All API endpoint handlers for dashboard
"""
import asyncio, json, os, time, logging, smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from scanner import VScanner, parse_klines, atr, p_funding, p_ls, p_oi, p_ob, p_liq, price_precision
from signals import QuantSignalEngine, sl_tp
from db import (
    init_db, gp, sp, open_trade, close_trade, update_pos_price,
    get_open_positions, get_trades, all_time_stats, log_capital,
    get_symbol_stats, snap_hour, close_epoch_record, open_epoch_record,
    get_all_epochs, get_capital_log, all_symbol_stats, all_params,
)

logger = logging.getLogger(__name__)

REPORT_EMAIL = os.getenv("REPORT_EMAIL", "lgwaithaka@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")

# ── Trading Window (UTC) ──────────────────────────────────────────────────────
# Strategy: block 01:00–18:59 UTC (dead zone = 0 wins, 17 losses in Week 1)
# Active window: 19:00–00:59 UTC
DEAD_ZONE_START = 1   # 01:00 UTC
DEAD_ZONE_END   = 19  # 19:00 UTC (exclusive — 19 is allowed)

# ── Session sizing multipliers ────────────────────────────────────────────────
# Data: 19:00 UTC = 85.2% WR, 00:00 UTC = 88.2% WR
SESSION_MULT = {
    "PEAK":   1.25,   # 19:00–20:59 UTC: US close, peak liquidity
    "NORMAL": 1.00,   # 21:00–00:59 UTC
    "REDUCED":0.80,   # Outside active window (should be blocked, safety fallback)
}


def _trading_window_ok() -> bool:
    """Returns True if current UTC hour is in active trading window."""
    h = datetime.now(timezone.utc).hour
    # Dead zone: 01:00–18:59 UTC → block entries
    return not (DEAD_ZONE_START <= h < DEAD_ZONE_END)


def _session_multiplier() -> Tuple[str, float]:
    """Returns (label, size_multiplier) based on UTC hour."""
    h = datetime.now(timezone.utc).hour
    if 19 <= h <= 20:
        return "PEAK", SESSION_MULT["PEAK"]
    if h == 0 or (21 <= h <= 23):
        return "NORMAL", SESSION_MULT["NORMAL"]
    return "REDUCED", SESSION_MULT["REDUCED"]


# ── Asset class groupings ─────────────────────────────────────────────────────
ASSET_CLASSES = {
    "BTC":    ["BTCUSDT"],
    "ETH":    ["ETHUSDT", "WETHUSDT"],
    "SOL":    ["SOLUSDT"],
    "BNB":    ["BNBUSDT"],
    "DOGE":   ["DOGEUSDT"],
    "XRP":    ["XRPUSDT"],
    "AVAX":   ["AVAXUSDT"],
    "LINK":   ["LINKUSDT"],
    "ADA":    ["ADAUSDT"],
    "MEME":   ["PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT"],
    "LAYER2": ["ARBUSDT", "OPUSDT", "MATICUSDT", "STRKUSDT"],
    "DEFI":   ["UNIUSDT", "AAVEUSDT", "CRVUSDT"],
    "OTHER":  [],
}


def get_asset_class(symbol: str) -> str:
    for cls, syms in ASSET_CLASSES.items():
        if symbol in syms:
            return cls
    for cls in ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "AVAX", "LINK", "ADA"]:
        if symbol.startswith(cls):
            return cls
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
            if k is None or len(k) < 22:
                continue
            closes = k[:, 4]; highs = k[:, 2]; lows = k[:, 3]
            price  = float(closes[-1])
            if price <= 0:
                continue
            e8  = _ema(closes, 8)
            e21 = _ema(closes, 21)
            slope = abs((e8[-1] - e8[-5]) / (e8[-5] + 1e-9))
            trend_vals.append(slope)
            rng = (highs[-10:].max() - lows[-10:].min()) / price
            range_vals.append(float(rng))

        if not trend_vals:
            return
        avg_trend = float(np.mean(trend_vals))
        avg_range = float(np.mean(range_vals))
        self.trend_strength = avg_trend
        self.volatility_pct = avg_range * 100
        if avg_range > 0.065:    self.regime = self.VOLATILE
        elif avg_trend > 0.0025: self.regime = self.TRENDING
        else:                    self.regime = self.RANGING
        self.last_updated = int(time.time())

    def adjustments(self) -> Dict:
        if self.regime == self.TRENDING:
            return {"conf_bonus": -3, "tp_bonus": 0.6, "sl_bonus": -0.1, "lev_bonus": 2,
                    "label": "TRENDING — momentum entries, let winners run"}
        elif self.regime == self.VOLATILE:
            return {"conf_bonus": +5, "tp_bonus": -0.4, "sl_bonus": 0.3, "lev_bonus": -3,
                    "label": "VOLATILE — selective entries, wider stops, quick TP"}
        else:
            return {"conf_bonus": +2, "tp_bonus": -0.2, "sl_bonus": 0.0, "lev_bonus": 0,
                    "label": "RANGING — mean-reversion, tighter targets"}


class CompoundState:
    """Minimal compound engine state (no external compound_engine.py dependency)."""
    def __init__(self):
        self.mode              = "NORMAL"
        self.epoch_num         = 1
        self.epoch_start_ts    = int(time.time())
        self.epoch_start_bal   = 0.0
        self.epoch_target      = 0.0
        self.day_in_epoch      = 1
        self.streak            = 0
        self.consecutive_losses= 0
        self.rolling_wins      = 0
        self.rolling_total     = 0

    def win_rate(self) -> float:
        if self.rolling_total == 0:
            return 0.5
        return self.rolling_wins / self.rolling_total

    def compute_risk_pct(self) -> float:
        base = 0.12  # 12% minimum per strategy fix
        if self.mode == "TURBO":        return min(0.25, base * 2.0)
        if self.mode == "AGGRESSIVE":   return min(0.20, base * 1.5)
        return base

    def compute_leverage(self, range_pct: float = 5.0) -> int:
        base = 15
        if range_pct > 8:   base = 10
        elif range_pct < 3: base = 20
        if self.mode == "TURBO":      base = min(25, base + 5)
        if self.mode == "AGGRESSIVE": base = min(22, base + 3)
        return max(8, base)

    def compute_max_concurrent(self) -> int:
        return int(gp("max_concurrent", "8") or "8")

    def compute_confidence_floor(self) -> float:
        floors = {"NORMAL": 68.0, "AGGRESSIVE": 62.0, "TURBO": 55.0}
        return floors.get(self.mode, 68.0)

    def update_mode(self, balance: float):
        if self.epoch_start_bal <= 0:
            return
        progress = (balance - self.epoch_start_bal) / (self.epoch_target - self.epoch_start_bal + 1e-9)
        days_left = max(1, 5 - self.day_in_epoch)
        if progress >= 0.8 or (self.streak >= 3 and days_left <= 2):
            self.mode = "TURBO"
        elif progress > 0.4 or self.streak >= 2:
            self.mode = "AGGRESSIVE"
        else:
            self.mode = "NORMAL"

    def record_outcome(self, win: bool, pnl: float):
        self.rolling_total += 1
        if win:
            self.rolling_wins += 1
            self.streak        = max(0, self.streak) + 1
            self.consecutive_losses = 0
        else:
            self.streak        = min(0, self.streak) - 1
            self.consecutive_losses += 1

    def check_circuit_breakers(self, balance: float) -> bool:
        """Returns True if trading should be halted."""
        epoch_max_dd = float(gp("epoch_max_dd_pct", "35") or "35") / 100
        daily_max_dd = float(gp("daily_max_dd_pct",  "25") or "25") / 100
        if self.epoch_start_bal > 0:
            dd = (self.epoch_start_bal - balance) / self.epoch_start_bal
            if dd > epoch_max_dd and self.mode == "NORMAL":
                logger.warning(f"Circuit breaker: epoch DD {dd:.1%} > {epoch_max_dd:.0%}")
                return True
        return False


class CompoundTradingEngine:
    def __init__(self):
        self.client:  Optional[object]      = None
        self.scanner: Optional[VScanner]    = None
        self.signals  = QuantSignalEngine()
        self.state    = CompoundState()
        self.regime   = MarketRegime()

        self.running       = False
        self.status        = "IDLE"
        self.start_time    = int(time.time())
        self.balance       = 0.0
        self.capital_auto_detected = True

        self.live_positions: Dict[str, Dict] = {}
        self.cooldown_until: Dict[str, float] = {}
        self.losing_symbols: Set[str]         = set()
        self._klines_cache: Dict[str, np.ndarray] = {}
        self._btc_4h_cache: Optional[np.ndarray]  = None

        self.scan_results: List[Dict] = []
        self.last_scan_ts  = 0
        self.last_hour_ts  = 0
        self.errors: List[str] = []
        self.alerts = None  # set by server.py if alerting is available

    def init(self, api_key: str, api_secret: str, testnet: bool = False):
        try:
            from bybit_client import BybitClient
            self.client  = BybitClient(api_key, api_secret, testnet)
        except ImportError:
            logger.warning("bybit_client not available — using mock")

        self.scanner = VScanner(self.client)
        init_db()
        self._load_state_from_db()
        logger.info(f"Engine init | Epoch {self.state.epoch_num} | mode={self.state.mode}")

    def _load_state_from_db(self):
        epoch_num  = int(gp("current_epoch",    "1") or "1")
        epoch_ts   = int(gp("epoch_start_ts",   str(int(time.time()))) or str(int(time.time())))
        epoch_bal  = float(gp("epoch_start_bal", "0.0") or "0.0")
        initial    = float(gp("initial_capital", "0.0") or "0.0")
        s = self.state
        s.epoch_num       = epoch_num
        s.epoch_start_ts  = epoch_ts
        s.epoch_start_bal = epoch_bal
        s.epoch_target    = epoch_bal * 2.0 if epoch_bal > 0 else 0.0
        s.day_in_epoch    = max(1, int((time.time() - epoch_ts) / 86400) + 1)
        self.balance      = epoch_bal
        self.capital_auto_detected = (epoch_bal <= 0 or initial <= 0)

    # ── Balance ───────────────────────────────────────────────────────────────

    async def refresh_balance(self) -> float:
        if self.client is None:
            return self.balance
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
        if self.client is None:
            return
        try:
            resp = await self.client.positions()
            self.live_positions = {
                p["symbol"]: p
                for p in resp.get("result", {}).get("list", [])
                if float(p.get("size", "0")) > 0
            }
        except Exception as e:
            logger.error(f"refresh_live_positions: {e}")

    # ── BTC 4H cache ─────────────────────────────────────────────────────────

    async def _refresh_btc_4h(self):
        if self.client is None:
            return
        try:
            raw = await self.client.klines("BTCUSDT", "240", 60)
            self._btc_4h_cache = parse_klines(raw)
        except Exception as e:
            logger.debug(f"BTC 4H fetch: {e}")

    # ── Fetch Market Data ──────────────────────────────────────────────────────

    async def fetch_data(self, symbol: str) -> Optional[Dict]:
        if self.client is None:
            return None
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
                "funding": p_funding(fr)       if not isinstance(fr,  Exception) else 0.0,
                "ls":      p_ls(lsr)           if not isinstance(lsr, Exception) else 0.5,
                "oi_pct":  p_oi(oi)            if not isinstance(oi,  Exception) else 0.0,
                "ob_imb":  p_ob(ob)            if not isinstance(ob,  Exception) else 0.0,
                "liqs":    p_liq(liq)          if not isinstance(liq, Exception) else {},
            }
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}")
            return None

    # ── Partial Close + Trailing Stop ──────────────────────────────────────────

    async def check_profit_targets(self):
        """
        Progressive profit protection with PARTIAL CLOSE at +5% (strategy fix).

        Tiers:
          +5%  → close 50% of position, move SL to breakeven+0.5%
          +10% → trail SL at 1.5× ATR
          +20% → trail SL at 0.7× ATR (tight squeeze)
          No hard cap — winners run until trailing stop catches them.

        Delayed breakeven: no SL movement until +4% (avoids noise-outs).
        """
        db_open = get_open_positions()
        initial = float(gp("epoch_start_bal", "0.0") or "0.0") or self.balance or 10.0

        for pos in db_open:
            sym      = pos["symbol"]
            entry    = float(pos.get("entry_price", 0) or 0)
            qty      = float(pos.get("qty",         0) or 0)
            side     = pos.get("side", "Buy")
            trade_id = pos.get("trade_id")
            unr      = float(pos.get("unrealised_pnl", 0) or 0)
            cur      = float(pos.get("current_price",  0) or 0)
            partial  = bool(pos.get("partial_closed",  False))

            if entry <= 0 or qty <= 0 or cur <= 0:
                continue
            notional = entry * qty
            unr_pct  = (unr / notional * 100) if notional > 0 else 0.0

            # Delayed intervention — no changes until +4%
            if unr_pct < 4.0:
                continue

            # Fetch fresh ATR
            atr_v = entry * 0.01
            try:
                raw5 = await self.client.klines(sym, "5", 30)
                k5   = parse_klines(raw5)
                if k5 is not None:
                    atr_v = atr(k5, 14)
            except Exception:
                pass

            prec   = price_precision(cur)
            old_sl = float(pos.get("sl_price", 0) or 0)
            old_tp = float(pos.get("tp_price", 0) or 0)

            # ── PARTIAL CLOSE at +5% (first trigger only) ─────────────────
            if unr_pct >= 5.0 and not partial and self.client is not None:
                close_qty = round(qty * 0.50, 3)
                close_side = "Sell" if side == "Buy" else "Buy"
                try:
                    resp = await self.client.place_order(
                        sym, close_side, close_qty,
                        reduce_only=True, order_type="Market"
                    )
                    if resp.get("retCode", -1) == 0:
                        logger.info(f"PARTIAL CLOSE | {sym} 50% at +{unr_pct:.1f}% | qty={close_qty}")
                        # Mark partial close in DB
                        from db import _conn
                        c = _conn()
                        c.execute("UPDATE open_positions SET partial_closed=1, qty=? WHERE symbol=?",
                                  (round(qty * 0.50, 3), sym))
                        c.commit(); c.close()
                        partial = True
                except Exception as ex:
                    logger.debug(f"Partial close {sym}: {ex}")

            # ── SL LADDER ─────────────────────────────────────────────────
            if side == "Buy":
                if   unr_pct >= 20.0: min_sl = round(cur - atr_v * 0.7, prec)
                elif unr_pct >= 10.0: min_sl = round(cur - atr_v * 1.5, prec)
                else:                 min_sl = round(entry * 1.005, prec)   # BE+0.5%
            else:
                if   unr_pct >= 20.0: min_sl = round(cur + atr_v * 0.7, prec)
                elif unr_pct >= 10.0: min_sl = round(cur + atr_v * 1.5, prec)
                else:                 min_sl = round(entry * 0.995, prec)

            sl_improved = (
                (side == "Buy"  and (old_sl == 0 or min_sl > old_sl + atr_v * 0.05)) or
                (side == "Sell" and (old_sl == 0 or min_sl < old_sl - atr_v * 0.05))
            )

            # ── TP LADDER ─────────────────────────────────────────────────
            new_tp = 0.0; tp_improved = False
            tp_mult = 10.0 if unr_pct >= 20.0 else 7.0 if unr_pct >= 10.0 else 5.0
            new_tp  = round(cur + atr_v * tp_mult, prec) if side == "Buy" else round(cur - atr_v * tp_mult, prec)
            tp_improved = (
                (side == "Buy"  and (old_tp == 0 or new_tp > old_tp)) or
                (side == "Sell" and (old_tp == 0 or new_tp < old_tp))
            )

            if sl_improved or tp_improved:
                try:
                    kwargs = {}
                    if sl_improved: kwargs["sl"] = str(min_sl)
                    if tp_improved: kwargs["tp"] = str(new_tp)
                    if self.client:
                        await self.client.set_tpsl(sym, **kwargs)
                    logger.info(f"PROTECT | {sym} +{unr_pct:.1f}% | sl={min_sl:.4f}")
                except Exception:
                    pass

    # ── Diversification Gate ──────────────────────────────────────────────────

    def _can_open(self, symbol: str, signal: str) -> Tuple[bool, str]:
        db_open   = get_open_positions()
        open_syms = {p["symbol"] for p in db_open}
        if symbol in open_syms:
            return False, "Already open"
        if symbol in self.losing_symbols:
            return False, "Cooling off — last trade lost"
        if len(open_syms) >= self.state.compute_max_concurrent():
            return False, "Max concurrent reached"
        return True, "OK"

    # ── Trade Execution ───────────────────────────────────────────────────────

    async def execute(self, symbol: str, analysis: Dict, meta: Dict) -> bool:
        if self.client is None:
            return False
        s      = self.state
        signal = analysis["signal"]
        conf   = analysis["confidence"]
        comp   = analysis["composite"]
        mode   = s.mode
        adj    = self.regime.adjustments()

        raw5 = await self.client.klines(symbol, "5", 50)
        k5   = parse_klines(raw5)
        if k5 is None or len(k5) == 0:
            return False

        price   = float(k5[-1, 4])
        atr_val = atr(k5, 14)

        ss      = get_symbol_stats(symbol)
        sl_mult = max(1.8, min(ss.get("sl_mult", 1.8) + adj["sl_bonus"], 2.8))  # floor 1.8×
        tp_mult = max(2.5, min(ss.get("tp_mult", 3.0) + adj["tp_bonus"], 6.0))
        leverage = max(8, min(s.compute_leverage(meta.get("range_pct", 5.0)) + adj["lev_bonus"], 25))

        # Session-weighted sizing
        sess_label, sess_mult = _session_multiplier()
        risk_pct     = s.compute_risk_pct() * sess_mult
        risk_usd     = self.balance * risk_pct
        min_notional = float(gp("min_notional_usdt", "5.5") or "5.5")
        qty          = (risk_usd * leverage) / price

        if qty * price < min_notional:
            qty = (min_notional * 1.05) / price
        max_n = self.balance * 0.40 * leverage
        if qty * price > max_n:
            qty = max_n / price

        if   price >= 1000: qty = round(qty, 3)
        elif price >= 1:    qty = round(qty, 2)
        else:               qty = max(round(qty, 0), 1)

        if self.balance < (min_notional / leverage) * 1.5:
            logger.warning(f"Balance too low for {symbol}"); return False

        side = "Buy" if signal == "LONG" else "Sell"
        k5_cached = self._klines_cache.get(symbol)
        sl, tp    = sl_tp(side, price, atr_val, k5m=k5_cached, sl_mult=sl_mult, tp_mult=tp_mult)

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception:
            pass

        resp = await self.client.place_order(symbol, side, qty, sl=sl, tp=tp)
        if resp.get("retCode", -1) != 0:
            err = resp.get("retMsg", "?")
            logger.error(f"Order fail {symbol}: {err}")
            self.errors.append(f"{datetime.now().isoformat()} | {symbol} | {err}")
            return False

        order_id = resp.get("result", {}).get("orderId", "")
        epoch    = s.epoch_num
        open_trade(
            epoch=epoch, symbol=symbol, side=side, signal=signal,
            confidence=int(conf), composite=comp,
            entry_price=price, qty=qty, leverage=leverage,
            sl=sl, tp=tp, order_id=order_id,
            tag=f"EP{epoch}_L{leverage}_{self.regime.regime[:3]}_{mode[:3]}_{sess_label[:1]}",
            mode=mode, components=analysis.get("components", {}),
            vol_score=meta.get("vol_score", 0), range_pct=meta.get("range_pct", 0),
        )
        logger.info(
            f"OPEN [{mode}/{sess_label}/{self.regime.regime}] {symbol} {signal} "
            f"qty={qty}@${price:.4f} SL={sl:.4f} TP={tp:.4f} L={leverage}x"
        )
        return True

    # ── Position Monitor ──────────────────────────────────────────────────────

    async def monitor(self):
        await self.refresh_live_positions()
        db_open = get_open_positions()
        initial = float(gp("epoch_start_bal", "0.0") or "0.0") or self.balance or 10.0

        self.losing_symbols = set()
        for pos in db_open:
            if (pos.get("unrealised_pnl") or 0) < -0.05:
                self.losing_symbols.add(pos["symbol"])

        for pos in db_open:
            sym = pos["symbol"]; tid = pos["trade_id"]
            if sym in self.live_positions:
                live = self.live_positions[sym]
                try:
                    unr  = float(live.get("unrealisedPnl", "0") or "0")
                    cprc = float(live.get("markPrice",     "0") or "0")
                    update_pos_price(sym, cprc, unr)
                except Exception:
                    pass
            else:
                pnl = 0.0; exit_p = 0.0
                try:
                    if self.client:
                        pr     = await self.client.closed_pnl(sym, 10)
                        pls    = pr.get("result", {}).get("list", [])
                        if pls:
                            pnl    = float(pls[0].get("closedPnl",    "0") or "0")
                            exit_p = float(pls[0].get("avgExitPrice", "0") or "0")
                except Exception:
                    pass

                if exit_p == 0:
                    logger.debug(f"{sym}: no confirmed exit yet, skipping")
                    continue

                outcome = close_trade(tid, exit_p, pnl, initial)
                win     = outcome == "WIN"
                self.state.record_outcome(win, pnl)
                self.state.update_mode(self.balance)
                sp("current_mode", self.state.mode)

                if win:
                    self.cooldown_until[sym] = time.time() + 5
                else:
                    ss_cd    = get_symbol_stats(sym)
                    net_loss = max(0, ss_cd.get("losses", 0) - ss_cd.get("wins", 0))
                    cd_secs  = min(600, 120 + net_loss * 60)
                    self.cooldown_until[sym] = time.time() + cd_secs
                    self.losing_symbols.add(sym)
                logger.info(f"CLOSED | {sym} | {outcome} | ${pnl:.4f}")

        await self._pyramid_winners()

    async def _pyramid_winners(self):
        """Add 50% to profitable positions — only above +2.5% (raised from +0.8%)."""
        mode = self.state.mode
        if mode not in ("AGGRESSIVE", "TURBO"):
            return

        db_open = get_open_positions()
        for pos in db_open:
            sym   = pos["symbol"]
            entry = pos.get("entry_price") or 0
            side  = pos.get("side") or ""
            qty   = pos.get("qty") or 0
            tag   = pos.get("tag") or ""

            if "PYR" in tag or entry <= 0 or qty <= 0:
                continue

            live_p = self.live_positions.get(sym, {})
            cur    = float(live_p.get("markPrice", "0") or 0)
            if cur <= 0:
                continue

            # Pyramid trigger: +2.5% (raised from +0.8% — avoids noise adds)
            min_profit_pct = 0.025
            pct = (cur - entry) / entry if side == "Buy" else (entry - cur) / entry
            if pct < min_profit_pct:
                continue

            pyramid_qty  = round(qty * 0.50, 3)
            min_notional = float(gp("min_notional_usdt", "5.5") or "5.5")
            if pyramid_qty * cur < min_notional:
                continue

            try:
                if self.client:
                    resp = await self.client.place_order(sym, side, str(pyramid_qty))
                    if resp.get("retCode", 0) == 0:
                        logger.info(f"PYRAMID | {sym} +{pyramid_qty}@${cur:.4f} ({pct*100:.1f}%)")
                        open_trade(
                            epoch=pos.get("epoch", 1), symbol=sym, side=side,
                            signal=pos["signal"], confidence=pos.get("confidence", 50),
                            composite=0.0, entry_price=cur, qty=pyramid_qty,
                            leverage=pos.get("leverage", 10),
                            sl=pos.get("sl_price", 0), tp=pos.get("tp_price", 0),
                            order_id=resp.get("result", {}).get("orderId", ""),
                            tag=f"PYR_{tag}", mode=mode, components={},
                        )
            except Exception as e:
                logger.debug(f"Pyramid {sym}: {e}")

    # ── Scan & Trade ──────────────────────────────────────────────────────────

    async def scan_and_trade(self):
        s = self.state
        if s.check_circuit_breakers(self.balance):
            return

        # ── Dead-zone gate ────────────────────────────────────────────────
        if not _trading_window_ok():
            h = datetime.now(timezone.utc).hour
            logger.info(f"Dead zone: {h:02d}:00 UTC — no new entries (01:00–18:59 blocked)")
            sp("trading_window", "DEAD_ZONE")
            # Still monitor and protect open positions
            return

        sp("trading_window", "ACTIVE")
        await self.refresh_balance()

        # Auto-blacklist: symbols with 0% WR over ≥5 trades
        auto_blacklist: Set[str] = set()
        try:
            for _ss in all_symbol_stats():
                n = _ss.get("wins", 0) + _ss.get("losses", 0)
                if n >= 5 and _ss.get("wins", 0) == 0:
                    auto_blacklist.add(_ss["symbol"])
        except Exception:
            pass

        # Fear & Greed
        fg_val = 50; fg_label = "Neutral"; fg_conf_adj = 0
        try:
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                async with sess.get("https://api.alternative.me/fng/?limit=1",
                                    timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        fng     = await r.json()
                        fg      = fng.get("data", [{}])[0]
                        fg_val  = int(fg.get("value", 50))
                        fg_label= fg.get("value_classification", "Neutral")
                        if   fg_val <= 20: fg_conf_adj = -3
                        elif fg_val <= 35: fg_conf_adj = -1
                        elif fg_val >= 80: fg_conf_adj = +3
                        elif fg_val >= 65: fg_conf_adj = +1
        except Exception:
            pass

        mode       = s.mode
        n_syms     = int(gp("vol_scan_n", "12") or "12")
        adj        = self.regime.adjustments()
        conf_floor = max(35.0, s.compute_confidence_floor() + adj["conf_bonus"] + fg_conf_adj)

        top_syms = await self.scanner.scan(n=n_syms) if self.scanner else []

        # Refresh BTC 4H for macro gate
        await self._refresh_btc_4h()

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

            # Pass BTC 4H only for non-BTC symbols
            k4h_btc = self._btc_4h_cache if not sym.startswith("BTC") else None

            an = self.signals.analyze(
                sym=sym, k3m=data["k3m"], k5m=data["k5m"],
                k15m=data["k15m"], k1h=data["k1h"], k4h=data.get("k4h"),
                k4h_btc=k4h_btc,
                funding=data["funding"], ls=data["ls"],
                oi_pct=data["oi_pct"], ob_imb=data["ob_imb"],
                liqs=data["liqs"], learned_bias=bias, mode=mode,
            )

            result = {
                "symbol":     sym,
                "signal":     an["signal"],
                "confidence": an["confidence"],
                "composite":  an["composite"],
                "leverage":   s.compute_leverage(meta.get("range_pct", 5.0)),
                "vol_score":  meta.get("vol_score", 0),
                "range_pct":  meta.get("range_pct", 0),
                "chg_24h":    meta.get("chg_24h", 0),
                "regime":     self.regime.regime,
                "mode":       mode,
                "ts":         int(time.time()),
                "action":     "HOLD",
                "fear_greed": fg_val,
                "fg_label":   fg_label,
            }

            # LTF veto check
            comps = an.get("components", {})
            if comps.get("ltf_long_veto") and an["signal"] == "LONG":
                result["action"] = "LTF_VETO"
                results.append(result)
                continue
            if comps.get("ltf_short_veto") and an["signal"] == "SHORT":
                result["action"] = "LTF_VETO"
                results.append(result)
                continue
            if not an.get("vol_gate", True):
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
        sp("last_scan_ts", str(self.last_scan_ts))

    # ── Hourly Update ─────────────────────────────────────────────────────────

    async def hourly_strategy_update(self):
        await self.refresh_balance()
        s = self.state

        if self._klines_cache:
            self.regime.update(self._klines_cache)

        h_trades = get_trades(hours=1)
        h_closed = [t for t in h_trades if t["outcome"] not in ("OPEN", None)]
        h_wins   = sum(1 for t in h_closed if t["outcome"] == "WIN")
        h_pnl    = sum(t["pnl_usdt"] or 0 for t in h_closed)
        stats    = all_time_stats()

        snap_hour(s.epoch_num, self.balance, h_pnl, len(h_closed), h_wins,
                  stats["open_positions"], s.mode)
        log_capital(
            epoch=s.epoch_num, day_in_epoch=s.day_in_epoch,
            balance=self.balance,
            target_now=s.epoch_target * 0.2 * s.day_in_epoch,
            target_eod=s.epoch_target,
            ahead_pct=((self.balance - s.epoch_start_bal) / (s.epoch_target - s.epoch_start_bal + 1e-9)) * 100,
            mode=s.mode, open_pos=stats["open_positions"],
        )
        sp("market_regime",    self.regime.regime)
        sp("regime_label",     self.regime.adjustments()["label"])
        sp("hourly_win_rate",  str(round(h_wins / len(h_closed) * 100, 1) if h_closed else 0))
        sp("hourly_pnl",       str(round(h_pnl, 4)))
        sp("last_strategy_ts", str(int(time.time())))
        s.update_mode(self.balance)
        sp("current_mode", s.mode)
        logger.info(f"Hourly | regime={self.regime.regime} | pnl=${h_pnl:.4f} | mode={s.mode}")

    # ── Epoch Boundary ────────────────────────────────────────────────────────

    async def check_epoch_boundary(self):
        await self.refresh_balance()
        s = self.state
        if s.epoch_target > 0 and self.balance >= s.epoch_target:
            old = s.epoch_num
            close_epoch_record(old, self.balance)
            s.epoch_num       += 1
            s.epoch_start_bal  = self.balance
            s.epoch_target     = self.balance * 2.0
            s.epoch_start_ts   = int(time.time())
            s.day_in_epoch     = 1
            sp("current_epoch",   str(s.epoch_num))
            sp("epoch_start_ts",  str(s.epoch_start_ts))
            sp("epoch_start_bal", str(round(s.epoch_start_bal, 4)))
            open_epoch_record(s.epoch_num, s.epoch_start_bal)
            self.losing_symbols.clear()
            logger.info(f"EPOCH {old} COMPLETE → Epoch {s.epoch_num} | target=${s.epoch_target:.4f}")
            self._send_email(
                f"[ByBit] Epoch {old} DOUBLED!",
                f"Balance ${self.balance:.4f} | Next epoch target ${s.epoch_target:.4f}"
            )

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def main_loop(self):
        self.running = True
        self.status  = "RUNNING"

        live_bal = await self.refresh_balance()
        logger.info(f"Live balance on startup: ${live_bal:.4f}")

        if live_bal > 0 and self.capital_auto_detected:
            sp("initial_capital", str(round(live_bal, 4)))
            sp("epoch_start_bal", str(round(live_bal, 4)))
            sp("epoch_start_ts",  str(int(time.time())))
            sp("current_epoch",   "1")
            s = self.state
            s.epoch_start_bal = live_bal
            s.epoch_target    = live_bal * 2.0
            self.balance      = live_bal
            open_epoch_record(1, live_bal)

            # Auto-configure based on balance
            if live_bal < 20:
                sp("min_notional_usdt", "5.5");  sp("max_concurrent", "3");  sp("vol_scan_n", "8")
            elif live_bal < 100:
                sp("min_notional_usdt", "5.5");  sp("max_concurrent", "5");  sp("vol_scan_n", "10")
            elif live_bal < 500:
                sp("min_notional_usdt", "10.0"); sp("max_concurrent", "8");  sp("vol_scan_n", "12")
            else:
                sp("min_notional_usdt", "20.0"); sp("max_concurrent", "12"); sp("vol_scan_n", "14")
            self.capital_auto_detected = False

        elif live_bal > 0:
            self.balance = live_bal

        logger.info(f"Engine START | ${self.balance:.4f} | mode={self.state.mode} | target=${self.state.epoch_target:.4f}")
        scan_secs    = int(gp("scan_interval_s", "60") or "60")
        last_daily_h = -1

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
                cur_h = datetime.now(timezone.utc).hour
                if cur_h == 0 and last_daily_h != 0:
                    self.state.day_in_epoch = min(5, self.state.day_in_epoch + 1)
                    self.losing_symbols.clear()
                last_daily_h = cur_h
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop: {e}", exc_info=True)
                self.errors.append(f"{datetime.now().isoformat()} | {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(max(0, scan_secs - (time.time() - t0)))

        self.status  = "STOPPED"
        self.running = False
        if self.client and hasattr(self.client, "close"):
            await self.client.close()

    async def start(self):
        asyncio.create_task(self.main_loop())

    async def stop(self):
        self.running = False

    # ── Dashboard data helpers ────────────────────────────────────────────────

    def get_overview(self) -> Dict:
        s     = self.state
        stats = all_time_stats()
        bal   = self.balance
        init  = float(gp("initial_capital", "0") or "0")
        ep    = s.epoch_start_bal
        tgt   = s.epoch_target
        prog  = ((bal - ep) / (tgt - ep + 1e-9) * 100) if tgt > ep else 0.0
        ahead = prog - (s.day_in_epoch / 5 * 100)

        trades = get_trades(hours=168)
        closed = [t for t in trades if t["outcome"] not in ("OPEN", None)]
        wins   = [t for t in closed if t["outcome"] == "WIN"]
        streak = s.streak

        return {
            "engine_status":    self.status,
            "est_balance":      round(bal, 4),
            "initial":          round(init, 4),
            "testnet":          False,
            "market_regime":    self.regime.regime,
            "liquidity_session":gp("trading_window", "UNKNOWN"),
            "last_scan_ts":     self.last_scan_ts,
            "scan_results":     self.scan_results[-20:],
            "errors":           self.errors[-5:],
            "compound": {
                "mode":               s.mode,
                "epoch_num":          s.epoch_num,
                "day_in_epoch":       s.day_in_epoch,
                "epoch_start_bal":    round(s.epoch_start_bal, 4),
                "epoch_target":       round(s.epoch_target, 4),
                "epoch_progress_pct": round(prog, 2),
                "days_remaining":     max(0, 5 - s.day_in_epoch),
                "ahead_pct":          round(ahead, 2),
                "on_track":           ahead >= 0,
                "streak":             streak,
                "consecutive_losses": s.consecutive_losses,
                "rolling_win_rate":   round(s.win_rate(), 3),
                "sizing": {
                    "conf_floor":     s.compute_confidence_floor(),
                    "max_concurrent": s.compute_max_concurrent(),
                },
            },
            "stats": {**stats},
            "epoch_history": get_all_epochs(),
            "projection":   self._build_projection(),
            "whale_summary":{},
        }

    def _build_projection(self) -> List[Dict]:
        s    = self.state
        bal  = s.epoch_start_bal or self.balance or 10.0
        rows = []
        for ep in range(1, 11):
            rows.append({
                "epoch":       ep,
                "days_elapsed":(ep - 1) * 5,
                "start":       round(bal, 4),
                "target":      round(bal * 2.0, 4),
            })
            bal *= 2.0
        return rows

    # ── Email ─────────────────────────────────────────────────────────────────

    def _send_email(self, subject: str, body: str):
        if not SMTP_USER or not SMTP_PASS:
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SMTP_USER
            msg["To"]      = REPORT_EMAIL
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as sv:
                sv.starttls(); sv.login(SMTP_USER, SMTP_PASS); sv.send_message(msg)
        except Exception as e:
            logger.error(f"email: {e}")


# Global singleton
engine = CompoundTradingEngine()
