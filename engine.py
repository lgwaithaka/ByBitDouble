"""
engine.py — Perpetual Compounding Trading Engine
Doubles every 5-day epoch, indefinitely. Adapts aggression to epoch progress.
"""
import asyncio, json, os, time, logging, smtplib, threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set
import numpy as np

from bybit_client   import BybitClient
from scanner        import VScanner, parse_klines, atr, p_funding, p_ls, p_oi, p_ob, p_liq, price_precision
from signals        import SignalEngine, sl_tp
from compound_engine import CompoundEngine, DAILY_REQUIRED_PCT, EPOCH_DAYS
from db import (
    init_db, gp, sp, open_trade, close_trade, update_pos_price,
    get_open_positions, get_trades, all_time_stats, log_capital,
    get_symbol_stats, snap_hour, close_epoch_record, open_epoch_record,
    get_all_epochs, get_hourly_snaps, all_time_stats
)

logger = logging.getLogger(__name__)
REPORT_EMAIL = os.getenv("REPORT_EMAIL", "lgwaithaka@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")


class CompoundTradingEngine:
    def __init__(self):
        self.client:  Optional[BybitClient]     = None
        self.scanner: Optional[VScanner]        = None
        self.signals  = SignalEngine()
        self.compound = CompoundEngine()

        self.running       = False
        self.status        = "IDLE"
        self.balance       = 0.0
        self.live_positions: Dict[str, Dict]    = {}
        self.cooldown_until: Dict[str, float]   = {}   # symbol → ts

        self.scan_results: List[Dict]           = []
        self.last_scan_ts  = 0
        self.last_hour_ts  = 0
        self.last_daily_ts = 0
        self.errors: List[str]                  = []

    def init(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client  = BybitClient(api_key, api_secret, testnet)
        self.scanner = VScanner(self.client)
        init_db()

        # Restore epoch state from DB
        epoch_num  = int(gp("current_epoch", "1"))
        epoch_ts   = int(gp("epoch_start_ts", str(int(time.time()))))
        epoch_bal  = float(gp("epoch_start_bal", "100.0"))
        initial    = float(gp("initial_capital", "100.0"))

        self.compound.initialise(
            start_balance   = initial,
            epoch_num       = epoch_num,
            epoch_start_ts  = epoch_ts,
            epoch_start_bal = epoch_bal,
        )
        logger.info(f"Engine init | Epoch {epoch_num} | start=${epoch_bal:.2f}")

    # ── Balance ──────────────────────────────────────────────────────────────

    async def refresh_balance(self) -> float:
        try:
            resp = await self.client.wallet()
            for acc in resp.get("result", {}).get("list", []):
                b = acc.get("totalWalletBalance")
                if b:
                    self.balance = float(b); return self.balance
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

    # ── Market Data ──────────────────────────────────────────────────────────

    async def fetch_data(self, symbol: str) -> Optional[Dict]:
        try:
            results = await asyncio.gather(
                self.client.klines(symbol, "3",  80),
                self.client.klines(symbol, "5",  130),
                self.client.klines(symbol, "15", 60),
                self.client.klines(symbol, "60", 40),
                self.client.klines(symbol, "240", 60),
                self.client.funding(symbol),
                self.client.ls_ratio(symbol),
                self.client.open_interest(symbol, "1h"),
                self.client.orderbook(symbol, 50),
                self.client.liquidations(symbol),
                return_exceptions=True
            )
            k3, k5, k15, k1h, k4h, fr, lsr, oi, ob, liq = results
            return {
                "k3m":  parse_klines(k3)   if not isinstance(k3,  Exception) else None,
                "k5m":  parse_klines(k5)   if not isinstance(k5,  Exception) else None,
                "k15m": parse_klines(k15)  if not isinstance(k15, Exception) else None,
                "k1h":  parse_klines(k1h)  if not isinstance(k1h, Exception) else None,
                "k4h":  parse_klines(k4h)  if not isinstance(k4h, Exception) else None,
                "funding":  p_funding(fr)  if not isinstance(fr,  Exception) else 0.0,
                "ls":       p_ls(lsr)      if not isinstance(lsr, Exception) else 1.0,
                "oi_pct":   p_oi(oi)       if not isinstance(oi,  Exception) else 0.0,
                "ob_imb":   p_ob(ob)       if not isinstance(ob,  Exception) else 1.0,
                "liqs":     p_liq(liq)     if not isinstance(liq, Exception) else {},
            }
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}"); return None

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(self, symbol: str, analysis: Dict, meta: Dict) -> bool:
        ce     = self.compound
        signal = analysis["signal"]
        conf   = analysis["confidence"]
        comp   = analysis["composite"]
        mode   = ce.state.mode

        # Fetch fresh price + ATR
        raw5  = await self.client.klines(symbol, "5", 50)
        k5    = parse_klines(raw5)
        if k5 is None or len(k5) == 0: return False

        price    = float(k5[-1, 4])
        atr_val  = atr(k5, 14)

        sym_stats= get_symbol_stats(symbol)
        sl_mult, tp_mult = ce.compute_sl_tp_mults(
            sym_stats.get("sl_mult", 1.2),
            sym_stats.get("tp_mult", 2.4)
        )
        leverage = ce.compute_leverage(meta.get("range_pct", 5.0))

        # Position size — micro-account safe sizing
        risk_pct = ce.compute_risk_pct()
        risk_usd = self.balance * risk_pct
        sl_dist  = atr_val * sl_mult if atr_val > 0 else price * 0.009
        qty      = (risk_usd * leverage) / price
        qty      = max(qty, 0.001)

        # Bybit minimum notional check ($5.5 USDT minimum order value)
        min_notional = float(gp("min_notional_usdt", "5.5"))
        if qty * price < min_notional:
            # Bump qty up to meet minimum — use leverage to keep margin small
            qty = (min_notional * 1.05) / price   # 5% buffer over minimum

        # Cap single-trade notional at 40% of balance * leverage
        # For $10: max notional = $10 * 0.40 * 15x = $60 (margin used = $4)
        max_notional = self.balance * 0.40 * leverage
        if qty * price > max_notional:
            qty = max_notional / price

        # Round to correct precision for the asset price
        if price >= 1000:
            qty = round(qty, 3)
        elif price >= 1:
            qty = round(qty, 2)
        else:
            qty = round(qty, 0)  # e.g. SHIB, very small price = whole units
            qty = max(qty, 1)

        # Final sanity: if balance too low to meet minimum, skip trade
        required_margin = (min_notional) / leverage
        if self.balance < required_margin * 1.5:
            logger.warning(f"Balance ${self.balance:.2f} too low for {symbol} minimum order. Skipping.")
            return False

        side = "Buy" if signal == "LONG" else "Sell"
        sl, tp = sl_tp(side, price, atr_val, sl_mult=sl_mult, tp_mult=tp_mult)

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception: pass

        resp = await self.client.place_order(symbol, side, qty, sl=sl, tp=tp)
        if resp.get("retCode", -1) != 0:
            err = resp.get("retMsg", "?")
            logger.error(f"Order fail {symbol}: {err}")
            self.errors.append(f"{datetime.now().isoformat()} | {symbol} | {err}")
            return False

        order_id = resp.get("result", {}).get("orderId", "")
        epoch    = ce.state.epoch_num

        tid = open_trade(
            epoch=epoch, symbol=symbol, side=side, signal=signal,
            confidence=conf, composite=comp,
            entry_price=price, qty=qty, leverage=leverage,
            sl=sl, tp=tp, order_id=order_id,
            tag=f"EP{epoch}_L{leverage}_C{conf}_{mode}",
            mode=mode, components=analysis.get("components", {}),
            vol_score=meta.get("vol_score", 0),
            range_pct=meta.get("range_pct", 0),
        )
        logger.info(
            f"OPEN [{mode}] {symbol} {signal} | qty={qty} @ ${price:.4f} "
            f"| SL={sl:.4f} TP={tp:.4f} | L={leverage}x | ep={epoch} | conf={conf}%"
        )
        return True

    # ── Position Monitor ─────────────────────────────────────────────────────

    async def monitor(self):
        await self.refresh_live_positions()
        db_open = get_open_positions()
        initial = float(gp("initial_capital", "100.0"))
        epoch   = self.compound.state.epoch_num

        for pos in db_open:
            sym = pos["symbol"]
            tid = pos["trade_id"]

            if sym in self.live_positions:
                live = self.live_positions[sym]
                try:
                    unr  = float(live.get("unrealisedPnl", "0") or "0")
                    cprc = float(live.get("markPrice",     "0") or "0")
                    update_pos_price(sym, cprc, unr)
                except Exception: pass
            else:
                # Closed by SL/TP — fetch result
                pnl = 0.0; exit_p = 0.0
                try:
                    pr   = await self.client.closed_pnl(sym, 10)
                    pls  = pr.get("result", {}).get("list", [])
                    if pls:
                        pnl    = float(pls[0].get("closedPnl",    "0") or "0")
                        exit_p = float(pls[0].get("avgExitPrice", "0") or "0")
                except Exception: pass

                outcome = close_trade(tid, exit_p, pnl, initial)
                win     = outcome == "WIN"

                # Feed compound engine
                self.compound.record_outcome(win, pnl, self.balance)

                # Cooldown on loss
                if not win:
                    cd = 120
                    self.cooldown_until[sym] = time.time() + cd

                logger.info(f"CLOSED | {sym} | {outcome} | PnL=${pnl:.4f}")

    # ── Scan & Trade ─────────────────────────────────────────────────────────

    async def scan_and_trade(self):
        ce = self.compound

        # Check circuit breakers
        if ce.check_circuit_breakers(
            self.balance,
            epoch_max_dd = float(gp("epoch_max_dd_pct", "30")) / 100,
            daily_max_dd = float(gp("daily_max_dd_pct", "20")) / 100,
        ):
            logger.warning(f"Circuit breaker active | mode={ce.state.mode}")
            return

        await self.refresh_balance()
        top_syms   = await self.scanner.scan(n=int(gp("vol_scan_n","14")))
        db_open    = {p["symbol"] for p in get_open_positions()}
        conf_floor = ce.compute_confidence_floor()
        max_conc   = ce.compute_max_concurrent()
        mode       = ce.state.mode

        results = []
        for meta in top_syms:
            sym = meta["symbol"]

            # Cooldown check
            if self.cooldown_until.get(sym, 0) > time.time(): continue

            data = await self.fetch_data(sym)
            if not data: continue

            sym_stats    = get_symbol_stats(sym)
            learned_bias = sym_stats.get("learned_bias", 0.0)

            analysis = self.signals.analyze(
                sym      = sym,
                k3m      = data["k3m"],
                k5m      = data["k5m"],
                k15m     = data["k15m"],
                k1h      = data["k1h"],
                k4h      = data["k4h"],
                funding  = data["funding"],
                ls       = data["ls"],
                oi_pct   = data["oi_pct"],
                ob_imb   = data["ob_imb"],
                liqs     = data["liqs"],
                learned_bias = learned_bias,
                mode     = mode,
            )

            signal = analysis["signal"]; conf = analysis["confidence"]
            result = {
                "symbol": sym, "signal": signal, "confidence": conf,
                "composite": analysis["composite"],
                "leverage": ce.compute_leverage(meta.get("range_pct", 5)),
                "vol_score": meta.get("vol_score", 0),
                "range_pct": meta.get("range_pct", 0),
                "chg_24h":   meta.get("chg_24h", 0),
                "mode": mode, "ts": int(time.time()), "action": "HOLD"
            }

            already_open = sym in db_open
            can_open     = len(db_open) < max_conc and not already_open

            if signal != "HOLD" and conf >= conf_floor:
                if can_open:
                    ok = await self.execute(sym, analysis, meta)
                    if ok:
                        db_open.add(sym)
                        result["action"] = "OPENED"
                elif already_open:
                    result["action"] = "MONITORING"
                else:
                    result["action"] = "MAX_POSITIONS"
            results.append(result)
            await asyncio.sleep(0.25)

        self.scan_results = results
        self.last_scan_ts = int(time.time())

    # ── Epoch Boundary ───────────────────────────────────────────────────────

    async def check_epoch_boundary(self):
        """Check if epoch has elapsed and advance if so."""
        await self.refresh_balance()
        ce  = self.compound
        old = ce.state.epoch_num

        advanced = ce.advance_epoch(self.balance)
        if advanced:
            # Record completed epoch
            close_epoch_record(old, self.balance)
            new_epoch = ce.state.epoch_num

            # Persist new epoch state
            sp("current_epoch",   str(new_epoch))
            sp("epoch_start_ts",  str(ce.state.epoch_start_ts))
            sp("epoch_start_bal", str(round(ce.state.epoch_start_bal, 4)))

            # Create new epoch record
            open_epoch_record(new_epoch, ce.state.epoch_start_bal)

            await self.send_epoch_report(old, self.balance, ce.state.epoch_target)
            logger.info(f"New epoch {new_epoch} | stake=${ce.state.epoch_start_bal:.2f} | target=${ce.state.epoch_target:.2f}")

    # ── Hourly + Daily Tasks ─────────────────────────────────────────────────

    async def hourly_tasks(self):
        await self.refresh_balance()
        ce = self.compound
        s  = ce.state

        stats   = all_time_stats()
        h_trades= get_trades(hours=1)
        h_wins  = sum(1 for t in h_trades if t["outcome"] == "WIN")
        h_pnl   = sum(t["pnl_usdt"] or 0 for t in h_trades if t["outcome"] != "OPEN")

        snap_hour(s.epoch_num, self.balance, h_pnl,
                  len(h_trades), h_wins, stats["open_positions"], s.mode)
        log_capital(
            epoch        = s.epoch_num,
            day_in_epoch = s.day_in_epoch,
            balance      = self.balance,
            target_now   = ce.target_at_now(),
            target_eod   = ce.target_at_day_end(),
            ahead_pct    = ce.ahead_pct(self.balance),
            mode         = s.mode,
            open_pos     = stats["open_positions"],
        )
        await self.scanner.scan(force=True)
        logger.info(f"Hourly | ep={s.epoch_num} | bal=${self.balance:.2f} | mode={s.mode}")

    async def daily_tasks(self):
        await self.refresh_balance()
        self.compound.advance_day(self.balance)
        self.compound.reset_daily_cb()
        await self.send_daily_report()

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def main_loop(self):
        self.running = True
        self.status  = "RUNNING"
        await self.refresh_balance()
        logger.info(f"CompoundEngine START | balance=${self.balance:.2f} | mode={self.compound.state.mode}")

        scan_secs      = int(gp("scan_interval_s", "40"))
        last_daily_hour= -1

        while self.running:
            t0 = time.time()
            try:
                await self.monitor()
                await self.check_epoch_boundary()
                await self.scan_and_trade()

                now = time.time()
                if now - self.last_hour_ts >= 3600:
                    await self.hourly_tasks()
                    self.last_hour_ts = now

                cur_hour = datetime.now(timezone.utc).hour
                if cur_hour == 0 and last_daily_hour != 0:
                    await self.daily_tasks()
                last_daily_hour = cur_hour

            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                self.errors.append(f"{datetime.now().isoformat()} | {e}")
                await asyncio.sleep(5)

            await asyncio.sleep(max(0, scan_secs - (time.time() - t0)))

        self.status  = "STOPPED"; self.running = False
        if self.client: await self.client.close()

    async def stop(self):
        self.running = False

    # ── Reports ──────────────────────────────────────────────────────────────

    async def send_epoch_report(self, epoch_num: int, achieved: float, next_target: float):
        try:
            target = self.compound.state.epoch_start_bal  # before advance = old start
            pct    = (achieved - target) / target * 100 if target > 0 else 0
            body   = f"""
EPOCH {epoch_num} COMPLETE — BYBIT COMPOUND MCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch Start:   ${target:.2f} USDT
Achieved:      ${achieved:.2f} USDT
Target was:    ${target * 2:.2f} USDT
Gain:          {pct:+.1f}%
Status:        {'✓ DOUBLED' if achieved >= target * 2 else '✗ MISSED TARGET'}

Next Epoch {epoch_num + 1}:
  New Stake:   ${achieved:.2f} USDT
  New Target:  ${next_target:.2f} USDT
  Required:    {DAILY_REQUIRED_PCT:.2f}%/day for {EPOCH_DAYS} days

Projected Milestones (if targets met):
"""
            proj = self.compound.project_compounding(achieved, 8)
            for p in proj[:8]:
                body += f"  Epoch {p['epoch']}: ${p['target']:.2f}  (Day {p['days_elapsed']})\n"
            self._send_email(f"[Bybit Compound] Epoch {epoch_num} Done | ${achieved:.2f}", body)
        except Exception as e:
            logger.error(f"epoch report: {e}")

    async def send_daily_report(self):
        try:
            ce    = self.compound
            stats = all_time_stats()
            est   = self.compound.state.epoch_start_bal + stats["total_pnl"]
            ahead = ce.ahead_pct(est)
            body  = f"""
BYBIT COMPOUND MCP — DAILY REPORT
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EPOCH {ce.state.epoch_num} — Day {ce.state.day_in_epoch}/{EPOCH_DAYS}
  Mode:          {ce.state.mode}
  Epoch Stake:   ${ce.state.epoch_start_bal:.2f}
  Epoch Target:  ${ce.state.epoch_target:.2f}
  Est. Balance:  ${est:.4f}
  Target Now:    ${ce.target_at_now():.4f}
  Status:        {'✓ ON TRACK' if ahead >= -0.08 else '✗ BEHIND — ' + f'{ahead*100:.1f}%'}
  Days Left:     {ce.days_remaining():.1f}

PERFORMANCE
  Total Trades:  {stats['total_trades']}
  Win Rate:      {stats['win_rate']}%
  Total PnL:     ${stats['total_pnl']:.4f}
  Today PnL:     ${stats['today_pnl']:.4f}
  Open Pos:      {stats['open_positions']}
  Streak:        {ce.state.streak:+d}

SIZING
  Risk/Trade:    {ce.compute_risk_pct()*100:.2f}%
  Conf Floor:    {ce.compute_confidence_floor()}
  Max Positions: {ce.compute_max_concurrent()}
"""
            self._send_email(
                f"[Bybit Compound] Day {ce.state.day_in_epoch} Ep{ce.state.epoch_num} | ${est:.2f} | {ce.state.mode}",
                body
            )
        except Exception as e:
            logger.error(f"daily report: {e}")

    def _send_email(self, subject: str, body: str):
        if not SMTP_USER or not SMTP_PASS:
            logger.info(f"[EMAIL] {subject}\n{body[:200]}..."); return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SMTP_USER
            msg["To"]      = REPORT_EMAIL
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as sv:
                sv.starttls(); sv.login(SMTP_USER, SMTP_PASS); sv.send_message(msg)
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"email error: {e}")


engine = CompoundTradingEngine()
