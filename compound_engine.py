"""
compound_engine.py
═══════════════════════════════════════════════════════════════════════════════
The perpetual compounding engine. Every 5-day epoch doubles the stake.

EPOCH MATH
  Epoch N starts at balance B₀ = 100 × 2^(N-1)
  Target at epoch end: B₀ × 2.0
  Daily required: 2^(1/5) - 1 = 14.87%
  Hourly required: 2^(1/120) - 1 = 0.578%

POSITION SIZING — Three-Layer Model
  Layer 1 — Base Kelly fraction from rolling win rate (last 20 trades)
  Layer 2 — Epoch progress multiplier (ahead = push harder, behind = protect)
  Layer 3 — Streak multiplier (win streak boosts size, loss streak cuts)

  risk_pct = base_kelly × epoch_mult × streak_mult
  qty      = (balance × risk_pct × leverage) / price

AGGRESSION MODES
    NORMAL        — within 15% of target trajectory: standard params
  AGGRESSIVE    — ahead of target by >10%: push harder, more positions
  TURBO         — ahead by >25% with win streak: maximum aggression

CIRCUIT BREAKER
  Per-epoch max drawdown: 30% of epoch start balance
  Hard daily drawdown: 20% of day start balance
  Auto-resets at epoch boundary (every 5 days)
═══════════════════════════════════════════════════════════════════════════════
"""
import time, logging, math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

EPOCH_DAYS         = 5
EPOCH_SECS         = EPOCH_DAYS * 86400
DAILY_REQUIRED_PCT = (2.0 ** (1.0 / EPOCH_DAYS) - 1.0) * 100   # 14.87%
HOURLY_REQUIRED_PCT= (2.0 ** (1.0 / 120) - 1.0) * 100           # 0.578%
EPOCH_MULTIPLIER   = 2.0

# Aggression mode thresholds — CONSERVATIVE REMOVED
# System runs NORMAL minimum, scales up to TURBO
AGGR_BEHIND_SOFT  = -0.15   # >15% behind target → NORMAL (protect, but keep trading)
AGGR_AHEAD_PUSH   =  0.08   # 8%+ ahead → AGGRESSIVE
AGGR_AHEAD_TURBO  =  0.20   # 20%+ ahead + streak → TURBO

@dataclass
class EpochState:
    epoch_num:      int   = 1
    epoch_start_ts: int   = 0
    epoch_start_bal:float = 100.0
    epoch_target:   float = 200.0
    day_in_epoch:   int   = 1
    day_start_bal:  float = 100.0

    # Rolling performance
    streak:         int   = 0        # positive = win streak, negative = loss streak
    rolling_wins:   int   = 0
    rolling_total:  int   = 0
    rolling_win_rate: float = 0.5
    consecutive_losses: int = 0

    # Circuit breakers
    epoch_cb_active: bool = False
    daily_cb_active: bool = False
    epoch_cb_reason: str  = ""
    daily_cb_reason: str  = ""

    # Mode
    mode:            str  = "NORMAL"   # NORMAL | AGGRESSIVE | TURBO
    mode_since:      int  = 0


class CompoundEngine:
    """Manages perpetual doubling epochs and dynamic position sizing."""

    def __init__(self):
        self.state = EpochState()
        self._trade_buffer: List[Dict] = []   # last 30 trades for rolling stats

    # ── Epoch Management ────────────────────────────────────────────────────

    def initialise(self, start_balance: float, epoch_num: int = 1,
                   epoch_start_ts: int = 0, epoch_start_bal: float = None):
        s = self.state
        s.epoch_num       = epoch_num
        s.epoch_start_ts  = epoch_start_ts or int(time.time())
        s.epoch_start_bal = epoch_start_bal if epoch_start_bal is not None else start_balance
        s.epoch_target    = s.epoch_start_bal * EPOCH_MULTIPLIER
        s.day_start_bal   = start_balance
        s.day_in_epoch    = 1
        self._update_mode(start_balance)
        logger.info(
            f"Epoch {epoch_num} initialised | start=${s.epoch_start_bal:.2f} "
            f"| target=${s.epoch_target:.2f} | daily_req={DAILY_REQUIRED_PCT:.2f}%"
        )

    def advance_epoch(self, current_balance: float) -> bool:
        """
        Check if epoch has elapsed. If so, advance to next epoch.
        Returns True if epoch was advanced.
        """
        s   = self.state
        now = int(time.time())
        elapsed = now - s.epoch_start_ts

        if elapsed < EPOCH_SECS:
            return False

        # Epoch complete — lock in whatever balance we have
        achieved   = current_balance
        target_met = achieved >= s.epoch_target
        pct_gain   = (achieved - s.epoch_start_bal) / s.epoch_start_bal * 100

        logger.info(
            f"EPOCH {s.epoch_num} COMPLETE | "
            f"start=${s.epoch_start_bal:.2f} achieved=${achieved:.2f} "
            f"target=${s.epoch_target:.2f} | {pct_gain:+.1f}% | "
            f"{'✓ DOUBLED' if target_met else '✗ MISSED'}"
        )

        # Begin next epoch from current balance (achieved), not target
        # This is crucial: we compound what we actually have
        s.epoch_num       += 1
        s.epoch_start_ts   = now
        s.epoch_start_bal  = achieved
        s.epoch_target     = achieved * EPOCH_MULTIPLIER
        s.day_in_epoch     = 1
        s.day_start_bal    = achieved
        s.epoch_cb_active  = False
        s.daily_cb_active  = False
        s.epoch_cb_reason  = ""
        s.daily_cb_reason  = ""
        s.consecutive_losses = 0

        self._update_mode(achieved)
        logger.info(
            f"EPOCH {s.epoch_num} START | "
            f"stake=${s.epoch_start_bal:.2f} | target=${s.epoch_target:.2f}"
        )
        return True

    def advance_day(self, current_balance: float):
        """Call at UTC midnight."""
        s = self.state
        now   = int(time.time())
        day_n = max(1, int((now - s.epoch_start_ts) / 86400) + 1)
        s.day_in_epoch  = min(day_n, EPOCH_DAYS)
        s.day_start_bal = current_balance
        s.daily_cb_active  = False
        s.daily_cb_reason  = ""
        self._update_mode(current_balance)
        logger.info(f"Day {s.day_in_epoch} of Epoch {s.epoch_num} | balance=${current_balance:.2f}")

    # ── Target Trajectory ───────────────────────────────────────────────────

    def target_at_now(self) -> float:
        """What should our balance be RIGHT NOW to be on track?"""
        s        = self.state
        elapsed  = time.time() - s.epoch_start_ts
        frac     = min(elapsed / EPOCH_SECS, 1.0)
        return s.epoch_start_bal * (EPOCH_MULTIPLIER ** frac)

    def target_at_epoch_end(self) -> float:
        return self.state.epoch_target

    def target_at_day_end(self) -> float:
        s   = self.state
        day = min(s.day_in_epoch, EPOCH_DAYS)
        return s.epoch_start_bal * (EPOCH_MULTIPLIER ** (day / EPOCH_DAYS))

    def ahead_pct(self, balance: float) -> float:
        """How far ahead/behind target we are. Positive = ahead."""
        t = self.target_at_now()
        return (balance - t) / t if t > 0 else 0.0

    def epoch_progress_pct(self, balance: float) -> float:
        """% of epoch target achieved (0-100+)."""
        s = self.state
        denom = s.epoch_target - s.epoch_start_bal
        if denom <= 0: return 0.0
        return (balance - s.epoch_start_bal) / denom * 100

    def days_remaining(self) -> float:
        s = self.state
        elapsed = time.time() - s.epoch_start_ts
        return max(0, (EPOCH_SECS - elapsed) / 86400)

    # ── Aggression Mode ─────────────────────────────────────────────────────

    def _update_mode(self, balance: float):
        """
        Three modes only: NORMAL | AGGRESSIVE | TURBO
        CONSERVATIVE is removed — we always trade, just scale intensity.
        Circuit breakers stop trading entirely if needed; no half-measures.
        """
        s     = self.state
        ahead = self.ahead_pct(balance)

        # Circuit breaker active = stop trading entirely (not slow down)
        if s.epoch_cb_active or s.daily_cb_active:
            new_mode = "NORMAL"   # Will be blocked at CB level anyway

        # Behind target: stay NORMAL — keep trading, don't panic-size up
        elif ahead <= AGGR_BEHIND_SOFT or s.consecutive_losses >= 4:
            new_mode = "NORMAL"

        # TURBO: well ahead + win streak = maximum exploitation
        elif ahead >= AGGR_AHEAD_TURBO and s.streak >= 3:
            new_mode = "TURBO"

        # AGGRESSIVE: ahead of target or win streak
        elif ahead >= AGGR_AHEAD_PUSH or s.streak >= 2:
            new_mode = "AGGRESSIVE"

        else:
            new_mode = "NORMAL"

        if new_mode != s.mode:
            logger.info(f"Mode: {s.mode} → {new_mode} | ahead={ahead:+.1%} | streak={s.streak}")
            s.mode       = new_mode
            s.mode_since = int(time.time())

    # ── Position Sizing ─────────────────────────────────────────────────────

    def compute_risk_pct(self) -> float:
        """
        Returns % of balance to risk per trade.
        Combines: base Kelly × epoch_mult × streak_mult
        """
        s = self.state

        # Layer 1: base Kelly from rolling win rate
        wr = s.rolling_win_rate
        # Simplified Kelly: f = 2W - 1 (assuming 1:2 R:R)
        kelly_raw = max(0.02, min(2 * wr - 1, 0.40))
        # Use fractional Kelly (0.4 of full Kelly is safe-aggressive)
        base = kelly_raw * 0.40

        # Layer 2: mode multiplier — no conservative
        # NORMAL is baseline, AGGRESSIVE and TURBO push harder
        mode_mult = {
            "NORMAL":     1.00,
            "AGGRESSIVE": 1.45,   # 45% more than NORMAL
            "TURBO":      1.90,   # 90% more than NORMAL
        }.get(s.mode, 1.0)

        # Layer 3: streak multiplier
        # Win streaks scale UP aggressively — compound the momentum
        # Loss streaks only cut to 75% — still trading, not cowering
        if   s.streak >= 7:   streak_mult = 1.50   # HOT streak — maximum compound
        elif s.streak >= 5:   streak_mult = 1.35
        elif s.streak >= 3:   streak_mult = 1.20
        elif s.streak >= 1:   streak_mult = 1.08
        elif s.streak <= -4:  streak_mult = 0.75   # floor at 75%, still active
        elif s.streak <= -2:  streak_mult = 0.85
        else:                 streak_mult = 1.00

        risk = base * mode_mult * streak_mult

        # Mode caps — AGGRESSIVE by default, NORMAL is still active
        # No conservative: minimum is always NORMAL-level risk
        caps = {"NORMAL": 0.10, "AGGRESSIVE": 0.15, "TURBO": 0.20}
        risk = min(risk, caps.get(s.mode, 0.10))
        risk = max(risk, 0.06)   # floor 6% — always trading meaningfully

        return risk

    def compute_confidence_floor(self) -> int:
        """Confidence threshold — lower = more trades.
        NORMAL allows more entries, TURBO is most permissive."""
        floors = {"NORMAL": 48, "AGGRESSIVE": 40, "TURBO": 35}
        return floors.get(self.state.mode, 48)

    def compute_max_concurrent(self) -> int:
        """More concurrent = more opportunities captured simultaneously."""
        limits = {"NORMAL": 8, "AGGRESSIVE": 12, "TURBO": 16}
        return limits.get(self.state.mode, 8)

    def compute_leverage(self, range_pct: float) -> int:
        """
        Leverage scaled to volatility + mode.
        For micro-accounts ($10): higher base leverage needed so position
        notional meets Bybit $5 minimum while keeping margin small.
        """
        s = self.state
        # Leverage — scaled for returns, protected by 1.8x ATR stops
        # Higher base leverage = more notional from small balance = bigger wins
        if   range_pct > 20: base_lev = 8    # extreme volatile: 8x
        elif range_pct > 12: base_lev = 12   # high volatile: 12x
        elif range_pct > 7:  base_lev = 15   # moderate: 15x
        elif range_pct > 3:  base_lev = 18   # BTC/ETH normal: 18x
        else:                base_lev = 20   # quiet: 20x

        # Mode boosts — TURBO pushes max leverage
        boosts = {"NORMAL": 0, "AGGRESSIVE": 3, "TURBO": 5}
        lev = base_lev + boosts.get(s.mode, 0)
        return max(8, min(lev, 25))   # floor 8x, cap 25x

    def compute_sl_tp_mults(self, symbol_sl: float = 1.2, symbol_tp: float = 2.4) -> Tuple[float, float]:
        """
        Tighter SL in conservative (protect capital), wider TP in turbo (let run).
        Per-symbol learned values are inputs, mode adjusts them.
        """
        s = self.state
        mode_adj = {
            "NORMAL":        (1.0,  1.0),
            "NORMAL":        (1.0,  1.0),
            "AGGRESSIVE":    (1.0,  1.15),
            "TURBO":         (0.95, 1.30),
        }.get(s.mode, (1.0, 1.0))
        return round(symbol_sl * mode_adj[0], 2), round(symbol_tp * mode_adj[1], 2)

    # ── Trade Outcome Feedback ───────────────────────────────────────────────

    def record_outcome(self, win: bool, pnl: float, balance: float):
        """Call after every closed trade to update rolling stats and streak."""
        s = self.state

        # Streak
        if win:
            s.streak             = max(0, s.streak) + 1
            s.consecutive_losses = 0
        else:
            s.streak             = min(0, s.streak) - 1
            s.consecutive_losses += 1

        # Rolling stats (last 30 trades)
        self._trade_buffer.append({"win": win, "pnl": pnl})
        if len(self._trade_buffer) > 30:
            self._trade_buffer.pop(0)

        buf = self._trade_buffer
        s.rolling_total    = len(buf)
        s.rolling_wins     = sum(1 for t in buf if t["win"])
        s.rolling_win_rate = s.rolling_wins / s.rolling_total if s.rolling_total else 0.5

        # Update mode based on new balance
        self._update_mode(balance)

    # ── Circuit Breakers ─────────────────────────────────────────────────────

    def check_circuit_breakers(self, balance: float, epoch_max_dd: float = 0.30,
                                daily_max_dd: float = 0.20) -> bool:
        """Returns True if ANY circuit breaker is active."""
        s = self.state

        # Epoch-level drawdown
        if not s.epoch_cb_active and s.epoch_start_bal > 0:
            dd_from_epoch = (s.epoch_start_bal - balance) / s.epoch_start_bal
            if dd_from_epoch >= epoch_max_dd:
                s.epoch_cb_active = True
                s.epoch_cb_reason = f"Epoch DD {dd_from_epoch:.1%} ≥ {epoch_max_dd:.0%} limit"
                logger.warning(f"EPOCH CIRCUIT BREAKER: {s.epoch_cb_reason}")

        # Daily drawdown
        if not s.daily_cb_active and s.day_start_bal > 0:
            dd_from_day = (s.day_start_bal - balance) / s.day_start_bal
            if dd_from_day >= daily_max_dd:
                s.daily_cb_active = True
                s.daily_cb_reason = f"Daily DD {dd_from_day:.1%} ≥ {daily_max_dd:.0%} limit"
                logger.warning(f"DAILY CIRCUIT BREAKER: {s.daily_cb_reason}")

        return s.epoch_cb_active or s.daily_cb_active

    def reset_daily_cb(self):
        self.state.daily_cb_active = False
        self.state.daily_cb_reason = ""

    def reset_epoch_cb(self):
        self.state.epoch_cb_active = False
        self.state.epoch_cb_reason = ""

    # ── Projection ──────────────────────────────────────────────────────────

    def project_compounding(self, initial: float, n_epochs: int = 10) -> List[Dict]:
        """Project balance at end of each epoch assuming target is met."""
        results = []
        bal = initial
        for i in range(1, n_epochs + 1):
            target = bal * 2.0
            results.append({
                "epoch": i,
                "start": round(bal, 2),
                "target": round(target, 2),
                "days_elapsed": i * 5,
            })
            bal = target
        return results

    # ── Status Summary ──────────────────────────────────────────────────────

    def get_status(self, balance: float) -> Dict:
        s   = self.state
        now = int(time.time())
        elapsed   = now - s.epoch_start_ts
        remaining = max(0, EPOCH_SECS - elapsed)
        ahead     = self.ahead_pct(balance)
        progress  = self.epoch_progress_pct(balance)

        return {
            "epoch_num":         s.epoch_num,
            "day_in_epoch":      s.day_in_epoch,
            "mode":              s.mode,
            "epoch_start_bal":   round(s.epoch_start_bal, 4),
            "epoch_target":      round(s.epoch_target, 4),
            "target_now":        round(self.target_at_now(), 4),
            "target_eod":        round(self.target_at_day_end(), 4),
            "epoch_progress_pct":round(progress, 1),
            "ahead_pct":         round(ahead * 100, 2),
            "on_track":          ahead >= -0.08,
            "days_remaining":    round(self.days_remaining(), 2),
            "hours_remaining":   round(remaining / 3600, 1),
            "streak":            s.streak,
            "rolling_win_rate":  round(s.rolling_win_rate * 100, 1),
            "rolling_total":     s.rolling_total,
            "circuit_breakers": {
                "epoch_active":  s.epoch_cb_active,
                "daily_active":  s.daily_cb_active,
                "epoch_reason":  s.epoch_cb_reason,
                "daily_reason":  s.daily_cb_reason,
            },
            "sizing": {
                "risk_pct":       round(self.compute_risk_pct() * 100, 2),
                "conf_floor":     self.compute_confidence_floor(),
                "max_concurrent": self.compute_max_concurrent(),
            },
            "consecutive_losses": s.consecutive_losses,
        }
