"""
compound_engine.py — Perpetual Compounding Engine
═══════════════════════════════════════════════════
Manages epoch-based doubling targets, mode escalation, risk sizing,
and circuit breakers.

CRITICAL FIX: CONSERVATIVE mode REMOVED.
  - Week 1 data: CONSERVATIVE had 9.1% WR, caused 83% of all losses.
  - Minimum mode is now NORMAL with 1.8× ATR stops.
  - Mode ladder: NORMAL → AGGRESSIVE → TURBO
"""
import time, logging, math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

EPOCH_DAYS         = 5
DAILY_REQUIRED_PCT = 14.87  # (2.0^(1/5) - 1) × 100 ≈ 14.87%
EPOCH_MULTIPLIER   = 2.0

@dataclass
class CompoundState:
    initial_capital:  float = 10.0
    epoch_num:        int   = 1
    epoch_start_ts:   int   = 0
    epoch_start_bal:  float = 10.0
    epoch_target:     float = 20.0
    day_in_epoch:     int   = 1
    mode:             str   = "NORMAL"
    consecutive_wins: int   = 0
    consecutive_loss: int   = 0
    daily_pnl:        float = 0.0
    daily_trades:     int   = 0
    daily_wins:       int   = 0
    circuit_breaker:  bool  = False
    cb_reason:        str   = ""

class CompoundEngine:
    def __init__(self):
        self.state = CompoundState()

    def initialise(self, start_balance: float, epoch_num: int,
                   epoch_start_ts: int, epoch_start_bal: float):
        s = self.state
        s.initial_capital = start_balance
        s.epoch_num       = epoch_num
        s.epoch_start_ts  = epoch_start_ts
        s.epoch_start_bal = epoch_start_bal
        s.epoch_target    = epoch_start_bal * EPOCH_MULTIPLIER
        s.day_in_epoch    = max(1, min(EPOCH_DAYS,
                                       (int(time.time()) - epoch_start_ts) // 86400 + 1))
        s.mode = "NORMAL"
        logger.info(f"Compound init | Ep{epoch_num} ${epoch_start_bal:.4f}→${s.epoch_target:.4f}")

    # ── Mode Escalation (no CONSERVATIVE) ─────────────────────────────────

    def _auto_mode(self, balance: float):
        """
        Auto-escalate mode based on streak and epoch progress.
        CONSERVATIVE is REMOVED — minimum mode is NORMAL.
        """
        s = self.state
        ahead = self.ahead_pct(balance)

        if s.consecutive_wins >= 5 or ahead < -30:
            s.mode = "TURBO"
        elif s.consecutive_wins >= 2 or ahead < -15:
            s.mode = "AGGRESSIVE"
        else:
            s.mode = "NORMAL"

    def set_mode(self, mode: str):
        """Manual mode override. CONSERVATIVE is rejected."""
        mode = mode.upper()
        if mode == "CONSERVATIVE":
            logger.warning("CONSERVATIVE mode is disabled (9.1% WR). Using NORMAL.")
            mode = "NORMAL"
        if mode in ("NORMAL", "AGGRESSIVE", "TURBO"):
            self.state.mode = mode
            logger.info(f"Mode set → {mode}")
        else:
            logger.warning(f"Unknown mode: {mode}")

    # ── Record Outcomes ──────────────────────────────────────────────────

    def record_outcome(self, win: bool, pnl: float, balance: float):
        s = self.state
        s.daily_pnl    += pnl
        s.daily_trades += 1
        if win:
            s.daily_wins       += 1
            s.consecutive_wins += 1
            s.consecutive_loss  = 0
        else:
            s.consecutive_loss += 1
            s.consecutive_wins  = 0
        self._auto_mode(balance)

    # ── Risk Sizing ──────────────────────────────────────────────────────

    def compute_risk_pct(self) -> float:
        """
        Risk per trade as fraction of balance.
        NORMAL: 10-12%  |  AGGRESSIVE: 14-16%  |  TURBO: 18-20%
        Strategy doc: minimum 12% for 14.87% daily target.
        """
        s = self.state
        base = {"NORMAL": 0.10, "AGGRESSIVE": 0.14, "TURBO": 0.18}.get(s.mode, 0.10)
        # Streak bonus: +1% per consecutive win (max +4%)
        streak_bonus = min(s.consecutive_wins * 0.01, 0.04)
        # Loss penalty: -1% per consecutive loss (max -3%)
        loss_penalty = min(s.consecutive_loss * 0.01, 0.03)
        return max(0.06, min(0.22, base + streak_bonus - loss_penalty))

    def compute_leverage(self, range_pct: float) -> int:
        """
        Dynamic leverage: lower when volatile, higher when calm.
        Floor raised to 8x per strategy doc.
        """
        s = self.state
        base = {"NORMAL": 12, "AGGRESSIVE": 16, "TURBO": 20}.get(s.mode, 12)
        if range_pct > 8:
            base = max(8, base - 4)
        elif range_pct < 3:
            base = min(25, base + 3)
        return max(8, min(25, base))

    def compute_confidence_floor(self) -> int:
        """Minimum confidence to enter a trade."""
        return {"NORMAL": 65, "AGGRESSIVE": 58, "TURBO": 50}.get(self.state.mode, 65)

    def compute_max_concurrent(self) -> int:
        return {"NORMAL": 5, "AGGRESSIVE": 8, "TURBO": 12}.get(self.state.mode, 5)

    # ── Epoch Tracking ───────────────────────────────────────────────────

    def target_at_now(self) -> float:
        """Interpolated target for current moment within epoch."""
        s = self.state
        elapsed = time.time() - s.epoch_start_ts
        frac    = min(1.0, elapsed / (EPOCH_DAYS * 86400))
        return s.epoch_start_bal * (EPOCH_MULTIPLIER ** frac)

    def target_at_day_end(self) -> float:
        s = self.state
        frac = min(1.0, s.day_in_epoch / EPOCH_DAYS)
        return s.epoch_start_bal * (EPOCH_MULTIPLIER ** frac)

    def ahead_pct(self, balance: float) -> float:
        target = self.target_at_now()
        if target <= 0:
            return 0.0
        return (balance - target) / target * 100

    def advance_epoch(self, balance: float) -> bool:
        """Check if epoch target met and advance."""
        s = self.state
        if balance >= s.epoch_target:
            logger.info(f"EPOCH {s.epoch_num} COMPLETE! ${s.epoch_start_bal:.2f}→${balance:.2f}")
            s.epoch_num      += 1
            s.epoch_start_ts  = int(time.time())
            s.epoch_start_bal = balance
            s.epoch_target    = balance * EPOCH_MULTIPLIER
            s.day_in_epoch    = 1
            s.consecutive_wins = 0
            s.consecutive_loss = 0
            s.mode = "NORMAL"
            return True
        return False

    def advance_day(self, balance: float):
        s = self.state
        elapsed = (int(time.time()) - s.epoch_start_ts) // 86400 + 1
        s.day_in_epoch = min(EPOCH_DAYS, elapsed)
        self._auto_mode(balance)

    def reset_daily_cb(self):
        s = self.state
        s.daily_pnl     = 0.0
        s.daily_trades  = 0
        s.daily_wins    = 0
        s.circuit_breaker = False
        s.cb_reason       = ""

    # ── Circuit Breakers ─────────────────────────────────────────────────

    def check_circuit_breakers(self, balance: float,
                               epoch_max_dd: float = 0.35,
                               daily_max_dd: float = 0.25) -> bool:
        s = self.state

        # Epoch drawdown
        if s.epoch_start_bal > 0:
            epoch_dd = (s.epoch_start_bal - balance) / s.epoch_start_bal
            if epoch_dd >= epoch_max_dd:
                if s.mode == "NORMAL":
                    s.circuit_breaker = True
                    s.cb_reason = f"Epoch DD {epoch_dd*100:.1f}% >= {epoch_max_dd*100:.0f}%"
                    logger.warning(f"CB TRIGGERED: {s.cb_reason}")
                    return True
                else:
                    # AGGRESSIVE/TURBO bypass CB — just log warning
                    logger.warning(f"CB would trigger (epoch DD {epoch_dd*100:.1f}%) but bypassed in {s.mode}")

        # Daily drawdown
        if s.daily_pnl < 0 and s.epoch_start_bal > 0:
            daily_dd = abs(s.daily_pnl) / s.epoch_start_bal
            if daily_dd >= daily_max_dd:
                if s.mode == "NORMAL":
                    s.circuit_breaker = True
                    s.cb_reason = f"Daily DD {daily_dd*100:.1f}% >= {daily_max_dd*100:.0f}%"
                    logger.warning(f"CB TRIGGERED: {s.cb_reason}")
                    return True

        # 5 consecutive losses — pause for 10 minutes in any mode
        if s.consecutive_loss >= 5:
            s.circuit_breaker = True
            s.cb_reason = f"5 consecutive losses — 10min cooldown"
            logger.warning(f"CB TRIGGERED: {s.cb_reason}")
            return True

        return s.circuit_breaker

    # ── Projection ───────────────────────────────────────────────────────

    def project_growth(self, num_epochs: int = 10) -> list:
        """Project balance growth over future epochs."""
        bal = self.state.epoch_start_bal
        rows = []
        for i in range(1, num_epochs + 1):
            target = bal * EPOCH_MULTIPLIER
            rows.append({
                "epoch":   self.state.epoch_num + i - 1,
                "start":   round(bal, 4),
                "target":  round(target, 4),
                "days":    EPOCH_DAYS,
            })
            bal = target
        return rows
