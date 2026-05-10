"""
db.py — Full persistence layer
Tables: trades, open_positions, epochs, capital_log, symbol_stats,
hourly_snaps, params
"""
import sqlite3, json, time, os, logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# DB_PATH: use env var, fall back to ./data/compound.db if /data not writable
_raw_path = os.getenv("DB_PATH", "/data/compound.db")

def _resolve_db_path(raw: str) -> str:
    """Pick a writable path — prefers the configured path, falls back to ./data/"""
    parent = os.path.dirname(raw)
    try:
        os.makedirs(parent, exist_ok=True)
        # Test write access
        test = os.path.join(parent, ".write_test")
        with open(test, "w") as f: f.write("x")
        os.remove(test)
        return raw
    except Exception:
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "compound.db")
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        logger.warning(f"Cannot write to {raw}, using fallback: {fallback}")
        return fallback

DB_PATH = _resolve_db_path(_raw_path)
logger.info(f"DB_PATH resolved to: {DB_PATH}")

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c

def init_db():
    c = _conn()
    c.executescript("""
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_open      INTEGER NOT NULL,
    ts_close     INTEGER,
    epoch        INTEGER DEFAULT 1,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    signal       TEXT NOT NULL,
    confidence   INTEGER,
    composite    REAL,
    entry_price  REAL,
    exit_price   REAL,
    qty          REAL,
    leverage     INTEGER,
    notional     REAL,
    sl_price     REAL,
    tp_price     REAL,
    pnl_usdt     REAL,
    pnl_pct      REAL,
    outcome      TEXT DEFAULT 'OPEN',
    duration_s   INTEGER,
    order_id     TEXT,
    tag          TEXT,
    mode         TEXT,
    components   TEXT,
    vol_score    REAL,
    range_pct    REAL
);

CREATE TABLE IF NOT EXISTS open_positions (
    symbol         TEXT PRIMARY KEY,
    epoch          INTEGER DEFAULT 1,
    side           TEXT,
    signal         TEXT,
    trade_id       INTEGER,
    order_id       TEXT,
    entry_price    REAL,
    qty            REAL,
    leverage       INTEGER,
    notional       REAL,
    sl_price       REAL,
    tp_price       REAL,
    ts_open        INTEGER,
    confidence     INTEGER,
    unrealised_pnl REAL DEFAULT 0.0,
    current_price  REAL DEFAULT 0.0,
    partial_closed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS epochs (
    epoch_num      INTEGER PRIMARY KEY,
    start_ts       INTEGER,
    end_ts         INTEGER,
    start_bal      REAL,
    end_bal        REAL,
    target_bal     REAL,
    target_met     INTEGER DEFAULT 0,
    pnl_usdt       REAL DEFAULT 0,
    pct_gain       REAL DEFAULT 0,
    total_trades   INTEGER DEFAULT 0,
    wins           INTEGER DEFAULT 0,
    losses         INTEGER DEFAULT 0,
    win_rate       REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS capital_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER NOT NULL,
    epoch        INTEGER,
    day_in_epoch INTEGER,
    balance      REAL,
    target_now   REAL,
    target_eod   REAL,
    ahead_pct    REAL,
    mode         TEXT,
    open_pos     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS symbol_stats (
    symbol       TEXT PRIMARY KEY,
    wins         INTEGER DEFAULT 0,
    losses       INTEGER DEFAULT 0,
    total_pnl    REAL DEFAULT 0,
    learned_bias REAL DEFAULT 0.0,
    sl_mult      REAL DEFAULT 1.2,
    tp_mult      REAL DEFAULT 2.4,
    avg_dur_s    REAL DEFAULT 0,
    last_updated INTEGER
);

CREATE TABLE IF NOT EXISTS hourly_snaps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER,
    hour_label   TEXT, 
    epoch        INTEGER,
    balance      REAL,
    hourly_pnl   REAL,
    trades       INTEGER,
    wins         INTEGER,
    win_rate     REAL,
    open_pos     INTEGER,
    mode         TEXT
);

CREATE TABLE IF NOT EXISTS params (
    key          TEXT PRIMARY KEY,
    value        TEXT,
    updated_ts   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades(ts_open);
CREATE INDEX IF NOT EXISTS idx_trades_sym    ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_epoch  ON trades(epoch);
CREATE INDEX IF NOT EXISTS idx_trades_out    ON trades(outcome);
""")
    c.commit()

    # Migration: add partial_closed if upgrading from older schema
    try:
        c.execute("ALTER TABLE open_positions ADD COLUMN partial_closed INTEGER DEFAULT 0")
        c.commit()
    except Exception:
        pass  # Column already exists
    # initial_capital and epoch_start_bal are seeded as 0.0
    # and auto-updated on first startup from the real Bybit balance.
    defaults = {
        "initial_capital":      "0.0",      # auto-set from Bybit on first run
        "epoch_days":           "5",
        "epoch_multiplier":     "2.0",
        "epoch_max_dd_pct":     "35",
        "daily_max_dd_pct":     "25",
        "vol_scan_n":           "12",       # auto-adjusted based on balance size
        "scan_interval_s":      "30",
        "start_ts":             str(int(time.time())),
        "current_epoch":        "1",
        "epoch_start_ts":       str(int(time.time())),
        "epoch_start_bal":      "0.0",      # auto-set from Bybit on first run
        "min_notional_usdt":    "5.5",      # Bybit minimum order value
        "max_concurrent":       "8",        # auto-adjusted based on balance size
    }
    for k, v in defaults.items():
        c.execute("INSERT OR IGNORE INTO params (key,value,updated_ts) VALUES (?,?,?)",
                  (k, v, int(time.time())))
    c.commit()

    # Epoch 1 seeded with 0.0 — real balance filled in by engine on first startup
    c.execute("INSERT OR IGNORE INTO epochs (epoch_num,start_ts,start_bal,target_bal) VALUES (?,?,?,?)",
              (1, int(time.time()), 0.0, 0.0))
    c.commit(); c.close()
    logger.info(f"DB ready: {DB_PATH}")

def gp(key: str, default=None) -> Optional[str]:
    try:
        c = _conn()
        r = c.execute("SELECT value FROM params WHERE key=?", (key,)).fetchone()
        c.close(); return r["value"] if r else default
    except: return default

def sp(key: str, value: str):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO params (key,value,updated_ts) VALUES (?,?,?)",
              (key, str(value), int(time.time())))
    c.commit(); c.close()

def open_trade(
    epoch: int, symbol: str, side: str, signal: str,
    confidence: int, composite: float,
    entry_price: float, qty: float, leverage: int,
    sl: float, tp: float, order_id: str,
    tag: str, mode: str, components: Dict,
    vol_score: float = 0.0, range_pct: float = 0.0,
) -> int:
    notional = qty * entry_price
    c = _conn()
    cur = c.execute(
        """INSERT INTO trades
        (ts_open,epoch,symbol,side,signal,confidence,composite,
        entry_price,qty,leverage,notional,sl_price,tp_price,
        outcome,order_id,tag,mode,components,vol_score,range_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?,?,?,?,?,?)""",
        (int(time.time()), epoch, symbol, side, signal, confidence, composite,
        entry_price, qty, leverage, notional, sl, tp,
        order_id, tag, mode, json.dumps(components), vol_score, range_pct)
    )
    tid = cur.lastrowid
    c.execute(
        """INSERT OR REPLACE INTO open_positions
        (symbol,epoch,side,signal,trade_id,order_id,entry_price,qty,
        leverage,notional,sl_price,tp_price,ts_open,confidence,partial_closed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (symbol, epoch, side, signal, tid, order_id, entry_price, qty,
        leverage, notional, sl, tp, int(time.time()), confidence)
    )
    c.commit(); c.close()
    return tid

def close_trade(trade_id: int, exit_price: float, pnl: float, initial_cap: float):
    now  = int(time.time())
    out  = "WIN" if pnl > 0.005 else ("LOSS" if pnl < -0.005 else "BREAKEVEN")
    pct  = pnl / initial_cap * 100
    c    = _conn()
    row  = c.execute("SELECT ts_open, symbol FROM trades WHERE id=?", (trade_id,)).fetchone()
    dur  = now - row["ts_open"] if row else 0
    c.execute(
        """UPDATE trades SET ts_close=?,exit_price=?,pnl_usdt=?,pnl_pct=?,
        outcome=?,duration_s=? WHERE id=?""",
        (now, exit_price, pnl, pct, out, dur, trade_id)
    )
    if row:
        c.execute("DELETE FROM open_positions WHERE symbol=?", (row["symbol"],))
        _upd_sym(c, row["symbol"], out, pnl, dur)
    c.commit(); c.close()
    return out

# ── NEW FUNCTION: log_trade (for engine.py compatibility) ──────────────
def log_trade(data: Dict) -> int:
    """
    Log a completed trade directly (used by engine.py close_position).
    Expects dict with: symbol, side, entry_price, qty, ts_open, ts_close,
    outcome, pnl_usdt, confidence, and optional fields.
    Returns trade ID.
    """
    c = _conn()
    cur = c.execute(
        """INSERT INTO trades
        (ts_open, ts_close, symbol, side, signal, confidence, entry_price,
         exit_price, qty, leverage, notional, pnl_usdt, pnl_pct, outcome, mode)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data.get("ts_open", int(time.time())),
            data.get("ts_close", int(time.time())),
            data["symbol"],
            data["side"],
            data.get("signal", "MCP"),
            data.get("confidence", 0),
            data["entry_price"],
            data.get("exit_price", data["entry_price"] * (1 + data.get("pnl_usdt", 0)/data["entry_price"])),
            data["qty"],
            data.get("leverage", 10),
            data.get("notional", data["qty"] * data["entry_price"]),
            data.get("pnl_usdt", 0),
            data.get("pnl_pct", 0),
            data.get("outcome", "WIN" if data.get("pnl_usdt", 0) > 0 else "LOSS"),
            data.get("mode", "NORMAL")
        )
    )
    tid = cur.lastrowid
    
    # Remove from open_positions if exists
    c.execute("DELETE FROM open_positions WHERE symbol=?", (data["symbol"],))
    
    # Update symbol stats
    outcome = data.get("outcome", "WIN" if data.get("pnl_usdt", 0) > 0 else "LOSS")
    _upd_sym(c, data["symbol"], outcome, data.get("pnl_usdt", 0), 
             data.get("duration_s", data.get("ts_close", time.time()) - data.get("ts_open", time.time())))
    
    c.commit(); c.close()
    return tid
# ────────────────────────────────────────────────────────────────────────

def _upd_sym(c, symbol: str, outcome: str, pnl: float, dur: float):
    r = c.execute("SELECT * FROM symbol_stats WHERE symbol=?", (symbol,)).fetchone()
    wins   = (r["wins"]   if r else 0) + (1 if outcome == "WIN" else 0)
    losses = (r["losses"] if r else 0) + (1 if outcome == "LOSS" else 0)
    n      = wins + losses
    old_p  = r["total_pnl"] if r else 0
    old_b  = r["learned_bias"] if r else 0.0
    sl_m   = r["sl_mult"] if r else 1.2
    tp_m   = r["tp_mult"] if r else 2.4
    old_dur= r["avg_dur_s"] if r else 0
    
    # Exponential bias update
    direction = 1.0 if outcome == "WIN" else -1.0
    new_bias  = old_b * 0.85 + direction * 0.15 * min(abs(pnl) / 2.0, 0.55)
    new_bias  = max(-0.55, min(0.55, new_bias))

    # Adapt SL/TP by win rate
    wr = wins / n if n > 0 else 0.5
    if wr < 0.40:
        sl_m = min(sl_m * 1.05, 2.2)
        tp_m = max(tp_m * 0.97, 1.8)
    elif wr > 0.62:
        sl_m = max(sl_m * 0.97, 0.85)
        tp_m = min(tp_m * 1.04, 4.2)

    # Avg duration EMA
    avg_dur = old_dur * 0.8 + dur * 0.2

    if r:
        c.execute(
            """UPDATE symbol_stats SET wins=?,losses=?,total_pnl=?,
            learned_bias=?,sl_mult=?,tp_mult=?,avg_dur_s=?,last_updated=?
            WHERE symbol=?""",
            (wins, losses, old_p + pnl, new_bias, round(sl_m,2), round(tp_m,2),
             round(avg_dur), int(time.time()), symbol)
        )
    else:
        c.execute(
            """INSERT INTO symbol_stats
            (symbol,wins,losses,total_pnl,learned_bias,sl_mult,tp_mult,avg_dur_s,last_updated)
            VALUES (?,?,?,?,0.0,1.2,2.4,?,?)""",
            (symbol, wins, losses, old_p+pnl, round(avg_dur), int(time.time()))
        )

def update_pos_price(symbol: str, price: float, unr: float):
    c = _conn()
    c.execute("UPDATE open_positions SET current_price=?,unrealised_pnl=? WHERE symbol=?",
              (price, unr, symbol))
    c.commit(); c.close()

def get_open_positions() -> List[Dict]:
    c    = _conn()
    rows = c.execute("SELECT * FROM open_positions ORDER BY ts_open DESC").fetchall()
    c.close(); return [dict(r) for r in rows]

def get_trades(hours: int = 24, symbol: str = None, epoch: int = None) -> List[Dict]:
    cutoff = int(time.time()) - hours * 3600
    c = _conn()
    q = "SELECT * FROM trades WHERE ts_open >= ?"; p = [cutoff]
    if symbol: q += " AND symbol=?"; p.append(symbol)
    if epoch:  q += " AND epoch=?";  p.append(epoch)
    rows = c.execute(q + " ORDER BY ts_open DESC", p).fetchall()
    c.close(); return [dict(r) for r in rows]

def get_epoch_trades(epoch_num: int) -> List[Dict]:
    c    = _conn()
    rows = c.execute("SELECT * FROM trades WHERE epoch=? ORDER BY ts_open",
                     (epoch_num,)).fetchall()
    c.close(); return [dict(r) for r in rows]

def get_symbol_stats(symbol: str) -> Dict:
    c   = _conn()
    row = c.execute("SELECT * FROM symbol_stats WHERE symbol=?", (symbol,)).fetchone()
    c.close()
    if row:
        d = dict(row); n = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"]/n*100, 1) if n > 0 else 50.0
        return d
    return {"symbol":symbol, "wins":0, "losses":0, "total_pnl":0.0,
            "learned_bias":0.0, "sl_mult":1.2, "tp_mult":2.4, "win_rate":50.0}

def all_symbol_stats() -> List[Dict]:
    c    = _conn()
    rows = c.execute("SELECT * FROM symbol_stats ORDER BY total_pnl DESC").fetchall()
    c.close()
    res  = []
    for r in rows:
        d = dict(r); n = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"]/n*100,1) if n>0 else 0.0
        res.append(d)
    return res

def log_capital(epoch: int, day_in_epoch: int, balance: float,
                target_now: float, target_eod: float,
                ahead_pct: float, mode: str, open_pos: int):
    c = _conn()
    c.execute(
        """INSERT INTO capital_log
        (ts,epoch,day_in_epoch,balance,target_now,target_eod,ahead_pct,mode,open_pos)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (int(time.time()), epoch, day_in_epoch, balance,
        target_now, target_eod, ahead_pct, mode, open_pos)
    )
    c.commit(); c.close()

def get_capital_log(limit: int = 300) -> List[Dict]:
    c    = _conn()
    rows = c.execute(
        "SELECT * FROM capital_log ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close(); return [dict(r) for r in reversed(rows)]

def close_epoch_record(epoch_num: int, end_bal: float):
    c    = _conn()
    row  = c.execute("SELECT * FROM epochs WHERE epoch_num=?", (epoch_num,)).fetchone()
    if not row: c.close(); return
    trades = c.execute("SELECT * FROM trades WHERE epoch=? AND outcome!='OPEN'",
                       (epoch_num,)).fetchall()
    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    pnl    = sum(t["pnl_usdt"] or 0 for t in trades)
    pct    = (end_bal - row["start_bal"]) / row["start_bal"] * 100 if row["start_bal"] else 0
    met    = 1 if end_bal >= row["target_bal"] else 0
    c.execute(
        """UPDATE epochs SET end_ts=?,end_bal=?,target_met=?,pnl_usdt=?,
        pct_gain=?,total_trades=?,wins=?,losses=?,win_rate=? WHERE epoch_num=?""",
        (int(time.time()), end_bal, met, pnl, pct,
        len(trades), wins, len(trades)-wins,
        round(wins/len(trades)*100,1) if trades else 0, epoch_num)
    )
    c.commit(); c.close()

def open_epoch_record(epoch_num: int, start_bal: float):
    c = _conn()
    c.execute(
        "INSERT OR IGNORE INTO epochs (epoch_num,start_ts,start_bal,target_bal) VALUES (?,?,?,?)",
        (epoch_num, int(time.time()), start_bal, start_bal * 2.0)
    )
    c.commit(); c.close()

def get_all_epochs() -> List[Dict]:
    c    = _conn()
    rows = c.execute("SELECT * FROM epochs ORDER BY epoch_num").fetchall()
    c.close(); return [dict(r) for r in rows]

def snap_hour(epoch: int, balance: float, hourly_pnl: float,
              trades: int, wins: int, open_pos: int, mode: str):
    label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00")
    wr    = round(wins/trades*100,1) if trades > 0 else 0.0
    c     = _conn()
    c.execute(
        """INSERT INTO hourly_snaps
        (ts,hour_label,epoch,balance,hourly_pnl,trades,wins,win_rate,open_pos,mode)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (int(time.time()), label, epoch, balance, hourly_pnl, trades, wins, wr, open_pos, mode)
    )
    c.commit(); c.close()

def get_hourly_snaps(limit: int = 96) -> List[Dict]:
    c    = _conn()
    rows = c.execute(
        "SELECT * FROM hourly_snaps ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close(); return [dict(r) for r in reversed(rows)]

def all_time_stats() -> Dict:
    c = _conn()
    closed = c.execute("SELECT * FROM trades WHERE outcome != 'OPEN'").fetchall()
    open_p = c.execute("SELECT * FROM open_positions").fetchall()
    now    = int(time.time())
    today  = int(datetime.now(timezone.utc).replace(hour=0,minute=0,second=0).timestamp())
    today_trades = [t for t in closed if t["ts_open"] >= today]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]
    c.close()
    n = len(closed)
    return {
        "total_trades":   n,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins)/n*100,1) if n >0 else 0.0,
        "total_pnl":      round(sum(t["pnl_usdt"] or 0 for t in closed), 4),
        "today_pnl":      round(sum(t["pnl_usdt"] or 0 for t in today_trades), 4),
        "today_trades":   len(today_trades),
        "open_positions": len(open_p),
        "unrealised_pnl": round(sum(r["unrealised_pnl"] or 0 for r in open_p), 4),
        "avg_win":        round(sum(t["pnl_usdt"] for t in wins)/len(wins),4) if wins else 0,
        "avg_loss":       round(sum(t["pnl_usdt"] for t in losses)/len(losses),4) if losses else 0,
    }

def all_params() -> Dict:
    c    = _conn()
    rows = c.execute("SELECT key,value FROM params").fetchall()
    c.close(); return {r["key"]: r["value"] for r in rows}