import os, time, asyncio, aiohttp, logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self):
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_wh = os.getenv("DISCORD_WEBHOOK_URL")
        self.cooldown = int(os.getenv("ALERT_COOLDOWN_SEC", "300"))
        self.last_alerts: Dict[str, float] = {}
        self.req_count = 0
        self.window_start = time.time()
        self.start_time = time.time()

    async def _send(self, payload: Dict, url: str) -> bool:
        """Internal method to send HTTP POST requests."""
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=payload, timeout=10) as r:
                    return r.status in (200, 204)
        except Exception as e:
            logger.error(f"Alert send failed: {e}")
            return False

    async def send(self, msg: str, level: str = "INFO", force: bool = False, channels: List[str] = None) -> bool:
        """Send alert to configured channels (Telegram/Discord)."""
        if not force and not self._check_cooldown(level):
            return False
            
        if not channels: 
            channels = ["telegram", "discord"]
            
        emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "TRADE": "💹"
        }.get(level, "🔔")
        
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"{emoji} [{ts}] {msg}"
        
        tasks = []
        
        # Telegram
        if "telegram" in channels and self.tg_token and self.tg_chat:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            payload = {"chat_id": self.tg_chat, "text": formatted, "parse_mode": "HTML"}
            tasks.append(self._send(payload, url))
            
        # Discord
        if "discord" in channels and self.discord_wh:
            color_map = {
                "INFO": 3447003,
                "SUCCESS": 3066993,
                "WARNING": 15105570,
                "ERROR": 15158332,
                "TRADE": 10181046
            }
            embed = {
                "title": f"ByBitDouble {level}",
                "description": msg,
                "color": color_map.get(level, 9807270),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            tasks.append(self._send({"embeds": [embed]}, self.discord_wh))
        
        if not tasks:
            return False
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if any(r is True for r in results):
            self.last_alerts[level] = time.time()
            return True
        return False

    def _check_cooldown(self, level: str) -> bool:
        """Check if alert type is in cooldown period."""
        return (time.time() - self.last_alerts.get(level, 0)) >= self.cooldown

    async def heartbeat(self):
        """Send periodic heartbeat alert."""
        uptime = int(time.time() - self.start_time)
        await self.send(f"💓 Alive | Uptime: {uptime}s | Scanning 24/7", "INFO", force=True)

    async def trade_alert(self, sym: str, side: str, price: float, qty: float, conf: float):
        """Send trade execution alert."""
        emoji = "🟢" if side == "Buy" else "🔴"
        msg = f"{emoji} {sym} {side} @ ${price:,.4f} | Qty: {qty:.4f} | Conf: {conf:.1f}%"
        await self.send(msg, "TRADE")

    async def error_alert(self, ctx: str, err: str):
        """Send critical error alert."""
        msg = f"🚨 {ctx}\n{err[:200]}"
        await self.send(msg, "ERROR", force=True)

    def check_rate_limit(self) -> bool:
        """Check Bybit API rate limit buffer."""
        now = time.time()
        if now - self.window_start >= 5:
            self.req_count = 0
            self.window_start = now
        if self.req_count >= 600:  # Bybit limit: 600 per 5s
            return False
        self.req_count += 1
        return True

    async def safe_request(self, coro):
        """Wrapper for Bybit requests with rate limiting."""
        if not self.check_rate_limit():
            wait = 5 - (time.time() - self.window_start)
            if wait > 0: 
                await asyncio.sleep(wait)
            self.req_count = 0
            self.window_start = time.time()
        return await coro