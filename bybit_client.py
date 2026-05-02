"""
bybit_client.py — Bybit V5 REST client
Signing: timestamp + api_key + recv_window + payload (HMAC-SHA256)
"""
import hashlib, hmac, time, asyncio, aiohttp, json, os, logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)
RECV_WIN   = "5000"
BASE_LIVE  = "https://api.bybit.com"
BASE_TEST  = "https://api-testnet.bybit.com"


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base       = BASE_TEST if testnet else BASE_LIVE
        self._sess: Optional[aiohttp.ClientSession] = None

    async def _session(self):
        if not self._sess or self._sess.closed:
            self._sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12))
        return self._sess

    def _sign(self, ts: str, payload: str) -> str:
        raw = ts + self.api_key + RECV_WIN + payload
        return hmac.new(self.api_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _hdrs(self, ts: str, sign: str) -> Dict:
        return {"X-BAPI-API-KEY": self.api_key, "X-BAPI-SIGN": sign,
                "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": RECV_WIN,
                "Content-Type": "application/json"}

    async def _get(self, path, params=None, auth=True):
        sess = await self._session(); params = params or {}
        qs   = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        ts   = str(int(time.time() * 1000))
        hdrs = self._hdrs(ts, self._sign(ts, qs)) if auth else {}
        try:
            async with sess.get(f"{self.base}{path}", params=params, headers=hdrs) as r:
                data = await r.json(content_type=None)
                if data.get("retCode", 0) != 0:
                    logger.debug(f"GET {path} rc={data.get('retCode')} {data.get('retMsg')}")
                return data
        except Exception as e:
            logger.error(f"GET {path}: {e}"); return {"retCode": -1, "result": {}}

    async def _post(self, path, body=None, auth=True):
        sess   = await self._session(); body = body or {}
        bs     = json.dumps(body)
        ts     = str(int(time.time() * 1000))
        hdrs   = self._hdrs(ts, self._sign(ts, bs)) if auth else {"Content-Type":"application/json"}
        try:
            async with sess.post(f"{self.base}{path}", data=bs, headers=hdrs) as r:
                data = await r.json(content_type=None)
                if data.get("retCode", 0) != 0:
                    logger.warning(f"POST {path} rc={data.get('retCode')} {data.get('retMsg')}")
                return data
        except Exception as e:
            logger.error(f"POST {path}: {e}"); return {"retCode": -1, "result": {}}

    async def close(self):
        if self._sess and not self._sess.closed: await self._sess.close()

    # ── Market Data ──────────────────────────────────────────────────────────
    async def all_tickers(self):
        return await self._get("/v5/market/tickers", {"category":"linear"}, auth=False)
    async def klines(self, sym, interval, limit=200):
        return await self._get("/v5/market/kline",
            {"category":"linear","symbol":sym,"interval":interval,"limit":limit}, auth=False)
    async def orderbook(self, sym, limit=50):
        return await self._get("/v5/market/orderbook",
            {"category":"linear","symbol":sym,"limit":limit}, auth=False)
    async def funding(self, sym):
        return await self._get("/v5/market/funding/history",
            {"category":"linear","symbol":sym,"limit":1}, auth=False)
    async def open_interest(self, sym, period="1h"):
        return await self._get("/v5/market/open-interest",
            {"category":"linear","symbol":sym,"intervalTime":period,"limit":3}, auth=False)
    async def ls_ratio(self, sym):
        # fields: buyRatio / sellRatio (NOT longRatio/shortRatio)
        return await self._get("/v5/market/account-ratio",
            {"category":"linear","symbol":sym,"period":"1h","limit":1}, auth=False)
    async def liquidations(self, sym):
        # side="Buy" = long position liquidated
        return await self._get("/v5/market/liquidation",
            {"category":"linear","symbol":sym}, auth=False)

    # ── Account ──────────────────────────────────────────────────────────────
    async def wallet(self):
        return await self._get("/v5/account/wallet-balance", {"accountType":"UNIFIED"})
    async def positions(self, sym=None):
        p = {"category":"linear","settleCoin":"USDT","limit":"50"}
        if sym: p["symbol"] = sym
        return await self._get("/v5/position/list", p)
    async def closed_pnl(self, sym=None, limit=100):
        p = {"category":"linear","limit":limit}
        if sym: p["symbol"] = sym
        return await self._get("/v5/position/closed-pnl", p)

    # ── Orders ───────────────────────────────────────────────────────────────
    async def set_leverage(self, sym, lev):
        return await self._post("/v5/position/set-leverage",
            {"category":"linear","symbol":sym,"buyLeverage":str(lev),"sellLeverage":str(lev)})
    async def place_order(self, sym, side, qty, sl=None, tp=None, reduce=False):
        body = {"category":"linear","symbol":sym,"side":side,
                "orderType":"Market","qty":str(qty),"timeInForce":"GTC"}
        if sl:     body["stopLoss"]    = str(sl);  body["slTriggerBy"] = "LastPrice"
        if tp:     body["takeProfit"]  = str(tp);  body["tpTriggerBy"] = "LastPrice"
        if reduce: body["reduceOnly"]  = True
        return await self._post("/v5/order/create", body)
    async def cancel_all(self, sym=None):
        body = {"category":"linear"}
        if sym: body["symbol"] = sym
        return await self._post("/v5/order/cancel-all", body)
    async def close_pos(self, sym, side, qty):
        return await self.place_order(sym, "Sell" if side=="Buy" else "Buy", qty, reduce=True)
    async def set_tpsl(self, sym, sl=None, tp=None, trailing=None):
        body = {"category":"linear","symbol":sym,"positionIdx":0}
        if sl:       body["stopLoss"]     = str(sl)
        if tp:       body["takeProfit"]   = str(tp)
        if trailing: body["trailingStop"] = str(trailing)
        return await self._post("/v5/position/trading-stop", body)
