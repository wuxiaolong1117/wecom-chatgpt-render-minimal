import os
import time
import json
import logging
import threading
from collections import deque
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from lxml import etree
from openai import OpenAI

from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# ---------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wecom-app")

app = FastAPI(title="WeCom + ChatGPT with Memory Support")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
oai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# WeCom 配置
CORP_ID = os.getenv("WEWORK_CORP_ID", "")
AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
SECRET = os.getenv("WEWORK_SECRET", "")

WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")

crypto: Optional[WeChatCrypto] = None
if WECOM_TOKEN and WECOM_AES_KEY and CORP_ID:
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, CORP_ID)
        log.info("WeCom safe-mode enabled.")
    except Exception as e:
        log.exception("Init WeChatCrypto failed: %s", e)
else:
    log.info("WeCom running in PLAINTEXT mode.")

# ---------------------------------------------------------------------
# Access Token 缓存
# ---------------------------------------------------------------------
_TOKEN_CACHE = {"value": None, "exp": 0}

async def get_wecom_token() -> str:
    now = int(time.time())
    if _TOKEN_CACHE["value"] and _TOKEN_CACHE["exp"] > now + 30:
        return _TOKEN_CACHE["value"]

    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORP_ID}&corpsecret={SECRET}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
    access_token = data.get("access_token", "")
    _TOKEN_CACHE["value"] = access_token
    _TOKEN_CACHE["exp"] = now + 7000
    return access_token

async def send_text(to_user: str, content: str) -> dict:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(AGENT_ID),
        "text": {"content": content[:2048]},
        "safe": 0,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, json=payload)
        return r.json()

def parse_plain_xml(raw_xml: str) -> dict:
    root = etree.fromstring(raw_xml.encode("utf-8"))
    def get(tag: str) -> Optional[str]:
        el = root.find(tag)
        return el.text if el is not None else None
    return {
        "ToUserName": get("ToUserName"),
        "FromUserName": get("FromUserName"),
        "CreateTime": get("CreateTime"),
        "MsgType": get("MsgType"),
        "Content": get("Content"),
        "MsgId": get("MsgId"),
        "AgentID": get("AgentID"),
        "Event": get("Event"),
    }

# ---------------------------------------------------------------------
# Memory：内存 + Redis 双实现
# ---------------------------------------------------------------------
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "false").lower() == "true"
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))
MEMORY_TTL = int(os.getenv("MEMORY_TTL_SECONDS", "86400"))
REDIS_URL = os.getenv("REDIS_URL", "")

# 内存存储
class MemoryStore:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._data = {}
    def get(self, user_id: str):
        with self._lock:
            return list(self._data.get(user_id, deque()))
    def append_turn(self, user_id: str, user_text: str, assistant_text: str):
        with self._lock:
            dq = self._data.setdefault(user_id, deque())
            dq.append(("user", user_text))
            dq.append(("assistant", assistant_text))
            while len(dq) > 2 * self.max_turns:
                dq.popleft()
            self._data[user_id] = dq

MEMORY = MemoryStore(max_turns=MEMORY_TURNS) if (ENABLE_MEMORY and not REDIS_URL) else None

# Redis 存储
REDIS_MEMORY = None
if ENABLE_MEMORY and REDIS_URL:
    try:
        from redis import asyncio as aioredis
        class RedisMemory:
            def __init__(self, url: str, max_turns: int = 6, ttl: int = 86400):
                self.url = url
                self.max_turns = max_turns
                self.ttl = ttl
                self._client = None
            async def client(self):
                if self._client is None:
                    self._client = await aioredis.from_url(self.url, decode_responses=True)
                return self._client
            def key(self, user_id: str) -> str:
                return f"wecom:chat:history:{user_id}"
            async def get(self, user_id: str):
                cli = await self.client()
                data = await cli.lrange(self.key(user_id), 0, -1)
                return [json.loads(x) for x in data]
            async def append_turn(self, user_id: str, user_text: str, assistant_text: str):
                cli = await self.client()
                k = self.key(user_id)
                await cli.rpush(k, json.dumps({"role":"user","content":user_text}))
                await cli.rpush(k, json.dumps({"role":"assistant","content":assistant_text}))
                await cli.ltrim(k, -2*self.max_turns, -1)
                await cli.expire(k, self.ttl)
        REDIS_MEMORY = RedisMemory(REDIS_URL, MEMORY_TURNS, MEMORY_TTL)
        log.info("Redis memory enabled.")
    except Exception as e:
        log.warning("Redis init failed: %s", e)

# ---------------------------------------------------------------------
# URL 验证
# ---------------------------------------------------------------------
@app.get("/wecom/callback", response_class=PlainTextResponse)
async def wecom_verify(request: Request,
                       echostr: Optional[str] = None,
                       msg_signature: Optional[str] = None,
                       timestamp: Optional[str] = None,
                       nonce: Optional[str] = None):
    if crypto and all([msg_signature, timestamp, nonce, echostr]):
        try:
            xml = f"<xml><ToUserName><![CDATA[{CORP_ID}]]></ToUserName><Encrypt><![CDATA[{echostr}]]></Encrypt></xml>"
            echo = crypto.decrypt_message(xml, msg_signature, timestamp, nonce)
            return PlainTextResponse(echo)
        except InvalidSignatureException:
            return PlainTextResponse("invalid signature", status_code=403)
        except Exception as e:
            log.exception("URL verify decrypt failed: %s", e)
            return PlainTextResponse(f"decrypt failed: {e}", status_code=500)
    return echostr or ""

# ---------------------------------------------------------------------
# 消息回调
# ---------------------------------------------------------------------
@app.post("/wecom/callback", response_class=PlainTextResponse)
async def wecom_callback(request: Request):
    raw = await request.body()
    params = request.query_params
    msg_signature = params.get("msg_signature")
    timestamp = params.get("timestamp")
    nonce = params.get("nonce")

    try:
        if crypto and all([msg_signature, timestamp, nonce]):
            decrypted_xml = crypto.decrypt_message(raw.decode("utf-8"), msg_signature, timestamp, nonce)
            msg = parse_plain_xml(decrypted_xml)
        else:
            msg = parse_plain_xml(raw.decode("utf-8"))
    except Exception as e:
        log.exception("parse/decrypt failed: %s", e)
        return JSONResponse({"ok": False, "error": f"parse/decrypt failed: {e}"}, status_code=200)

    from_user = msg.get("FromUserName") or ""
    content = msg.get("Content") or msg.get("Event") or ""

    # 构造上下文
    base_system = {"role": "system", "content": "You are a helpful assistant for WeCom users."}
    messages = [base_system]

    if ENABLE_MEMORY and from_user:
        try:
            if REDIS_MEMORY:
                history = await REDIS_MEMORY.get(from_user)
                messages.extend(history)
            elif MEMORY:
                for role, text in MEMORY.get(from_user):
                    messages.append({"role": role, "content": text})
        except Exception as e:
            log.warning("load memory failed: %s", e)

    messages.append({"role": "user", "content": content})

    # 调用 OpenAI
    try:
        completion = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        reply_text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        log.exception("OpenAI call failed: %s", e)
        reply_text = f"OpenAI 调用失败：{e}"

    # 写入记忆
    if ENABLE_MEMORY and from_user:
        try:
            if REDIS_MEMORY:
                await REDIS_MEMORY.append_turn(from_user, content, reply_text)
            elif MEMORY:
                MEMORY.append_turn(from_user, content, reply_text)
        except Exception as e:
            log.warning("append memory failed: %s", e)

    # 回复用户
    try:
        await send_text(from_user, reply_text)
    except Exception as e:
        log.exception("WeCom send failed: %s", e)
        return JSONResponse({"ok": False, "error": f"WeCom send failed: {e}"}, status_code=200)

    return PlainTextResponse("success")

# ---------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "mode": "safe" if crypto else "plain",
        "service": "WeCom + ChatGPT",
        "model": OPENAI_MODEL,
        "memory": "redis" if REDIS_MEMORY else ("memory" if MEMORY else "disabled"),
    }
