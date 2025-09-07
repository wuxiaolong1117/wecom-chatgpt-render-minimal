# app.py  — WeCom ↔ OpenAI 适配（Render）
# - 兼容明文/安全模式
# - Redis/内存双记忆
# - /ping /model 指令
# - 模型回退 & 组织ID
# - 两处稳健性补丁（OpenAI 强制文本、企微发送加固）

import os
import re
import time
import json
import hmac
import base64
import hashlib
import asyncio
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
import xmltodict
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException
from openai import OpenAI
import logging

# --------------------
# 基础配置 & 日志
# --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("wecom-app")

APP_PORT = int(os.getenv("PORT", "10000"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5").strip()

# 模型回退序列（从前到后尝试）
MODEL_CANDIDATES = [
    OPENAI_MODEL,
    "gpt-5-mini",
    "gpt-4o-mini",
]

# WeCom
WECOM_CORP_ID = os.getenv("WEWORK_CORP_ID", os.getenv("WECOM_CORP_ID", "")).strip()
WECOM_AGENT_ID = int(os.getenv("WEWORK_AGENT_ID", os.getenv("WECOM_AGENT_ID", "0")))
WECOM_SECRET = os.getenv("WEWORK_SECRET", os.getenv("WECOM_SECRET", "")).strip()
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "").strip()
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "").strip()

# Memory
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory")  # memory | redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

SAFE_MODE = True  # 自动兼容：若签名/Encrypt 存在就按加密处理，否则按明文

# --------------------
# OpenAI 客户端
# --------------------
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID,  # 显式带上组织
)

# --------------------
# 简易会话记忆：优先 Redis，失败则退回内存
# --------------------
class Memory:
    def __init__(self):
        self.backend = MEMORY_BACKEND
        self._mem: Dict[str, List[Dict]] = {}
        self.redis = None

    async def init(self):
        if self.backend == "redis":
            try:
                import redis.asyncio as redis
                self.redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
                await self.redis.ping()
                logger.info("Memory: redis connected")
            except Exception as e:
                logger.warning("load memory failed: %s", e)
                self.backend = "memory"

    async def load(self, key: str) -> List[Dict]:
        if self.backend == "redis" and self.redis:
            try:
                raw = await self.redis.get(f"hist:{key}")
                if not raw:
                    return []
                return json.loads(raw)
            except Exception as e:
                logger.warning("load memory failed: %s", e)
                return []
        return self._mem.get(key, [])

    async def save(self, key: str, messages: List[Dict]):
        if self.backend == "redis" and self.redis:
            try:
                await self.redis.set(f"hist:{key}", json.dumps(messages), ex=60 * 60 * 24)
                return
            except Exception as e:
                logger.warning("append memory failed: %s", e)
        self._mem[key] = messages[-20:]  # 内存模式压缩到 20 轮

memory = Memory()

# --------------------
# 工具函数：WeCom 获取 token（自动缓存）
# --------------------
_token_cache = {"token": "", "exp": 0}

async def get_wecom_token() -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["exp"] - now > 60:
        return _token_cache["token"]

    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WECOM_CORP_ID}&corpsecret={WECOM_SECRET}"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
        if data.get("errcode") == 0:
            _token_cache["token"] = data["access_token"]
            _token_cache["exp"] = now + data.get("expires_in", 7200)
            return _token_cache["token"]
        raise RuntimeError(f"gettoken failed: {data}")

# --------------------
# 工具函数：发送文本到 WeCom（补丁 B：更稳健发送）
# --------------------
async def send_text(to_user: str, text: str) -> Dict:
    # 兜底：模型若输出空文本，仍返回可读提示
    if not text or not text.strip():
        text = "（模型未输出文本，建议换个问法或稍后再试）"

    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": WECOM_AGENT_ID,
        "text": {"content": text},
        "safe": 0,
    }

    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    # 总超时加大；失败重试 3 次；不向 ASGI 顶层抛异常
    timeout = httpx.Timeout(15.0)
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=payload)
                data = r.json()
                logger.warning(
                    "WeCom send result -> to=%s payload_len=%s resp=%s",
                    to_user, len(text.encode("utf-8")), data
                )
                if data.get("errcode") == 0:
                    return data
                # 常见业务错误（如 44004 empty content），稍等重试
                await asyncio.sleep(1.2 * (attempt + 1))
        except httpx.ConnectTimeout:
            logger.warning("WeCom send attempt %s timeout, will retry...", attempt + 1)
            await asyncio.sleep(1.2 * (attempt + 1))
        except Exception as e:
            logger.exception("WeCom send attempt %s failed: %s", attempt + 1, e)
            await asyncio.sleep(1.2 * (attempt + 1))

    logger.error("WeCom send failed after retries, give up.")
    return {"errcode": -1, "errmsg": "send_failed_after_retries"}

# --------------------
# WeCom 加解密（URL 验证 & 消息解密）
# --------------------
def get_crypto() -> Optional[WeChatCrypto]:
    try:
        if WECOM_TOKEN and WECOM_AES_KEY and WECOM_CORP_ID:
            return WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORP_ID)
    except Exception as e:
        logger.warning("init WeChatCrypto failed: %s", e)
    return None

def try_decrypt_echo(crypto: WeChatCrypto, echostr: str, signature: str, ts: str, nonce: str) -> str:
    """URL 验证回调：兼容安全模式；失败回退原串"""
    try:
        # wechatpy 的 decrypt_message 也可用于 echostr 解密
        s = crypto.decrypt_message(echostr, signature, ts, nonce)
        if isinstance(s, bytes):
            s = s.decode("utf-8", "ignore")
        return s
    except Exception:
        return echostr

def try_decrypt_xml(crypto: WeChatCrypto, raw_xml: str, signature: str, ts: str, nonce: str) -> str:
    """正常消息：若为加密模式，解密出明文 xml；失败回退原文"""
    try:
        xml = crypto.decrypt_message(raw_xml, signature, ts, nonce)
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8", "ignore")
        return xml
    except InvalidSignatureException:
        logger.warning("invalid signature for encrypted message")
    except Exception as e:
        logger.warning("decrypt xml failed: %s", e)
    return raw_xml

# --------------------
# OpenAI 调用（补丁 A：强制文本输出 & 禁用 tools）
# --------------------
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是企业微信里的智能助手，回答要简洁、直接、可执行。"
)

async def call_openai_with_fallback(user_id: str, text: str) -> str:
    """携带会话记忆，按候选模型回退调用"""
    history = await memory.load(user_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": text}
    ]

    # 针对 gpt-5 系列：使用 max_completion_tokens（而非 max_tokens）
    for idx, model in enumerate(MODEL_CANDIDATES):
        try:
            completion = oai.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=800,
                # —— 补丁 A：强制产出文本 & 禁用工具调用（避免空 content）
                response_format={"type": "text"},
                tool_choice="none",
            )
            reply = (completion.choices[0].message.content or "").strip()
            if reply:
                # 记录到记忆
                history.extend([
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ])
                await memory.save(user_id, history)
                return reply

            logger.warning("primary model %s failed: empty content from primary model", model)
        except Exception as e:
            logger.warning("model %s call failed: %s", model, e)

    # 全部失败兜底
    return "（模型临时没有返回内容，建议换个说法或稍后再试）"

# --------------------
# 指令处理
# --------------------
def parse_command(text: str) -> Optional[Dict]:
    if not text:
        return None
    if text.startswith("/ping"):
        return {"cmd": "ping"}
    m = re.match(r"^/model\s+([\w\-\.\:]+)$", text.strip(), re.I)
    if m:
        return {"cmd": "model", "arg": m.group(1)}
    return None

async def handle_command(cmd: Dict, from_user: str) -> str:
    if cmd["cmd"] == "ping":
        return (
            f"当前活跃模型：{MODEL_CANDIDATES[0]}\n"
            f"候选列表：{', '.join(MODEL_CANDIDATES)}\n"
            f"组织ID：{OPENAI_ORG_ID or '-'}\n"
            f"记忆：{memory.backend}"
        )
    if cmd["cmd"] == "model":
        newm = cmd["arg"]
        # 简单改第一个候选模型
        MODEL_CANDIDATES[0] = newm
        return f"已切换首选模型为：{newm}"
    return "未知指令"

# --------------------
# FastAPI
# --------------------
app = FastAPI()

@app.on_event("startup")
async def _on_startup():
    await memory.init()
    logger.info("WeCom safe-mode enabled.")
    logger.info("Service starting on port %s", APP_PORT)

@app.get("/")
async def index():
    return {
        "status": "ok",
        "mode": "safe",
        "service": "WeCom + ChatGPT",
        "model": ",".join(MODEL_CANDIDATES),
        "memory": memory.backend,
        "pdf_support": True,
        "local_ocr": False
    }

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

# --------------------
# WeCom 回调
# --------------------
@app.get("/wecom/callback")
async def wecom_get(echostr: Optional[str] = None,
                    msg_signature: Optional[str] = None,
                    timestamp: Optional[str] = None,
                    nonce: Optional[str] = None):
    """URL 验证"""
    if not echostr:
        return PlainTextResponse("")

    if SAFE_MODE and msg_signature and WECOM_TOKEN and WECOM_AES_KEY:
        crypto = get_crypto()
        if crypto:
            s = try_decrypt_echo(crypto, echostr, msg_signature, timestamp or "", nonce or "")
            return PlainTextResponse(s)
    # 明文回显
    return PlainTextResponse(echostr)

@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    query = request.query_params
    msg_signature = query.get("msg_signature")
    timestamp = query.get("timestamp")
    nonce = query.get("nonce")

    raw_body = await request.body()
    raw_xml = raw_body.decode("utf-8", "ignore")

    # 兼容安全模式：如果存在 Encrypt 或 query 带签名，就尝试解密
    if SAFE_MODE and msg_signature:
        crypto = get_crypto()
        if crypto:
            # 加密 xml 解密
            raw_xml = try_decrypt_xml(crypto, raw_xml, msg_signature, timestamp or "", nonce or "")

    # 解析 XML
    try:
        data = xmltodict.parse(raw_xml).get("xml", {})
    except Exception:
        data = {}

    from_user = data.get("FromUserName", "")
    msg_type = data.get("MsgType", "text")
    content = (data.get("Content") or "").strip()

    # 指令优先
    cmd = parse_command(content)
    if cmd:
        reply = await handle_command(cmd, from_user)
        await send_text(from_user, reply)
        return PlainTextResponse("success")

    # 普通文本，走 OpenAI
    if msg_type == "text" and content:
        reply_text = await call_openai_with_fallback(from_user, content)
        await send_text(from_user, reply_text)
        return PlainTextResponse("success")

    # 其他类型（图片/文件/事件等）简单兜底
    await send_text(from_user, "我暂时只支持文本消息。可以直接把问题发给我～")
    return PlainTextResponse("success")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
