# -*- coding: utf-8 -*-
import os
import re
import json
import time
import asyncio
import logging
from typing import Dict, List, Tuple, Optional

import httpx
import xmltodict
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

# === OpenAI v1 SDK ===
from openai import OpenAI

# === WeCom 加解密 ===
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.utils import to_text
from xml.parsers.expat import ExpatError

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("wecom-app")

# ========== 环境变量 ==========
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")  # 组织 ID（可选，但你已验证，就显式带上）

# 模型与回退
PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-5").strip()
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-5-mini,gpt-4o-mini").split(",") if m.strip()
]

# 记忆
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()  # redis/memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# WeCom
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")

# 安全模式兼容与解密降级
WECOM_SAFE_MODE = os.getenv("WECOM_SAFE_MODE", "true").lower() == "true"

# PDF/图片摘要相关（保留占位，后续继续扩展）
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
LOCAL_OCR_ENABLE = os.getenv("LOCAL_OCR_ENABLE", "false").lower() == "true"
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_SUMMARY = int(os.getenv("CHUNK_SUMMARY", "400"))

# 联网搜索（本次新增，并默认使用 SerpAPI）
WEB_SEARCH_ENABLE = os.getenv("WEB_SEARCH_ENABLE", "false").lower() == "true"
WEB_PROVIDER = os.getenv("WEB_PROVIDER", "serpapi").lower()  # serpapi / cse
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "")

# ========== OpenAI 客户端 ==========
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,  # 显式带上组织
)

# ========== FastAPI ==========
app = FastAPI()

# ========== 记忆实现（内存 + Redis 兼容） ==========
class MemoryBase:
    async def load(self, uid: str) -> List[Dict]:
        raise NotImplementedError

    async def append(self, uid: str, role: str, content: str):
        raise NotImplementedError


class InMemory(MemoryBase):
    def __init__(self, limit=12):
        self.store: Dict[str, List[Dict]] = {}
        self.limit = limit

    async def load(self, uid: str) -> List[Dict]:
        return self.store.get(uid, [])

    async def append(self, uid: str, role: str, content: str):
        arr = self.store.setdefault(uid, [])
        arr.append({"role": role, "content": content})
        if len(arr) > self.limit:
            self.store[uid] = arr[-self.limit :]


class RedisMemory(MemoryBase):
    def __init__(self, url: str, limit=12, namespace="mem:"):
        self.url = url
        self.limit = limit
        self.ns = namespace
        try:
            import redis.asyncio as redis  # type: ignore
            self.r = redis.from_url(self.url, decode_responses=True)
        except Exception as e:
            logger.warning("Redis init failed: %s", e)
            self.r = None

    async def load(self, uid: str) -> List[Dict]:
        if not self.r:
            raise RuntimeError("Redis not ready")
        key = f"{self.ns}{uid}"
        data = await self.r.get(key)
        return json.loads(data) if data else []

    async def append(self, uid: str, role: str, content: str):
        if not self.r:
            raise RuntimeError("Redis not ready")
        key = f"{self.ns}{uid}"
        data = await self.r.get(key)
        arr = json.loads(data) if data else []
        arr.append({"role": role, "content": content})
        if len(arr) > self.limit:
            arr = arr[-self.limit :]
        await self.r.set(key, json.dumps(arr), ex=60 * 60 * 24 * 7)


if MEMORY_BACKEND == "redis":
    memory: MemoryBase = RedisMemory(REDIS_URL)
else:
    memory = InMemory()

# ========== WeCom 发送/鉴权 ==========
_token_cache = {"value": "", "expire": 0}


async def get_wecom_token() -> str:
    now = time.time()
    if _token_cache["value"] and _token_cache["expire"] > now + 30:
        return _token_cache["value"]
    url = (
        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        tok = data.get("access_token", "")
        exp = int(data.get("expires_in", 7200))
        _token_cache["value"] = tok
        _token_cache["expire"] = now + exp - 60
        return tok


async def send_text(to_user: str, content: str):
    # 避免 44004 空文本
    txt = (content or "").strip()
    if not txt:
        txt = "（模型未输出可读文本，建议换个问法，或发送 /ping 自检出站链路。）"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(WEWORK_AGENT_ID),
        "text": {"content": txt[:4096]},  # 企微单条上限
        "safe": 0,
    }
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    async with httpx.AsyncClient(timeout=12.0) as client:
        r = await client.post(url, json=payload)
        try:
            data = r.json()
        except Exception:
            data = {"errcode": -1, "errmsg": "json decode error"}
        logger.warning("WeCom send result -> to=%s payload_len=%s resp=%s", to_user, len(txt), data)
        if data.get("errcode") != 0:
            # 再试一次（常见 44004）
            if data.get("errcode") == 44004:
                payload["text"]["content"] = "（消息过短或被过滤，已替换为占位文本。）"
                r2 = await client.post(url, json=payload)
                logger.warning("WeCom send first attempt failed: %s, retrying...", data)
                return r2.json()
            raise RuntimeError(f"WeCom send err: {data}")
        return data


# ========== 文本清洗与触发 ==========
def _normalize_text(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("／", "/").replace("：", ":")
    t = re.sub(r"\s+", " ", t)
    return t


def _want_web_route(t: str) -> Tuple[bool, str]:
    """
    触发词：
      /web
      web <query>
      search: <query>
      search <query>
      搜索: <query> / 搜索：<query>
    """
    s = _normalize_text(t).lower()
    raw = _normalize_text(t)
    prefixes = ["/web", "web ", "search:", "search ", "搜索:", "搜索："]
    for p in prefixes:
        if s.startswith(p):
            q = raw[len(p):].strip()
            return True, q
    return False, raw


async def _web_search_serpapi(query: str, k: int = 5):
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY missing")
    params = {
        "engine": "google",
        "q": query,
        "hl": "zh-cn",
        "gl": "cn",
        "num": k,
        "api_key": SERPAPI_KEY,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get("https://serpapi.com/search.json", params=params)
        r.raise_for_status()
        items = r.json().get("organic_results", [])
        return [
            {
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", ""),
            }
            for it in items
        ]


def _render_search(items, provider="serpapi") -> str:
    if not items:
        return "（联网搜索没有检索到结果，换个关键词或加上地点/时间再试）"
    out = [f"🔎 已联网搜索（{provider}）："]
    for i, it in enumerate(items[:5], 1):
        title = it.get("title") or ""
        url = it.get("url") or ""
        out.append(f"[{i}] {title}\n{url}")
    return "\n".join(out)


# ========== OpenAI 对话（主模型 + 回退） ==========
async def ask_models(messages: List[Dict], models: List[str]) -> Tuple[str, str]:
    """
    依次尝试模型，返回 (reply_text, used_model)
    - gpt-5 / gpt-5-mini 不要传 temperature / max_tokens（只用默认）
    """
    for m in models:
        try:
            completion = oai.chat.completions.create(
                model=m,
                messages=messages,
            )
            text = (completion.choices[0].message.content or "").strip()
            if text:
                return text, m
            else:
                logger.warning("primary model %s failed: empty content from primary model", m)
        except Exception as e:
            logger.error("OpenAI call failed for %s: %s", m, e)
    return "（模型临时没有返回内容，建议换个说法或稍后再试）", models[-1] if models else "unknown"


# ========== 状态接口 ==========
@app.get("/")
async def root():
    return JSONResponse(
        {
            "status": "ok",
            "mode": "safe" if WECOM_SAFE_MODE else "plain",
            "service": "WeCom + ChatGPT",
            "model": PRIMARY_MODEL,
            "fallbacks": FALLBACK_MODELS,
            "organization": OPENAI_ORG_ID or "",
            "memory": MEMORY_BACKEND,
            "pdf_support": True,
            "local_ocr": LOCAL_OCR_ENABLE,
            "web_search": {"enabled": WEB_SEARCH_ENABLE, "provider": WEB_PROVIDER},
        }
    )


# ========== WeCom 回调 ==========
@app.get("/wecom/callback")
async def wecom_verify(request: Request):
    """
    企业微信“接收消息服务器配置”校验；GET 验证回显
    """
    params = dict(request.query_params)
    msg_signature = params.get("msg_signature", "")
    timestamp = params.get("timestamp", "")
    nonce = params.get("nonce", "")
    echostr = params.get("echostr", "")

    crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
    try:
        echo = crypto.decrypt_message(msg_signature, timestamp, nonce, echostr)
        return PlainTextResponse(echo)
    except Exception as e:
        logger.error("wecom-app:URL verify decrypt failed: %s", e)
        return PlainTextResponse("invalid")


@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    """
    业务消息处理（POST）
    兼容企微“安全模式”：先拿原始 XML -> 解密 -> 解析 -> 回复
    """
    params = dict(request.query_params)
    msg_signature = params.get("msg_signature", "")
    timestamp = params.get("timestamp", "")
    nonce = params.get("nonce", "")

    crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)

    # 读取原始 XML（必须是 str）
    raw_bytes = await request.body()
    xml_text = raw_bytes.decode("utf-8", errors="ignore").strip()

    # 安全模式：先尝试直接解密整段 XML；若 ExpatError，退化为“只包 Encrypt 的极简 XML”再试一次
    decrypted_xml = None
    try:
        decrypted_xml = crypto.decrypt_message(msg_signature, timestamp, nonce, xml_text)
    except ExpatError:
        # 尝试用正则抓取 Encrypt，并重建极简 xml
        try:
            m = re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", xml_text, re.S)
            if m:
                enc = m.group(1)
                rebuilt = f"<xml><ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
                logger.warning("safe-mode: decrypt ExpatError, try minimal rebuilt xml once")
                decrypted_xml = crypto.decrypt_message(msg_signature, timestamp, nonce, rebuilt)
            else:
                raise
        except Exception as e2:
            logger.error("ERROR:wecom-app:decrypt fail (retry): ExpatError: %s", e2)
    except Exception as e:
        logger.error("ERROR:wecom-app:decrypt fail (safe-mode): %s", e)

    if not decrypted_xml:
        # 无法解密仍返回 200，避免企微重试
        return PlainTextResponse("success")

    # 解析明文 XML
    try:
        d = xmltodict.parse(decrypted_xml).get("xml", {})
    except Exception as e:
        logger.error("ERROR:wecom-app:xml parse fail: %s", e)
        return PlainTextResponse("success")

    msg_type = (d.get("MsgType") or "").lower()
    from_user = d.get("FromUserName") or ""
    content = (d.get("Content") or "").strip()

    # ---- 文本消息 ----
    if msg_type == "text":
        # /ping
        if content.strip().lower().startswith("/ping"):
            info = [
                f"当前活跃模型：{PRIMARY_MODEL}",
                f"候选列表：{', '.join([PRIMARY_MODEL] + FALLBACK_MODELS)}",
                f"组织ID：{OPENAI_ORG_ID or '(未设)'}",
                f"记忆：{MEMORY_BACKEND}",
                f"联网搜索：{'on' if WEB_SEARCH_ENABLE else 'off'} / {WEB_PROVIDER}",
            ]
            await send_text(from_user, "\n".join(info))
            return PlainTextResponse("success")

        # 联网搜索触发
        should_web, web_q = _want_web_route(content)
        if should_web:
            if not WEB_SEARCH_ENABLE:
                await send_text(from_user, "（联网搜索未启用：请把 WEB_SEARCH_ENABLE=true 并配置 SERPAPI_KEY）")
                return PlainTextResponse("success")
            try:
                items = await _web_search_serpapi(web_q, k=5)
                reply_text = _render_search(items, "serpapi")
            except Exception as e:
                logger.exception("web search failed")
                reply_text = f"（联网搜索出错：{e}）"
            await send_text(from_user, reply_text)
            return PlainTextResponse("success")

        # ---- 普通对话：加载记忆 + 多模型回退 ----
        try:
            history = []
            try:
                history = await memory.load(from_user)
            except Exception as e:
                logger.warning("load memory failed: %s", e)

            messages = [{"role": "system", "content": "你是企业微信里的智能助手，回答要简洁、直给。"}]
            messages.extend(history[-8:])
            messages.append({"role": "user", "content": content})

            models_try = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m]
            reply_text, used_model = await ask_models(messages, models_try)

            # 发送
            await send_text(from_user, reply_text)

            # 写记忆（不因 Redis 故障阻塞）
            try:
                await memory.append(from_user, "user", content)
                await memory.append(from_user, "assistant", reply_text)
            except Exception as e:
                logger.warning("append memory failed: %s", e)

            return PlainTextResponse("success")
        except Exception as e:
            logger.exception("biz error: %s", e)
            await send_text(from_user, "（服务端异常，可稍后再试或 /ping 自检）")
            return PlainTextResponse("success")

    # ---- 其它类型（图片/文件等，后续扩展 PDF/图片解析）----
    await send_text(from_user, "已收到消息（当前仅支持文本提问；文件/PDF/图片解析已接入变量与占位，稍后完善）。")
    return PlainTextResponse("success")
