# -*- coding: utf-8 -*-
"""
WeCom + OpenAI 一体化最小应用（Render 友好）
- 明文/安全模式自动适配
- OpenAI 组织/自定义 Base URL
- 模型候选与自动兜底
- 内存/Redis 会话记忆
- /ping /model /switch /web 等指令
- 联网搜索：Brave/SerpAPI/Tavily 三选一
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import httpx
import uvicorn
import xmltodict
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

# 企业微信安全模式
from wechatpy.enterprise.crypto import WeChatCrypto

# OpenAI (>=1.x)
from openai import OpenAI

# Redis 可选
try:
    from redis import Redis
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


# -----------------------------
# 基础配置 & 日志
# -----------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wecom-app")

app = FastAPI(title="WeCom + ChatGPT")

# -----------------------------
# 环境变量：WeCom
# -----------------------------
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")  # 有值即代表安全模式
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")

SAFE_MODE = bool(WECOM_AES_KEY)  # 自动判断
crypto: Optional[WeChatCrypto] = None
if SAFE_MODE:
    crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
    logger.info("WeCom safe-mode enabled.")
else:
    logger.info("WeCom plaintext mode enabled.")

# -----------------------------
# 环境变量：OpenAI
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "").strip()

# 模型候选与当前模型
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MODEL_CANDIDATES = [
    m.strip()
    for m in os.getenv("OPENAI_MODEL_CANDIDATES", f"{DEFAULT_MODEL}").split(",")
    if m.strip()
]
if DEFAULT_MODEL not in MODEL_CANDIDATES:
    MODEL_CANDIDATES.insert(0, DEFAULT_MODEL)

# 兜底模型 & “严格新模型”集合
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini").strip()
STRICT_NEW_MODELS = {"gpt-5", "gpt-5-mini"}

# 创建 OpenAI 客户端（显式组织）
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,
)

# -----------------------------
# 环境变量：记忆存储
# -----------------------------
CHAT_MEMORY_BACKEND = os.getenv("CHAT_MEMORY", "memory")  # memory | redis
MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "8"))  # 每个用户历史轮次
REDIS_URL = os.getenv("REDIS_URL", "redis://host:6379/0")

# 内存存储
_memory: Dict[str, List[Dict[str, str]]] = {}

# Redis 连接（可选）
_redis: Optional[Redis] = None
if CHAT_MEMORY_BACKEND == "redis" and Redis is not None:
    try:
        # 允许 "redis://:password@host:port/db"
        from urllib.parse import urlparse

        u = urlparse(REDIS_URL)
        _redis = Redis(
            host=u.hostname or "localhost",
            port=u.port or 6379,
            username=u.username,
            password=u.password,
            db=int((u.path or "/0")[1:] or "0"),
            socket_timeout=1.5,
            socket_connect_timeout=1.5,
            decode_responses=True,
        )
        # 健康探测
        _redis.ping()
        logger.info("Redis connected.")
    except Exception as e:
        logger.warning(f"Redis init failed: {e}")
        _redis = None

# -----------------------------
# 联网搜索配置
# -----------------------------
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "").lower()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 额外（可选）SerpAPI 定位参数
SERPAPI_HL = os.getenv("SERPAPI_HL")
SERPAPI_GL = os.getenv("SERPAPI_GL")
SERPAPI_LOCATION = os.getenv("SERPAPI_LOCATION")

# -----------------------------
# 运行时状态
# -----------------------------
app_state: Dict[str, Any] = {
    "active_model": DEFAULT_MODEL,
    "access_token": {
        "value": "",
        "expire_at": 0,
    },
}


# -----------------------------
# 工具函数：WeCom 发送消息
# -----------------------------
async def get_wecom_token() -> str:
    """获取并缓存企业微信 access_token"""
    now = int(time.time())
    cache = app_state["access_token"]
    if cache["value"] and cache["expire_at"] > now + 60:
        return cache["value"]

    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 7200))
        cache["value"] = token
        cache["expire_at"] = now + max(60, min(6900, expires_in - 60))
        return token


async def send_text(to_user: str, text: str) -> None:
    """
    发送文本消息到企业微信；防止空内容引发 44004，做最小填充。
    """
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "agentid": int(WEWORK_AGENT_ID),
        "msgtype": "text",
        "text": {"content": (text or "").strip()[:2048] or "…"},
        "safe": 0,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 失败重试 2 次
        for i in range(3):
            r = await client.post(url, json=payload)
            try:
                r.raise_for_status()
                ret = r.json()
            except Exception as e:
                logger.exception("WeCom http error")
                raise

            if ret.get("errcode") == 0:
                logger.warning(
                    f"WeCom send result -> to={to_user} payload_len={len(payload['text']['content'])} resp={ret}"
                )
                return
            else:
                if i < 2 and ret.get("errcode") in (44004, 42001, 40014):
                    logger.warning(f"WeCom send first attempt failed: {ret}, retrying...")
                    # 重新拉 token 或补一个字符
                    if ret.get("errcode") == 44004:
                        payload["text"]["content"] = payload["text"]["content"] or "…"
                    if ret.get("errcode") in (42001, 40014):
                        app_state["access_token"] = {"value": "", "expire_at": 0}
                        token = await get_wecom_token()
                        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
                    continue
                logger.warning(
                    f"WeCom send result -> to={to_user} payload_len={len(payload['text']['content'])} resp={ret}"
                )
                raise RuntimeError(f"WeCom send err: {ret}")


# -----------------------------
# 工具函数：记忆存取
# -----------------------------
def _mem_key(uid: str) -> str:
    return f"conv:{uid}"


def load_memory(uid: str) -> List[Dict[str, str]]:
    if CHAT_MEMORY_BACKEND == "redis" and _redis:
        try:
            s = _redis.get(_mem_key(uid))  # type: ignore
            if s:
                return json.loads(s)
        except Exception as e:
            logger.warning(f"load memory failed: {e}")
    return _memory.get(uid, [])


def save_memory(uid: str, history: List[Dict[str, str]]) -> None:
    history = history[-(MEMORY_LIMIT * 2 + 2) :]  # 截断
    if CHAT_MEMORY_BACKEND == "redis" and _redis:
        try:
            _redis.setex(_mem_key(uid), 60 * 60 * 24, json.dumps(history))  # type: ignore
            return
        except Exception as e:
            logger.warning(f"append memory failed: {e}")
    _memory[uid] = history


# -----------------------------
# 工具函数：OpenAI 统一调用（文本抽取 + 兜底回退）
# -----------------------------
def _extract_text_from_choice(choice) -> str:
    """尽量从 choice 中抽取可读文本"""
    try:
        msg = choice.message
        if isinstance(getattr(msg, "content", None), str) and msg.content.strip():
            return msg.content.strip()
        parts = getattr(msg, "content", None)
        if isinstance(parts, list):
            buf = []
            for p in parts:
                if isinstance(p, dict):
                    t = p.get("text") or p.get("content") or ""
                    if isinstance(t, str) and t.strip():
                        buf.append(t.strip())
            if buf:
                return "\n".join(buf).strip()
    except Exception:
        pass
    return ""


def _complete_text(messages: List[Dict[str, str]], model: str, max_tokens: int = 512) -> str:
    """统一聊天补全：主模型 → 回退模型"""
    # 第一次：主模型
    try:
        kwargs = dict(model=model, messages=messages)
        if model in STRICT_NEW_MODELS:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        resp = oai.chat.completions.create(**kwargs)
        txt = _extract_text_from_choice(resp.choices[0])
        if txt:
            return txt
        raise ValueError("empty content from primary model")
    except Exception as e:
        logger.warning(f"primary model {model} failed: {e}")

    # 第二次：回退
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        try:
            kwargs = dict(model=FALLBACK_MODEL, messages=messages, max_tokens=max_tokens)
            resp = oai.chat.completions.create(**kwargs)
            txt = _extract_text_from_choice(resp.choices[0])
            if txt:
                return f"{txt}\n\n（已自动使用 {FALLBACK_MODEL} 兜底生成）"
        except Exception as e2:
            logger.error(f"fallback model {FALLBACK_MODEL} failed: {e2}")

    return ""


# -----------------------------
# 工具函数：联网搜索
# -----------------------------
async def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """统一搜索封装"""
    if not SEARCH_PROVIDER:
        return []

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            if SEARCH_PROVIDER == "brave" and BRAVE_API_KEY:
                url = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": BRAVE_API_KEY}
                params = {"q": query, "count": k}
                r = await client.get(url, headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
                items = []
                for it in (data.get("web", {}).get("results", []) or [])[:k]:
                    items.append(
                        {
                            "title": it.get("title") or "",
                            "url": it.get("url") or "",
                            "snippet": it.get("description") or "",
                        }
                    )
                return items

            if SEARCH_PROVIDER == "serpapi" and SERPAPI_API_KEY:
                url = "https://serpapi.com/search.json"
                params = {
                    "engine": "google",
                    "q": query,
                    "num": k,
                    "api_key": SERPAPI_API_KEY,
                    "hl": SERPAPI_HL or None,
                    "gl": SERPAPI_GL or None,
                    "location": SERPAPI_LOCATION or None,
                }
                params = {kk: vv for kk, vv in params.items() if vv}
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                items = []
                for it in (data.get("organic_results") or [])[:k]:
                    items.append(
                        {
                            "title": it.get("title") or "",
                            "url": it.get("link") or "",
                            "snippet": it.get("snippet") or "",
                        }
                    )
                return items

            if SEARCH_PROVIDER == "tavily" and TAVILY_API_KEY:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": k,
                }
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                items = []
                for it in (data.get("results") or [])[:k]:
                    items.append(
                        {
                            "title": it.get("title") or "",
                            "url": it.get("url") or "",
                            "snippet": it.get("content") or it.get("snippet") or "",
                        }
                    )
                return items
    except Exception as e:
        logger.warning(f"web_search error: {e}")

    return []


# -----------------------------
# 健康检查 & 首页
# -----------------------------
@app.get("/")
async def index():
    return JSONResponse(
        {
            "status": "ok",
            "mode": "safe" if SAFE_MODE else "plain",
            "service": "WeCom + ChatGPT",
            "model": ", ".join([app_state.get("active_model", DEFAULT_MODEL)] + [m for m in MODEL_CANDIDATES if m != app_state.get("active_model", DEFAULT_MODEL)]),
            "memory": CHAT_MEMORY_BACKEND,
            "pdf_support": True,
            "local_ocr": False,
        }
    )


# -----------------------------
# WeCom 回调
# -----------------------------
@app.api_route("/wecom/callback", methods=["GET", "POST"])
async def wecom_callback(request: Request):
    # 1) URL 验证（带 echostr）
    q = dict(request.query_params)
    echostr = q.get("echostr")
    if request.method == "GET" and echostr is not None:
        if SAFE_MODE and crypto:
            # 安全模式：解密 echostr
            try:
                echo = crypto.decrypt(
                    echostr,
                    q.get("msg_signature", ""),
                    q.get("timestamp", ""),
                    q.get("nonce", ""),
                )
                return PlainTextResponse(echo)
            except Exception as e:
                logger.exception("URL verify decrypt failed")
                return PlainTextResponse("invalid", status_code=400)
        else:
            # 明文模式：回显
            return PlainTextResponse(echostr or "")

    # 2) 接收消息（POST）
    raw = await request.body()
    text_xml: str
    if SAFE_MODE and crypto:
        try:
            text_xml = crypto.decrypt_message(
                raw.decode("utf-8"),
                q.get("msg_signature", ""),
                q.get("timestamp", ""),
                q.get("nonce", ""),
            )
        except Exception as e:
            logger.exception("decrypt_message failed")
            return PlainTextResponse("invalid", status_code=400)
    else:
        text_xml = raw.decode("utf-8")

    try:
        data = xmltodict.parse(text_xml)
    except Exception as e:
        logger.exception("parse xml failed")
        return PlainTextResponse("success")  # 避免重试风暴

    msg = data.get("xml") or {}
    msg_type = (msg.get("MsgType") or "").lower().strip()
    from_user = (msg.get("FromUserName") or "").strip()
    content = (msg.get("Content") or "").strip()
    low = content.lower()

    # 3) 指令：/ping
    if low == "/ping":
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    # 4) 模型信息/切换
    def _is_model_query(txt: str) -> bool:
        t = (txt or "").strip().lower()
        if t in {"/model", "model", "/version", "version"}:
            return True
        cn = txt or ""
        if "模型" in cn and any(k in cn for k in ["版本", "型号", "名称"]):
            return True
        if any(k in cn for k in ["什么模型", "用的什么模型"]):
            return True
        return False

    if _is_model_query(content):
        info = (
            f"当前活跃模型：{app_state.get('active_model', DEFAULT_MODEL)}\n"
            f"候选列表：{', '.join(MODEL_CANDIDATES)}\n"
            f"组织ID：{OPENAI_ORG_ID or '（未设置）'}"
        )
        await send_text(from_user, info)
        return PlainTextResponse("success")

    if low.startswith(("/switch ", "/use ")):
        try:
            new_m = low.split(None, 1)[1].strip()
        except Exception:
            new_m = ""
        if not new_m:
            await send_text(
                from_user, f"用法：/switch <model>\n候选：{', '.join(MODEL_CANDIDATES)}"
            )
            return PlainTextResponse("success")
        if new_m not in MODEL_CANDIDATES:
            await send_text(
                from_user, f"不在候选列表：{new_m}\n候选：{', '.join(MODEL_CANDIDATES)}"
            )
            return PlainTextResponse("success")
        app_state["active_model"] = new_m
        await send_text(from_user, f"已切换到：{new_m}")
        return PlainTextResponse("success")

    # 5) /web 或 “联网搜索”自然问法
    if low.startswith("/web ") or ("联网搜索" in (content or "")):
        qk = content
        if low.startswith("/web "):
            qk = low.split(" ", 1)[1].strip()
        if not qk:
            await send_text(from_user, "用法：/web 关键词\n例如：/web iPhone 16 发布时间")
            return PlainTextResponse("success")

        results = await web_search(qk, k=5)
        if not results:
            await send_text(
                from_user,
                "未配置搜索 API 或没有查到结果。请在环境变量中配置 SEARCH_PROVIDER 与对应 KEY。",
            )
            return PlainTextResponse("success")

        bullets = []
        for i, it in enumerate(results, 1):
            bullets.append(f"[{i}] {it['title']}\n{it['snippet']}\n{it['url']}")
        prompt = (
            "你是一名高质量的信息总结助手。请基于下列搜索结果，回答用户问题：\n\n"
            + "\n\n".join(bullets)
            + "\n\n要求：\n"
            "1) 用中文简洁回答；\n"
            "2) 只使用给定结果中的可证实信息；\n"
            "3) 在句末用 [序号] 标注引用；\n"
            "4) 若结果不足以得出结论，请直说不确定。\n\n"
            f"用户问题：{qk}"
        )
        messages = [
            {"role": "system", "content": "你只能依据提供的检索片段回答，不要编造来源。"},
            {"role": "user", "content": prompt},
        ]
        reply_text = _complete_text(
            messages, model=app_state.get("active_model", DEFAULT_MODEL), max_tokens=512
        )
        if not reply_text:
            reply_text = "搜索总结失败：模型未输出可读文本。"

        tail = "\n\n参考：\n" + "\n".join(
            [f"[{i+1}] {r['url']}" for i, r in enumerate(results)]
        )
        await send_text(from_user, (reply_text + tail)[:2048])
        return PlainTextResponse("success")

    # 自然问法：能否联网
    if any(
        k in (content or "")
        for k in ["能联网搜索吗", "可以联网搜索吗", "联网搜索吗", "是否联网", "能联网吗"]
    ):
        await send_text(
            from_user, "已支持联网搜索。\n用法：/web 关键词\n例如：/web OpenAI 新模型定价"
        )
        return PlainTextResponse("success")

    # 6) 普通文本对话
    if msg_type == "text" and content:
        # 记忆
        hist = load_memory(from_user)

        base_system = {
            "role": "system",
            "content": "你是一个企业微信助手，输出纯文本，不要 Markdown，不要代码块，不要链接预览。",
        }
        messages: List[Dict[str, str]] = [base_system]
        messages.extend(hist)
        messages.append({"role": "user", "content": content})

        reply_text = _complete_text(
            messages, model=app_state.get("active_model", DEFAULT_MODEL), max_tokens=768
        )
        if not reply_text:
            reply_text = "（模型未输出可读文本。已尝试回退模型仍失败）"

        # 追加记忆并保存
        hist.append({"role": "user", "content": content})
        hist.append({"role": "assistant", "content": reply_text})
        save_memory(from_user, hist)

        await send_text(from_user, reply_text)
        return PlainTextResponse("success")

    # 非文本：略
    await send_text(from_user, "（暂不支持该类型消息）")
    return PlainTextResponse("success")


# -----------------------------
# 本地启动（Render 用 Start Command：uvicorn app:app --host 0.0.0.0 --port $PORT）
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
