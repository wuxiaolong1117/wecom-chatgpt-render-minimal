import os
import json
import time
import base64
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

import xmltodict

# 企业微信加解密
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# OpenAI 官方 SDK
from openai import OpenAI

# ====== 日志 ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wecom-app")

# ====== 环境变量 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")          # 主模型（默认 gpt-5）
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini") # 回退模型
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))

# 企业微信
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")

# 记忆
REDIS_URL = os.getenv("REDIS_URL", "")   # eg. redis://:pwd@host:6379/0
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "8"))

# 联网搜索
WEB_SEARCH_DEFAULT = os.getenv("WEB_SEARCH_DEFAULT", "off").lower() in ("1", "true", "yes", "on")
# 只做“工具是否可用”的判定，不在此实现具体爬取流程（让模型自行整合）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ====== OpenAI 客户端 ======
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is empty!")
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=(OPENAI_ORG_ID or None),
)

# ====== HTTP 客户端 ======
TIMEOUT = httpx.Timeout(20.0, connect=10.0)
client = httpx.AsyncClient(timeout=TIMEOUT)

# ====== 企业微信加解密实例（密文模式时使用）======
crypto: Optional[WeChatCrypto] = None
if WECOM_TOKEN and WECOM_AES_KEY and WEWORK_CORP_ID:
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
        logger.info("WeCom safe-mode enabled.")
    except Exception as e:
        logger.warning(f"WeCom safe-mode init failed: {e}")

# ====== 记忆：Redis 首选，失败退内存 ======
class MemoryStore:
    def __init__(self):
        self.use_redis = False
        self.redis = None
        self.mem: Dict[str, List[Tuple[str, str]]] = {}

        if REDIS_URL:
            try:
                import redis  # type: ignore
                self.redis = redis.from_url(REDIS_URL, decode_responses=True)
                self.redis.ping()
                self.use_redis = True
                logger.info("Memory: using Redis.")
            except Exception as e:
                logger.warning(f"load memory failed: {e}")

    def _key(self, uid: str) -> str:
        return f"wecom:mem:{uid}"

    def load(self, uid: str) -> List[Tuple[str, str]]:
        if self.use_redis:
            try:
                data = self.redis.get(self._key(uid))
                if data:
                    return json.loads(data)
                return []
            except Exception as e:
                logger.warning(f"load memory failed: {e}")
                return []
        return self.mem.get(uid, [])

    def append(self, uid: str, role: str, content: str):
        if not content:
            return
        arr = self.load(uid)
        arr.append((role, content))
        arr = arr[-MEMORY_WINDOW:]
        if self.use_redis:
            try:
                self.redis.set(self._key(uid), json.dumps(arr), ex=60*60*24*7)
            except Exception as e:
                logger.warning(f"append memory failed: {e}")
        else:
            self.mem[uid] = arr

MEM = MemoryStore()

# ====== Web Search 工具开关 & 提供商检测 ======
web_enabled_flag = WEB_SEARCH_DEFAULT  # 进程级开关，可用 /web on off 切
def has_search_provider() -> bool:
    return bool(GOOGLE_API_KEY and GOOGLE_CSE_ID) or bool(SERPER_API_KEY) or bool(TAVILY_API_KEY)

# ====== WeCom Token 缓存 ======
_wecom_token_cache = {"token":"", "exp":0}

async def get_wecom_token() -> str:
    now = int(time.time())
    if _wecom_token_cache["token"] and _wecom_token_cache["exp"] - now > 60:
        return _wecom_token_cache["token"]
    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    r = await client.get(url)
    data = r.json()
    token = data.get("access_token", "")
    if token:
        _wecom_token_cache["token"] = token
        _wecom_token_cache["exp"] = now + data.get("expires_in", 7200)
    return token

# ====== WeCom 发送文本（分片 & 兜底）======
async def send_text(userid: str, text: str):
    # WeCom 不允许空内容
    if not text or not text.strip():
        text = "（生成内容为空，建议换个问法或稍后再试）"
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    max_len = 1800  # 安全阈值
    chunks = []
    s = text
    while s:
        chunks.append(s[:max_len])
        s = s[max_len:]

    sent = 0
    for part in chunks:
        payload = {
            "touser": userid,
            "agentid": int(WEWORK_AGENT_ID),
            "msgtype": "text",
            "text": {"content": part},
            "safe": 0,
        }
        r = await client.post(url, json=payload)
        try:
            ret = r.json()
        except Exception:
            ret = {"errcode": -1, "errmsg": "invalid json"}
        logger.warning(f"WeCom send result -> to={userid} payload_len={len(part)} resp={ret}")
        # 44004 空内容容错（极短碎片或全是不可见字符）
        if ret.get("errcode") == 44004:
            payload["text"]["content"] = "（生成内容为空，建议换个问法或稍后再试）"
            r2 = await client.post(url, json=payload)
            logger.warning(f"WeCom send first attempt failed: {ret}, retrying...")
            try:
                ret2 = r2.json()
            except Exception:
                ret2 = {"errcode": -1, "errmsg": "invalid json"}
            logger.warning(f"WeCom send result -> to={userid} payload_len={len(payload['text']['content'])} resp={ret2}")
            if ret2.get("errcode") != 0:
                raise RuntimeError(f"WeCom send err: {ret2}")
        elif ret.get("errcode") != 0:
            raise RuntimeError(f"WeCom send err: {ret}")
        sent += 1
    return sent

# ====== OpenAI 结果取值与兜底（补丁②）======
def _pick_text(choices: List[Dict[str, Any]]) -> str:
    """
    统一提取 message.content；若只有 tool_calls 则提示“已调用联网检索…”；仍为空则返回空串。
    """
    if not choices:
        return ""
    for ch in choices:
        msg = ch.get("message") or {}
        txt = (msg.get("content") or "").strip()
        if txt:
            return txt
    # 如果只有 tool_calls，无正文
    for ch in choices:
        msg = ch.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            return "（已调用联网检索工具，正在整合结果…如无输出请稍后重试或换种问法）"
    return ""

# ====== FastAPI ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=JSONResponse)
async def root():
    status = {
        "status":"ok",
        "mode":"safe" if crypto else "plain",
        "service":"WeCom + ChatGPT",
        "model": OPENAI_MODEL,
        "memory": "redis" if MEM.use_redis else "memory",
        "pdf_support": True,
        "local_ocr": False,
    }
    return status

# ====== 企业微信回调 ======
@app.api_route("/wecom/callback", methods=["GET", "POST"])
async def wecom_callback(request: Request):
    # --- 验证 GET （包括安全模式 echo）---
    if request.method == "GET":
        args = request.query_params
        echostr = args.get("echostr")
        if echostr:
            # 安全模式校验
            if "msg_signature" in args and crypto:
                msg_signature = args.get("msg_signature", "")
                timestamp = args.get("timestamp", "")
                nonce = args.get("nonce", "")
                try:
                    echo = crypto.decrypt(echostr, msg_signature, timestamp, nonce)
                    return PlainTextResponse(echo)
                except Exception as e:
                    logger.error(f"wecom-app:URL verify decrypt failed: {e}")
                    return PlainTextResponse("invalid", status_code=400)
            return PlainTextResponse(echostr)
        return PlainTextResponse("ok")

    # --- POST 处理消息 ---
    body = await request.body()
    args = request.query_params
    is_safe_mode = "msg_signature" in args and crypto is not None

    raw_xml = ""
    if is_safe_mode:
        try:
            msg_signature = args.get("msg_signature", "")
            timestamp = args.get("timestamp", "")
            nonce = args.get("nonce", "")
            raw_xml = crypto.decrypt(body.decode("utf-8"), msg_signature, timestamp, nonce)
        except InvalidSignatureException:
            logger.error("InvalidSignatureException")
            return PlainTextResponse("signature error", status_code=403)
        except Exception as e:
            logger.error(f"decrypt fail: {e}")
            return PlainTextResponse("decrypt error", status_code=400)
    else:
        raw_xml = body.decode("utf-8")

    try:
        data = xmltodict.parse(raw_xml).get("xml", {})
    except Exception as e:
        logger.error(f"xml parse error: {e}")
        return PlainTextResponse("bad xml", status_code=400)

    from_user = data.get("FromUserName", "")
    msg_type = (data.get("MsgType", "") or "").lower()
    content = (data.get("Content", "") or "").strip()

    # --- 命令 ---
    if content.startswith("/ping"):
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    if content.startswith("/model"):
        candidates = f"{OPENAI_MODEL}, gpt-5-mini, gpt-4o-mini"
        mem_kind = "redis" if MEM.use_redis else "memory"
        org = OPENAI_ORG_ID or "-"
        txt = (
            f"当前活跃模型：{OPENAI_MODEL}\n"
            f"候选列表：{candidates}\n"
            f"组织ID：{org}\n"
            f"记忆：{mem_kind}"
        )
        await send_text(from_user, txt)
        return PlainTextResponse("success")

    # /web on /web off /web xxx
    global web_enabled_flag
    if content.startswith("/web"):
        seg = content.split(None, 1)
        if len(seg) == 1:
            await send_text(from_user, "用法：/web on|off 或 /web 你的问题（临时联网）")
            return PlainTextResponse("success")
        arg = seg[1].strip()
        if arg in ("on","off","true","false","开启","关闭"):
            web_enabled_flag = arg in ("on","true","开启")
            await send_text(from_user, f"Web access 已{'开启' if web_enabled_flag else '关闭'}")
            return PlainTextResponse("success")
        # 临时联网：直接把问题改写为用户问法
        content = arg

    # --- 普通文本 ---
    if msg_type == "text":
        # 取记忆
        history = MEM.load(from_user)
        messages = [{"role":"system","content":"你是企业微信助手，回答简洁、可读性强；当被要求联网搜索时可引用结果给出简要结论和参考链接。"}]
        for r, c in history:
            messages.append({"role":r, "content":c})
        messages.append({"role":"user", "content":content})

        # ===== 决定是否启用工具（补丁①：仅在有提供商 + 开关开启时传）=====
        use_web_tool = (web_enabled_flag and has_search_provider())

        tools: List[Dict[str, Any]] = []
        if use_web_tool:
            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web and synthesize a concise Chinese answer with 3-5 citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type":"string","description":"中文检索式"},
                            "top_k": {"type":"integer","minimum":1,"maximum":10,"default":3}
                        },
                        "required": ["query"]
                    }
                }
            }]

        # ===== 主模型 =====
        kwargs: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "messages": messages,
        }
        # gpt-5 系列：不传 temperature，不传 max_tokens，用 max_completion_tokens
        if MAX_COMPLETION_TOKENS:
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        if tools:                          # 只有在存在工具时才传 tools / tool_choice（补丁①）
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            completion = oai.chat.completions.create(**kwargs)
            reply_text = _pick_text(completion.choices)  # （补丁②）
        except Exception as e:
            logger.error(f"OpenAI call failed(main): {e}")
            reply_text = ""

        # ===== 回退模型 =====
        if not reply_text.strip():
            logger.warning("primary model failed: empty content from primary model")
            fb_kwargs: Dict[str, Any] = {
                "model": FALLBACK_MODEL,
                "messages": messages,
            }
            # 4o-mini 可设 temperature，但保持默认即可；同样遵守“有 tools 才传”
            if MAX_COMPLETION_TOKENS:
                fb_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
            if tools:
                fb_kwargs["tools"] = tools
                fb_kwargs["tool_choice"] = "auto"
            try:
                completion2 = oai.chat.completions.create(**fb_kwargs)
                reply_text = _pick_text(completion2.choices)
            except Exception as e:
                logger.error(f"OpenAI call failed(fallback): {e}")
                reply_text = ""

        # 存记忆
        MEM.append(from_user, "user", content)
        if reply_text.strip():
            MEM.append(from_user, "assistant", reply_text)

        # 发送
        await send_text(from_user, reply_text)
        return PlainTextResponse("success")

    # 非文本消息
    await send_text(from_user, "暂时只支持文本消息（PDF/图片等请稍后使用“文件消息处理”版本）")
    return PlainTextResponse("success")

# ====== 关机清理 ======
@app.on_event("shutdown")
async def on_shutdown():
    await client.aclose()

# ====== 启动 ======
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
