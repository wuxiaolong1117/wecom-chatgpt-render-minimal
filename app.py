# app.py  —— WeCom (企业微信) ↔ OpenAI 网关（明/密文兼容，健壮解密）

import os
import re
import json
import gzip
import zlib
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple

import httpx
import xmltodict
from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse, JSONResponse

# ----------------------- 日志 -----------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wecom-app")

# ----------------------- 可选导入 wechatpy（仅安全模式解密需要） -----------------------
try:
    from wechatpy.enterprise.crypto import WeChatCrypto  # type: ignore
    from wechatpy.utils import to_text as _wechat_to_text  # type: ignore
except Exception:
    WeChatCrypto = None
    _wechat_to_text = None

def to_text(val):
    """wechatpy.utils.to_text 的轻量兜底：未安装 wechatpy 时也能用。"""
    if _wechat_to_text is not None:
        return _wechat_to_text(val)
    if val is None:
        return ""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", "ignore")
        except Exception:
            return val.decode("latin1", "ignore")
    return str(val)

# ----------------------- 环境变量 -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "") or None

# 模型主备（主：OPENAI_MODEL；备：OPENAI_FALLBACK_MODELS 逗号分隔）
PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FALLBACK_MODELS = [m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-4o-mini").split(",") if m.strip()]

# 企业微信
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")

# 其它
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15"))

# ----------------------- OpenAI 客户端（HTTP 调用） -----------------------
# 统一用 httpx 直调 Chat Completions（兼容 gpt-5 的参数差异）
OPENAI_CHAT_COMPLETIONS_URL = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
OAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
if OPENAI_ORG_ID:
    OAI_HEADERS["OpenAI-Organization"] = OPENAI_ORG_ID

async def call_openai_chat(model: str, messages: list, max_tokens: int = 800, temperature: Optional[float] = None) -> str:
    """
    兼容 gpt-5：不传 temperature（固定为默认1），并使用 max_completion_tokens。
    其它模型：使用 max_tokens 与可选 temperature。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # gpt-5 / gpt-5-mini 参数差异
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
        # 不传 temperature（gpt-5 仅支持默认 1）
    else:
        payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(OPENAI_CHAT_COMPLETIONS_URL, headers=OAI_HEADERS, json=payload)
        if r.status_code != 200:
            # 提供清晰错误日志
            try:
                detail = r.json()
            except Exception:
                detail = {"raw": r.text}
            logger.error("OpenAI call failed: %s - %s", r.status_code, detail)
            raise RuntimeError(f"openai_error_{r.status_code}")
        data = r.json()
        content = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return content.strip()

async def ask_llm(user_text: str) -> str:
    """主备模型自动回退，且避免空响应。"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text.strip()},
    ]
    # 主模型
    try:
        ans = await call_openai_chat(PRIMARY_MODEL, messages, max_tokens=800)
        if not ans:
            raise RuntimeError("empty content from primary model")
        return ans
    except Exception as e:
        logger.warning("primary model %s failed: %s", PRIMARY_MODEL, e)

    # 备选模型（依次尝试）
    for m in FALLBACK_MODELS:
        try:
            ans = await call_openai_chat(m, messages, max_tokens=800)
            if ans:
                return ans
        except Exception as e:
            logger.warning("fallback model %s failed: %s", m, e)

    return "（抱歉，我现在有点忙，稍后再试试吧。）"

# ----------------------- 企业微信：AccessToken 缓存 -----------------------
_token_cache: Dict[str, Any] = {"token": None, "expire_at": 0}

async def get_wecom_token() -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["expire_at"] - now > 60:
        return str(_token_cache["token"])

    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url)
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"gettoken err: {data}")
        token = data["access_token"]
        # 官方通常7200s，这里略缩，提前过期
        _token_cache["token"] = token
        _token_cache["expire_at"] = now + int(data.get("expires_in", 7200)) - 120
        return token

async def send_text(to_user: str, content: str):
    """
    企业微信文本消息发送。避免 errcode=44004（empty content）：
    - 去除两端空白后为空时，用占位符替代。
    - 首次失败 44004 再自动补位重试一次。
    """
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    def _payload(txt: str) -> Dict[str, Any]:
        return {
            "touser": to_user,
            "msgtype": "text",
            "agentid": int(WEWORK_AGENT_ID),
            "text": {"content": txt},
            "safe": 0,
            "enable_id_trans": 0,
            "enable_duplicate_check": 0,
        }

    txt = content if content and content.strip() else "（空回复）"
    payload = _payload(txt)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        ret = r.json()
        logger.warning("WeCom send result -> to=%s payload_len=%d resp=%s", to_user, len(txt), ret)
        if ret.get("errcode") == 0:
            return

        # 避免 empty content
        if ret.get("errcode") == 44004:
            logger.warning("WeCom send first attempt failed: %s, retrying...", ret)
            txt2 = txt if txt.strip() else "。"  # 非空白字符
            r2 = await client.post(url, json=_payload(txt2))
            ret2 = r2.json()
            logger.warning("WeCom retry result -> resp=%s", ret2)
            if ret2.get("errcode") == 0:
                return

        raise RuntimeError(f"WeCom send err: {ret}")

# ----------------------- 工具：日志预览 & 载荷修复 -----------------------
def _head_preview(s: str) -> str:
    return re.sub(r"[\r\n\t ]+", " ", (s or "")[:120])

def _ensure_xml_for_wechatpy(body_text: str) -> Tuple[str, str]:
    """
    把各种形态的 body 变成 wechatpy.decrypt_message 所需 XML。
    返回: (xml_string, how)。为空表示失败。
    """
    t = (body_text or "").lstrip("\ufeff").strip()
    if not t:
        return "", "empty"

    # JSON -> xml
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            enc = obj.get("Encrypt") or obj.get("encrypt") or ""
            tou = obj.get("ToUserName") or obj.get("to_user_name") or ""
            if enc:
                xml = f"<xml><ToUserName><![CDATA[{tou}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
                return xml, "json→xml"
        except Exception as e:
            return "", f"json error: {e}"

    # 原本就是 XML
    if t.startswith("<"):
        if ("<Encrypt>" in t) or ("<Encrypt><![CDATA[" in t):
            return t, "xml"

    # 兜底：正则抢救 Encrypt
    m = (re.search(r'"Encrypt"\s*:\s*"([^"]+)"', t) or
         re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", t, re.S) or
         re.search(r"encrypt=([A-Za-z0-9+/=]+)", t))
    if m:
        enc = m.group(1)
        xml = f"<xml><ToUserName><![CDATA[]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
        return xml, "regex-salvaged"

    return "", "no-encrypt-field"

# ----------------------- FastAPI -----------------------
app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "mode": "safe" if (WECOM_AES_KEY or "").strip() else "plain",
        "service": "WeCom + ChatGPT",
        "model": PRIMARY_MODEL,
        "candidates": FALLBACK_MODELS,
        "pdf_support": False,
        "local_ocr": False,
    })

# 探活（企业微信 & Render）
@app.get("/wecom/callback")
async def wecom_callback_get(request: Request):
    echostr = request.query_params.get("echostr")
    if echostr:
        return PlainTextResponse(echostr)
    return PlainTextResponse("ok")

# 回调（消息）
@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    """
    企业微信回调（消息）。兼容：
    - 安全模式（加密）：msg_signature 存在时走 WeChatCrypto 解密
    - 明文模式：直接 xmltodict 解析
    - 只读一次 body；若 gzip/deflate 则自动解压
    - 解析失败安全返回 "success" 防止企业微信重试
    """
    # -------- query --------
    msg_signature = request.query_params.get("msg_signature")
    timestamp     = request.query_params.get("timestamp")
    nonce         = request.query_params.get("nonce")

    # -------- body（只读一次）+ 解压 --------
    raw = await request.body()
    enc_hdr = (request.headers.get("Content-Encoding") or "").lower()
    try:
        if enc_hdr == "gzip":
            raw = gzip.decompress(raw)
        elif enc_hdr == "deflate":
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception as e:
        logger.warning("safe-mode: decompress fail enc=%s err=%s", enc_hdr, e)
    text = raw.decode("utf-8", "ignore").lstrip("\ufeff").strip()

    is_safe_mode = bool(msg_signature and (WECOM_AES_KEY or "").strip())
    data: Dict[str, Any] = {}

    if is_safe_mode:
        if WeChatCrypto is None:
            logger.error("decrypt fail: wechatpy not installed")
            return PlainTextResponse("success")

        xml_for_wechatpy, how = _ensure_xml_for_wechatpy(text)
        if not xml_for_wechatpy:
            # 再尝试直接从原始字节救一次（避免 decode 破坏结构）
            m = re.search(br"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", raw, re.S) or \
                re.search(br'"Encrypt"\s*:\s*"([^"]+)"', raw)
            if m:
                enc = (m.group(1).decode("utf-8", "ignore"))
                xml_for_wechatpy = f"<xml><ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
                how = "bytes-regex-salvaged"
            else:
                logger.error(
                    "safe-mode: cannot convert body to xml (%s); head=%r; ct=%s",
                    how, _head_preview(text), request.headers.get("Content-Type")
                )
                return PlainTextResponse("success")

        # 预解析验证（有的网关会在前面插奇怪字节）
        try:
            _ = xmltodict.parse(xml_for_wechatpy)
        except Exception as e:
            logger.warning("safe-mode: pre-parse xml fail (%s), try rebuild from bytes", e)
            m = re.search(br"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", raw, re.S) or \
                re.search(br'"Encrypt"\s*:\s*"([^"]+)"', raw)
            if m:
                enc = m.group(1).decode("utf-8", "ignore")
                xml_for_wechatpy = f"<xml><ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
                how += "+rebuild"
            else:
                logger.error("safe-mode: salvage-from-bytes failed. head_bytes=%r", raw[:100])
                return PlainTextResponse("success")

        logger.info("safe-mode: using payload (%s); head=%r", how, _head_preview(xml_for_wechatpy))

        try:
            crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
            decrypted_xml = crypto.decrypt_message(
                msg_signature, str(timestamp), str(nonce), xml_for_wechatpy
            )
        except Exception:
            logger.exception("ERROR:wecom-app:decrypt fail (safe-mode)")
            return PlainTextResponse("success")

        try:
            data = xmltodict.parse(to_text(decrypted_xml)).get("xml", {}) or {}
        except Exception:
            logger.exception("safe-mode: parse decrypted xml fail. head=%r", _head_preview(to_text(decrypted_xml)))
            return PlainTextResponse("success")

    else:
        # 明文模式
        if not text or not text.startswith("<"):
            logger.warning("plain-mode: body not xml. head=%r", _head_preview(text))
            return PlainTextResponse("success")
        try:
            data = xmltodict.parse(text).get("xml", {}) or {}
        except Exception:
            logger.exception("plain-mode: parse xml fail. head=%r", _head_preview(text))
            return PlainTextResponse("success")

    # -------- 业务处理 --------
    msg_type  = (data.get("MsgType") or "").lower()
    from_user = (data.get("FromUserName") or "").strip()
    content   = (data.get("Content") or "").strip()
    pic_url   = (data.get("PicUrl") or "").strip()
    media_id  = (data.get("MediaId") or "").strip()

    # 简单命令
    if content.lower() == "/ping":
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    # 文本消息 -> OpenAI
    if msg_type == "text":
        reply = await ask_llm(content or "（空消息）")
        try:
            await send_text(from_user, reply)
        except Exception:
            logger.exception("send_text failed")
        return PlainTextResponse("success")

    # 其它类型先回一条提示（可扩展：图片/语音/文件转写等）
    await send_text(from_user, f"已收到消息类型：{msg_type}（暂未实现该类型处理）")
    return PlainTextResponse("success")
