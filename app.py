# app.py —— WeCom(企业微信) ↔ OpenAI 网关
# 特性：
# - 安全/明文兼容；签名自检（定位 Token 问题）
# - 手工 AES 解密（绕过 wechatpy 对加密 XML 的解析，修复 ExpatError）
# - gzip/deflate 自动解压；body 只读一次
# - OpenAI gpt-5 兼容（max_completion_tokens、固定 temperature）
# - 空内容兜底与 44004 自动重试
# - 详细日志（不泄漏密钥）

import os
import re
import json
import gzip
import zlib
import time
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple

import httpx
import xmltodict
from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse, JSONResponse

# 可选：wechatpy（仅作兜底，不再依赖其解析加密 XML）
try:
    from wechatpy.enterprise.crypto import WeChatCrypto  # type: ignore
    from wechatpy.utils import to_text as _wechat_to_text  # type: ignore
except Exception:
    WeChatCrypto = None
    _wechat_to_text = None

# 手工 AES 解密所需
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
except Exception as _e:
    raise RuntimeError("缺少 cryptography 依赖，请在 requirements.txt 中加入 `cryptography`") from _e

# ----------------------- 日志 -----------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wecom-app")

def to_text(val):
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
OPENAI_API_KEY  = (os.getenv("OPENAI_API_KEY", "") or "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").strip()
OPENAI_ORG_ID   = (os.getenv("OPENAI_ORG_ID", "") or "").strip() or None

PRIMARY_MODEL   = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "").strip()
FALLBACK_MODELS = [m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-4o-mini").split(",") if m.strip()]

WEWORK_CORP_ID  = (os.getenv("WEWORK_CORP_ID", "") or "").strip()
WEWORK_SECRET   = (os.getenv("WEWORK_SECRET", "") or "").strip()
WEWORK_AGENT_ID = (os.getenv("WEWORK_AGENT_ID", "") or "").strip()

WECOM_TOKEN     = (os.getenv("WECOM_TOKEN", "") or "").strip()
WECOM_AES_KEY   = (os.getenv("WECOM_AES_KEY", "") or "").strip()   # 43 位

HTTP_TIMEOUT    = float(os.getenv("HTTP_TIMEOUT", "15"))

# ----------------------- OpenAI Chat Completions -----------------------
OPENAI_CHAT_COMPLETIONS_URL = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
OAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
if OPENAI_ORG_ID:
    OAI_HEADERS["OpenAI-Organization"] = OPENAI_ORG_ID

async def call_openai_chat(model: str, messages: list, max_tokens: int = 800, temperature: Optional[float] = None) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
        # gpt-5 仅支持默认 temperature=1，不要传
    else:
        payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(OPENAI_CHAT_COMPLETIONS_URL, headers=OAI_HEADERS, json=payload)
        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = {"raw": r.text}
            logger.error("OpenAI call failed: %s - %s", r.status_code, detail)
            raise RuntimeError(f"openai_error_{r.status_code}")
        data = r.json()
        content = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return (content or "").strip()

async def ask_llm(user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": (user_text or "").strip() or "（空消息）"},
    ]
    try:
        ans = await call_openai_chat(PRIMARY_MODEL, messages, max_tokens=800)
        if not ans:
            raise RuntimeError("empty content from primary model")
        return ans
    except Exception as e:
        logger.warning("primary model %s failed: %s", PRIMARY_MODEL, e)

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
        _token_cache["token"] = token
        _token_cache["expire_at"] = now + int(data.get("expires_in", 7200)) - 120
        return token

async def send_text(to_user: str, content: str):
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
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, json=_payload(txt))
        ret = r.json()
        logger.warning("WeCom send result -> to=%s payload_len=%d resp=%s", to_user, len(txt), ret)
        if ret.get("errcode") == 0:
            return
        if ret.get("errcode") == 44004:
            # 为空再补一个占位符重试
            r2 = await client.post(url, json=_payload("。"))
            ret2 = r2.json()
            logger.warning("WeCom retry result -> resp=%s", ret2)
            if ret2.get("errcode") == 0:
                return
        raise RuntimeError(f"WeCom send err: {ret}")

# ----------------------- 工具：签名、自检、载荷修复 -----------------------
def _head_preview(s: str) -> str:
    return re.sub(r"[\r\n\t ]+", " ", (s or "")[:120])

def _ensure_xml_for_wechatpy(body_text: str) -> Tuple[str, str]:
    t = (body_text or "").lstrip("\ufeff").strip()
    if not t:
        return "", "empty"
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            enc = obj.get("Encrypt") or obj.get("encrypt") or ""
            tou = obj.get("ToUserName") or obj.get("to_user_name") or WEWORK_CORP_ID
            if enc:
                xml = f"<xml><ToUserName><![CDATA[{tou}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
                return xml, "json→xml"
        except Exception as e:
            return "", f"json error: {e}"
    if t.startswith("<"):
        if ("<Encrypt>" in t) or ("<Encrypt><![CDATA[" in t):
            return t, "xml"
    m = (re.search(r'"Encrypt"\s*:\s*"([^"]+)"', t) or
         re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", t, re.S) or
         re.search(r"encrypt=([A-Za-z0-9+/=]+)", t))
    if m:
        enc = m.group(1)
        xml = f"<xml><ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
        return xml, "regex-salvaged"
    return "", "no-encrypt-field"

def compute_msg_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    arr = [token or "", str(timestamp or ""), str(nonce or ""), encrypt or ""]
    arr.sort()
    raw = "".join(arr).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def validate_crypto_env():
    ok = True
    aes = WECOM_AES_KEY
    if not WECOM_TOKEN:
        logger.error("crypto-check: WECOM_TOKEN missing")
        ok = False
    if not aes or len(aes) != 43 or not re.fullmatch(r"[A-Za-z0-9+/]{43}", aes):
        logger.error("crypto-check: WECOM_AES_KEY invalid (len=%s, preview=%r)", len(aes or ""), (aes[:6] + "..." if aes else aes))
        ok = False
    if not WEWORK_CORP_ID or not WEWORK_CORP_ID.startswith("ww"):
        logger.warning("crypto-check: WEWORK_CORP_ID looks unusual: %r", WEWORK_CORP_ID)
    return ok

try:
    validate_crypto_env()
except Exception:
    logger.exception("crypto-check: unexpected error when validating env")

# ----------------------- 手工 AES 解密（绕过 wechatpy 对密文 XML 的解析） -----------------------
def _pkcs7_unpad(b: bytes) -> bytes:
    if not b:
        raise ValueError("empty plaintext")
    pad = b[-1]
    if pad < 1 or pad > 32:
        raise ValueError(f"bad padding: {pad}")
    return b[:-pad]

def wecom_manual_decrypt(encrypt_b64: str, aes_key_43: str, corp_id: str) -> str:
    """
    企业微信消息体手工解密：
    - AESKey = base64_decode(EncodingAESKey + '=') -> 32 bytes
    - iv = AESKey[:16]
    - plaintext = AES-256-CBC decrypt(base64_decode(encrypt))
    - 去 PKCS#7；去掉 16B random；取 4B 网络序 msg_len；接下来的 msg_len 为明文XML；末尾为 receiveid
    - 校验 receiveid == corp_id（不相等仅告警）
    """
    if not encrypt_b64:
        raise ValueError("encrypt payload empty")
    if not aes_key_43 or len(aes_key_43) != 43:
        raise ValueError("aes key illegal")

    key = base64.b64decode(aes_key_43 + "=")  # 32 bytes
    if len(key) != 32:
        raise ValueError(f"aes key len != 32: {len(key)}")
    iv = key[:16]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    cipher_bytes = base64.b64decode(encrypt_b64)
    plain_padded = decryptor.update(cipher_bytes) + decryptor.finalize()
    plain = _pkcs7_unpad(plain_padded)

    if len(plain) < 20:
        raise ValueError("plaintext too short")

    content = plain[16:]  # skip 16 random bytes
    msg_len = int.from_bytes(content[0:4], "big")
    xml_bytes = content[4:4 + msg_len]
    recv_id  = content[4 + msg_len:].decode("utf-8", "ignore")

    if corp_id and recv_id and (corp_id != recv_id):
        logger.warning("manual-decrypt: corp_id mismatch expected=%s got=%s", corp_id, recv_id)

    return xml_bytes.decode("utf-8", "ignore")

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

# 探活/回调校验
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
    - 安全模式：签名自检 + 手工解密（优先）；wechatpy 仅作为最后兜底
    - 明文模式：直接解析
    - gzip/deflate 自动解压 & body 只读一次
    """
    msg_signature = request.query_params.get("msg_signature")
    timestamp     = request.query_params.get("timestamp")
    nonce         = request.query_params.get("nonce")

    raw = await request.body()
    enc_hdr = (request.headers.get("Content-Encoding") or "").lower()
    try:
        if enc_hdr == "gzip":
            raw = gzip.decompress(raw)
        elif enc_hdr == "deflate":
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception as e:
        logger.warning("decompress fail enc=%s err=%s", enc_hdr, e)

    text = raw.decode("utf-8", "ignore").lstrip("\ufeff").strip()
    is_safe_mode = bool(msg_signature and (WECOM_AES_KEY or "").strip())
    data: Dict[str, Any] = {}

    if is_safe_mode:
        # 先从 body 中提取 Encrypt，不依赖 wechatpy
        xml_or_how = _ensure_xml_for_wechatpy(text)
        xml_for_wechatpy, how = xml_or_how[0], xml_or_how[1]
        encrypt_val = ""

        if xml_for_wechatpy:
            try:
                parsed = xmltodict.parse(xml_for_wechatpy).get("xml", {}) or {}
                encrypt_val = (parsed.get("Encrypt") or "").strip()
            except Exception as e:
                logger.warning("pre-parse xml fail: %s", e)

        if not encrypt_val:
            m = re.search(br"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", raw, re.S) \
                or re.search(br'"Encrypt"\s*:\s*"([^"]+)"', raw)
            if m:
                encrypt_val = m.group(1).decode("utf-8", "ignore")

        if not encrypt_val:
            logger.error("safe-mode: cannot find Encrypt; head=%r", _head_preview(text))
            return PlainTextResponse("success")

        # 签名自检（定位 Token 问题）
        calc_sig = compute_msg_signature(WECOM_TOKEN, timestamp or "", nonce or "", encrypt_val)
        if (msg_signature or "").lower() != calc_sig.lower():
            logger.error(
                "safe-mode: signature mismatch! given=%s calc=%s token_preview=%s ts=%s nonce=%s enc_head=%r",
                msg_signature, calc_sig, (WECOM_TOKEN[:4] + "***"), timestamp, nonce, _head_preview(encrypt_val)
            )
            return PlainTextResponse("success")

        # —— 手工解密（避免 wechatpy 在解析密文 XML 时抛 ExpatError）——
        try:
            decrypted_xml = wecom_manual_decrypt(encrypt_val, WECOM_AES_KEY, WEWORK_CORP_ID)
            logger.info("safe-mode: manual-decrypt ok; head=%r", _head_preview(decrypted_xml))
        except Exception as e:
            logger.exception("manual-decrypt fail: %s: %s", e.__class__.__name__, e)
            # 兜底再尝试 wechatpy（极少需要）
            if WeChatCrypto is None:
                return PlainTextResponse("success")
            try:
                rebuilt = f"<xml><ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName><Encrypt><![CDATA[{encrypt_val}]]></Encrypt></xml>"
                crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
                decrypted_xml = crypto.decrypt_message(msg_signature, str(timestamp), str(nonce), rebuilt)
                logger.info("safe-mode: wechatpy-decrypt ok after manual failure")
            except Exception as e2:
                logger.exception("wechatpy-decrypt fail: %s: %s", e2.__class__.__name__, e2)
                return PlainTextResponse("success")

        # 解析明文 XML
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

    if (content or "").lower() == "/ping":
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    if msg_type == "text":
        reply = await ask_llm(content or "（空消息）")
        try:
            await send_text(from_user, reply)
        except Exception:
            logger.exception("send_text failed")
        return PlainTextResponse("success")

    await send_text(from_user, f"已收到消息类型：{msg_type}（暂未实现该类型处理）")
    return PlainTextResponse("success")
