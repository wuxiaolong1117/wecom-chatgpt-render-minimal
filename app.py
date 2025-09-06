import os
import time
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from lxml import etree
from openai import OpenAI

# 企业微信安全模式加解密
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# ---------------------------------------------------------------------
# 基础设置
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wecom-app")

app = FastAPI(title="WeCom + ChatGPT (plaintext & safe-mode compatible)")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
oai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# WeCom 基本配置
CORP_ID = os.getenv("WEWORK_CORP_ID", "")
AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
SECRET = os.getenv("WEWORK_SECRET", "")

# 安全模式密钥（留空则自动走明文模式）
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")

crypto: Optional[WeChatCrypto] = None
if WECOM_TOKEN and WECOM_AES_KEY and CORP_ID:
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, CORP_ID)
        log.info("WeCom safe-mode enabled (crypto ready).")
    except Exception as e:
        log.exception("Init WeChatCrypto failed: %s", e)
else:
    log.info("WeCom running in PLAINTEXT mode (no token/aes key).")

# ---------------------------------------------------------------------
# WeCom access_token 缓存
# ---------------------------------------------------------------------
_TOKEN_CACHE = {"value": None, "exp": 0}


async def get_wecom_token() -> str:
    now = int(time.time())
    if _TOKEN_CACHE["value"] and _TOKEN_CACHE["exp"] > now + 30:
        return _TOKEN_CACHE["value"]

    url = (
        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={CORP_ID}&corpsecret={SECRET}"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
    access_token = data.get("access_token", "")
    if not access_token:
        log.error("Get access_token failed: %s", data)
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
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}


def parse_plain_xml(raw_xml: str) -> dict:
    """
    解析（已解密或明文的）微信 XML。
    """
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
# 1) URL 验证（明文 / 安全模式自动兼容）
# ---------------------------------------------------------------------
@app.get("/wecom/callback", response_class=PlainTextResponse)
async def wecom_verify(
    request: Request,
    echostr: Optional[str] = None,
    msg_signature: Optional[str] = None,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
):
    # 安全模式
    if crypto and all([msg_signature, timestamp, nonce, echostr]):
        try:
            # 直接解密 echostr（不是 XML）
            echo = crypto.decrypt(echostr, msg_signature, timestamp, nonce)
            return PlainTextResponse(echo)
        except InvalidSignatureException:
            log.warning("URL verify invalid signature.")
            return PlainTextResponse("invalid signature", status_code=403)
        except Exception as e:
            log.exception("URL verify decrypt failed: %s", e)
            return PlainTextResponse(f"decrypt failed: {e}", status_code=500)

    # 明文模式
    return echostr or ""



# ---------------------------------------------------------------------
# 2) 消息回调（自动兼容明文/安全模式）
# ---------------------------------------------------------------------
@app.post("/wecom/callback", response_class=PlainTextResponse)
async def wecom_callback(request: Request):
    raw = await request.body()
    params = request.query_params
    msg_signature = params.get("msg_signature")
    timestamp = params.get("timestamp")
    nonce = params.get("nonce")

    # 解密（安全模式）或直接解析（明文）
    try:
        if crypto and all([msg_signature, timestamp, nonce]):
            decrypted_xml = crypto.decrypt_message(
                raw.decode("utf-8"), msg_signature, timestamp, nonce
            )
            msg = parse_plain_xml(decrypted_xml)
        else:
            msg = parse_plain_xml(raw.decode("utf-8"))
    except InvalidSignatureException:
        log.warning("POST callback invalid signature.")
        return PlainTextResponse("invalid signature", status_code=403)
    except Exception as e:
        log.exception("Parse/Decrypt failed: %s", e)
        # 企业微信期望 200；返回错误信息便于排查
        return JSONResponse({"ok": False, "error": f"parse/decrypt failed: {e}"}, status_code=200)

    from_user = msg.get("FromUserName") or ""
    content = msg.get("Content") or msg.get("Event") or ""

    # 调用 OpenAI
    try:
        completion = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for WeCom users."},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        reply_text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        log.exception("OpenAI call failed: %s", e)
        reply_text = f"OpenAI 调用失败：{e}"

    # 通过企业微信发送文本消息
    try:
        api_ret = await send_text(from_user, reply_text)
        log.info("WeCom send result: %s", api_ret)
    except Exception as e:
        log.exception("WeCom send failed: %s", e)
        # 仍返回 200，避免企业微信重复推送
        return JSONResponse({"ok": False, "error": f"WeCom send failed: {e}"}, status_code=200)

    # 按企业微信规范，回调接口返回 "success"
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
    }
