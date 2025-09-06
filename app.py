import os
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from lxml import etree
from typing import Optional
from openai import OpenAI

# NEW: 企业微信安全模式加解密
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

load_dotenv()
app = FastAPI(title="WeCom + ChatGPT (plaintext & safe-mode)")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
oai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --- WeCom ---
CORP_ID = os.getenv("WEWORK_CORP_ID", "")
AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
SECRET  = os.getenv("WEWORK_SECRET", "")

# NEW: 安全模式密钥（若未配置则自动按明文处理）
WECOM_TOKEN   = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")
crypto: Optional[WeChatCrypto] = None
if WECOM_TOKEN and WECOM_AES_KEY and CORP_ID:
    crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, CORP_ID)

# cache access_token
TOKEN_CACHE = {"value": None, "exp": 0}
async def get_wecom_token() -> str:
    now = int(time.time())
    if TOKEN_CACHE["value"] and TOKEN_CACHE["exp"] > now + 30:
        return TOKEN_CACHE["value"]
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORP_ID}&corpsecret={SECRET}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
    access_token = data.get("access_token", "")
    TOKEN_CACHE["value"] = access_token
    TOKEN_CACHE["exp"] = now + 7000
    return access_token

async def send_text(to_user: str, content: str) -> dict:
    token = await get_wecom_token()
    send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(AGENT_ID),
        "text": {"content": content[:2048]},
        "safe": 0,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(send_url, json=payload)
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

# --- URL 验证（自动兼容 明文 / 安全模式）---
@app.get("/wecom/callback", response_class=PlainTextResponse)
async def wecom_verify(request: Request, echostr: Optional[str] = None,
                       msg_signature: Optional[str] = None,
                       timestamp: Optional[str] = None,
                       nonce: Optional[str] = None):
    # 安全模式：企业微信会传 msg_signature/timestamp/nonce/echostr（加密）
    if crypto and all([msg_signature, timestamp, nonce, echostr]):
        try:
            # wechatpy 的 verify_url：校验签名并解密出明文
            echo = crypto.decrypt_message(echostr, msg_signature, timestamp, nonce)
            return PlainTextResponse(echo)
        except InvalidSignatureException:
            return PlainTextResponse("invalid signature", status_code=403)
        except Exception as e:
            return PlainTextResponse(f"decrypt failed: {e}", status_code=500)
    # 明文模式：直接回显 echostr
    return echostr or ""

# --- 消息回调（自动兼容 明文 / 安全模式）---
@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    raw = await request.body()
    params = request.query_params
    msg_signature = params.get("msg_signature")
    timestamp = params.get("timestamp")
    nonce = params.get("nonce")

    # 解密（安全模式）或直接解析（明文）
    try:
        if crypto and all([msg_signature, timestamp, nonce]):
            # raw 是加密的 XML 文本
            decrypted_xml = crypto.decrypt_message(raw.decode("utf-8"), msg_signature, timestamp, nonce)
            msg = parse_plain_xml(decrypted_xml)
        else:
            msg = parse_plain_xml(raw.decode("utf-8"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"parse/decrypt failed: {e}"}, status_code=400)

    from_user = msg.get("FromUserName")
    content = msg.get("Content") or msg.get("Event") or ""

    # 调 OpenAI
    try:
        chat = oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for WeCom users."},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        reply_text = chat.choices[0].message.content.strip()
    except Exception as e:
        reply_text = f"OpenAI 调用失败：{e}"

    # 通过企业微信消息接口回发
    try:
        await send_text(from_user, reply_text)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"WeCom send failed: {e}", "reply_preview": reply_text}, status_code=200)

    # 回调响应：企业微信要求返回 "success"
    # （安全模式理论上可加密响应，但这里不需要在回调中“被动回复消息”，只要返回 success 即可）
    return PlainTextResponse("success")

@app.get("/")
async def root():
    return {"status": "ok", "mode": "safe" if crypto else "plain", "service": "WeCom + ChatGPT"}
