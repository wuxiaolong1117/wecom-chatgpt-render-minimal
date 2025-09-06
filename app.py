
import os
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from lxml import etree
from typing import Optional
from openai import OpenAI

load_dotenv()

app = FastAPI(title="WeCom + ChatGPT minimal")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
oai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --- WeCom ---
CORP_ID = os.getenv("WEWORK_CORP_ID", "")
AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
SECRET = os.getenv("WEWORK_SECRET", "")

# cache access_token in memory
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
    # WeCom token is valid for 7200s typically
    TOKEN_CACHE["value"] = access_token
    TOKEN_CACHE["exp"] = now + 7000
    return access_token

def parse_xml(raw: bytes) -> dict:
    """
    Minimal XML parser for plaintext mode.
    """
    root = etree.fromstring(raw)
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
    }

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

@app.get("/wecom/callback", response_class=PlainTextResponse)
async def wecom_verify(echostr: Optional[str] = None):
    # URL verification for plaintext mode: just echo back 'echostr'
    return echostr or ""

@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    raw = await request.body()
    msg = parse_xml(raw)
    if not msg.get("MsgType"):
        return {"ok": True}

    from_user = msg.get("FromUserName")
    content = msg.get("Content") or ""

    # Call OpenAI with a single-turn prompt
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

    # Send reply back to WeCom user
    try:
        await send_text(from_user, reply_text)
    except Exception as e:
        # fall back: at least log in HTTP response
        return {"ok": False, "error": f"WeCom send failed: {e}", "reply_preview": reply_text}

    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "ok", "service": "WeCom + ChatGPT minimal"}
