import os
import time
import json
import base64
import logging
import threading
from collections import deque
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from lxml import etree
from openai import OpenAI

from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# ------- 可选：PDF 解析（必须在 requirements.txt 增加 PyPDF2>=3.0.1） -------
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ------- 可选：本地 OCR（不建议在 Render 上启用，需系统安装 tesseract） -------
USE_LOCAL_OCR = False
try:
    if USE_LOCAL_OCR:
        from PIL import Image
        import pytesseract
        HAS_OCR = True
    else:
        HAS_OCR = False
except Exception:
    HAS_OCR = False

# ---------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wecom-app")

app = FastAPI(title="WeCom + ChatGPT (files & images supported)")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # 视觉模型也支持，如 gpt-4o-mini/gpt-5
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID   = os.getenv("OPENAI_ORG_ID", "")   # ← 新增

# 新写法：显式带上 organization
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,
)

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

async def send_long_text(to_user: str, content: str, part_len: int = 1800, max_parts: int = 10):
    """
    企业微信文本消息有长度限制，这里分片发送（每片 ~1800 字，最多 10 段，可按需调整）
    """
    if not content:
        return
    parts = [content[i:i+part_len] for i in range(0, len(content), part_len)]
    for idx, p in enumerate(parts[:max_parts], start=1):
        suffix = f"\n\n({idx}/{len(parts)})" if len(parts) > 1 else ""
        await send_text(to_user, p + suffix)
    if len(parts) > max_parts:
        await send_text(to_user, f"……内容较长，已发送前 {max_parts} 段，共 {len(parts)} 段。")

def parse_plain_xml(raw_xml: str) -> dict:
    root = etree.fromstring(raw_xml.encode("utf-8"))
    def get(tag: str) -> Optional[str]:
        el = root.find(tag)
        return el.text if el is not None else None
    # 扩展：取 MediaId / PicUrl / FileName
    return {
        "ToUserName": get("ToUserName"),
        "FromUserName": get("FromUserName"),
        "CreateTime": get("CreateTime"),
        "MsgType": get("MsgType"),
        "Content": get("Content"),
        "MediaId": get("MediaId"),
        "PicUrl": get("PicUrl"),
        "FileName": get("FileName"),
        "MsgId": get("MsgId"),
        "AgentID": get("AgentID"),
        "Event": get("Event"),
    }

# ---------------------------------------------------------------------
# Memory：内存 + Redis 双实现（与之前一致）
# ---------------------------------------------------------------------
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "false").lower() == "true"
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))
MEMORY_TTL = int(os.getenv("MEMORY_TTL_SECONDS", "86400"))
REDIS_URL = os.getenv("REDIS_URL", "")

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
# 媒体下载 & 类型判断
# ---------------------------------------------------------------------
def guess_extension(content_type: str, fallback: str = "") -> str:
    if not content_type:
        return fallback
    ct = content_type.lower()
    if "pdf" in ct: return ".pdf"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "gif" in ct: return ".gif"
    if "webp" in ct: return ".webp"
    return fallback

async def download_media_to_tmp(media_id: str) -> Tuple[str, str]:
    """
    返回 (file_path, content_type)
    """
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(url)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        # 尝试从 Content-Disposition 获取文件名
        filename = "wecom_file"
        cd = r.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"; ')
        ext = os.path.splitext(filename)[-1].lower()
        if not ext:
            ext = guess_extension(content_type, "")
            filename = filename + ext if ext else filename
        tmp_path = f"/tmp/{int(time.time()*1000)}_{filename}"
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        return tmp_path, content_type

# ---------------------------------------------------------------------
# PDF / 图片 处理
# ---------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    if not HAS_PDF:
        return "服务器暂未安装 PDF 解析库（PyPDF2）。请先在 requirements.txt 中加入：PyPDF2>=3.0.1"
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()

def encode_image_to_data_url(path: str, content_type: Optional[str] = None) -> str:
    ctype = content_type or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{ctype};base64,{b64}"

async def analyze_image_with_openai(path: str, content_type: Optional[str] = None, task: str = "请提取图片中的文字，并简要说明图像的要点。"):
    data_url = encode_image_to_data_url(path, content_type)
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text", "text": task},
                     {"type": "image_url", "image_url": {"url": data_url}},
                 ]}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI 视觉解析失败：{e}"

def ocr_image_locally(path: str) -> str:
    if not HAS_OCR:
        return "本地 OCR 未启用（Render 上通常不推荐）。建议改用 OpenAI 视觉模型。"
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text.strip()
    except Exception as e:
        return f"OCR 失败：{e}"

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
# 消息回调（含文本/图片/文件）
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
    msg_type = (msg.get("MsgType") or "").lower()
    content = msg.get("Content") or ""
    media_id = msg.get("MediaId")
    file_name = msg.get("FileName") or ""

    # ========= 1) 图片消息 =========
    if msg_type == "image" and media_id:
        path, ctype = await download_media_to_tmp(media_id)
        if USE_LOCAL_OCR and HAS_OCR:
            text = ocr_image_locally(path)
        else:
            text = await analyze_image_with_openai(path, ctype)
        await send_long_text(from_user, f"图片解析结果：\n{text}")
        return PlainTextResponse("success")

    # ========= 2) 文件消息（PDF 优先处理） =========
    if msg_type == "file" and media_id:
        path, ctype = await download_media_to_tmp(media_id)
        ext = os.path.splitext(file_name or path)[-1].lower()
        if ext == ".pdf" or "pdf" in (ctype or "").lower():
            pdf_text = extract_text_from_pdf(path)
            if not pdf_text.strip():
                await send_text(from_user, "该 PDF 提取不到可搜索文本（可能是扫描版）。可尝试将 PDF 每页转图片后走图片解析（OpenAI 视觉）或启用本地 OCR。")
            else:
                # 直接“尽量全量”回传：分片多条发
                await send_long_text(from_user, f"《{file_name or 'PDF 文件'}》全文提取：\n{pdf_text}", part_len=1800, max_parts=20)
            return PlainTextResponse("success")
        else:
            await send_text(from_user, f"已收到文件：{file_name or os.path.basename(path)}（类型：{ctype or '未知'}）。当前仅对 PDF 做全文提取，其它类型可先转为 PDF 或图片。")
            return PlainTextResponse("success")

    # ========= 3) 文本 / 其它事件：走原有多轮对话 =========
    # 构造上下文
    base_system = {"role": "system", "content": "You are a helpful assistant for WeCom users."}
    messages = [base_system]

    # 记忆：Redis 优先，其次内存
    if ENABLE_MEMORY and from_user:
        try:
            if REDIS_MEMORY:
                history = await REDIS_MEMORY.get(from_user)
                messages.extend(history)
            elif MEMORY:
                for role, text0 in MEMORY.get(from_user):
                    messages.append({"role": role, "content": text0})
        except Exception as e:
            log.warning("load memory failed: %s", e)

    messages.append({"role": "user", "content": content or (msg.get("Event") or "")})

    # 调用 OpenAI
    try:
        completion = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
           max_completion_tokens=300,
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
        "pdf_support": HAS_PDF,
        "local_ocr": HAS_OCR and USE_LOCAL_OCR,
    }
