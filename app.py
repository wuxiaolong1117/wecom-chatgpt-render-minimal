# app.py
import os
import io
import json
import base64
import asyncio
import logging
from typing import Optional, Tuple, List

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI

# ---------- WeCom / XML ----------
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.utils import to_text
import xmltodict

# ---------- 文档/图像 ----------
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ---------- 记忆（可选 Redis） ----------
import redis

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("wecom-app")

# =========================================================
# 环境变量
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")

MODEL_PRIMARY = os.getenv("MODEL_PRIMARY", "gpt-5")
MODEL_BACKUP = os.getenv("MODEL_BACKUP", "gpt-5-mini")
MODEL_FALLBACK = os.getenv("MODEL_FALLBACK", "gpt-4o-mini")

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

LOCAL_OCR_ENABLE = os.getenv("LOCAL_OCR_ENABLE", "false").lower() in ("true", "1", "on")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "150000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
CHUNK_SUMMARY = int(os.getenv("CHUNK_SUMMARY", "800"))

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
WEB_SEARCH_ENABLE = os.getenv("WEB_SEARCH_ENABLE", "on").lower() in ("on", "true", "1")

# WeCom
WECOM_CORP_ID = os.getenv("WEWORK_CORP_ID") or os.getenv("WECOM_CORP_ID", "")
WECOM_SECRET = os.getenv("WEWORK_SECRET") or os.getenv("WECOM_SECRET", "")
WECOM_AGENT_ID = os.getenv("WEWORK_AGENT_ID") or os.getenv("WECOM_AGENT_ID", "")
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")
SAFE_MODE = os.getenv("SAFE_MODE", "true").lower() in ("true", "on", "1")

# 记忆
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory")  # memory / redis
REDIS_URL = os.getenv("REDIS_URL", "redis://host:6379/0")
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))

# 其它
SEND_MAX_LEN = int(os.getenv("SEND_MAX_LEN", "1900"))  # 企业微信文字长度安全阈值
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# =========================================================
# OpenAI Client
# =========================================================
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=(OPENAI_ORG_ID or None),  # 显式组织 ID
)

# =========================================================
# FastAPI
# =========================================================
app = FastAPI()

# =========================================================
# 记忆实现
# =========================================================
class MemoryStore:
    def __init__(self):
        self.r = None
        self.mem: dict[str, List[dict]] = {}
        if MEMORY_BACKEND == "redis":
            try:
                self.r = redis.from_url(REDIS_URL)
                self.r.ping()
                log.info("memory: redis enabled")
            except Exception as e:
                log.warning(f"memory: redis connect fail: {e}, fallback to memory")
                self.r = None

    def _key(self, user_id: str) -> str:
        return f"wecom:session:{user_id}"

    def load(self, user_id: str) -> List[dict]:
        try:
            if self.r:
                raw = self.r.get(self._key(user_id))
                if raw:
                    return json.loads(raw)
                return []
            return self.mem.get(user_id, [])
        except Exception as e:
            log.warning(f"load memory failed: {e}")
            return []

    def save(self, user_id: str, messages: List[dict]):
        try:
            if self.r:
                self.r.setex(self._key(user_id), SESSION_TTL, json.dumps(messages, ensure_ascii=False))
            else:
                self.mem[user_id] = messages
        except Exception as e:
            log.warning(f"append memory failed: {e}")

memory = MemoryStore()

# =========================================================
# WeCom API
# =========================================================
async def get_wecom_token() -> str:
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={WECOM_CORP_ID}&corpsecret={WECOM_SECRET}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url)
        data = r.json()
        return data["access_token"]

async def send_text(to_user: str, content: str):
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    # 分片发送
    chunks: List[str] = []
    text = (content or "").strip() or "（空）"
    while len(text) > SEND_MAX_LEN:
        chunks.append(text[:SEND_MAX_LEN])
        text = text[SEND_MAX_LEN:]
    chunks.append(text)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for chunk in chunks:
            payload = {
                "touser": to_user,
                "msgtype": "text",
                "agentid": int(WECOM_AGENT_ID),
                "text": {"content": chunk},
                "safe": 0,
            }
            r = await client.post(url, json=payload)
            data = r.json()
            log.warning(f"WeCom send result -> to={to_user} payload_len={len(chunk)} resp={data}")

            # 44004 空文本保护
            if data.get("errcode") == 44004:
                payload["text"]["content"] = "（模型未输出文本，建议换个问法或稍后重试）"
                await client.post(url, json=payload)

async def download_wecom_media(media_id: str) -> Tuple[bytes, str]:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url)
        ct = r.headers.get("Content-Type", "")
        return r.content, ct

# =========================================================
# OCR & PDF 解析
# =========================================================
def _local_ocr(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return (text or "").strip()
    except Exception as e:
        log.warning(f"local ocr fail: {e}")
        return ""

async def _vision_ocr(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    try:
        resp = oai.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "从图片中提取所有可见文字，按自然段返回，不要解释。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}" }},
                ],
            }],
            max_completion_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"vision ocr fail: {e}")
        return ""

async def ocr_image(image_bytes: bytes) -> str:
    if LOCAL_OCR_ENABLE:
        txt = _local_ocr(image_bytes)
        if txt:
            return txt
    return await _vision_ocr(image_bytes)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    for page in doc:
        txt = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        txt = (txt or "").strip()
        if len(txt) < 20 and LOCAL_OCR_ENABLE:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            txt = _local_ocr(img_bytes) or ""
        parts.append(txt)
    text = "\n\n".join([p for p in parts if p])
    return text[:MAX_INPUT_CHARS]

# =========================================================
# 文本分块摘要
# =========================================================
async def summarize_long_text(raw_text: str, system_prompt: str = "你是专业的文档助理。") -> str:
    text = (raw_text or "").strip()
    if not text:
        return "（未从文件中提取到可读文字）"

    # 单段
    if len(text) <= CHUNK_SIZE:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请用要点+结论总结（150-300字）：\n{text}"},
        ]
        for m in (SUMMARIZER_MODEL, MODEL_FALLBACK):
            try:
                kwargs = dict(model=m, messages=msgs)
                if m.startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = CHUNK_SUMMARY
                else:
                    kwargs["max_tokens"] = CHUNK_SUMMARY
                    kwargs["temperature"] = 1
                resp = oai.chat.completions.create(**kwargs)
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
            except Exception as e:
                log.warning(f"summarize fail {m}: {e}")
        return "（摘要失败）"

    # 多段
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    bullets: List[str] = []
    for idx, ck in enumerate(chunks, 1):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"第{idx}/{len(chunks)}段，提取3-5条要点，保留关键数字/名词：\n{ck}"},
        ]
        piece = ""
        for m in (SUMMARIZER_MODEL, MODEL_FALLBACK):
            try:
                kwargs = dict(model=m, messages=msgs)
                if m.startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = CHUNK_SUMMARY
                else:
                    kwargs["max_tokens"] = CHUNK_SUMMARY
                    kwargs["temperature"] = 1
                resp = oai.chat.completions.create(**kwargs)
                piece = (resp.choices[0].message.content or "").strip()
                if piece:
                    break
            except Exception as e:
                log.warning(f"chunk summarize fail {m}: {e}")
        if piece:
            bullets.append(f"- {piece}")

    join_text = "\n".join(bullets)[:MAX_INPUT_CHARS]
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"以下为各段要点，请合并去重并给出‘结论/建议’：\n{join_text}"},
    ]
    for m in (SUMMARIZER_MODEL, MODEL_FALLBACK):
        try:
            kwargs = dict(model=m, messages=msgs)
            if m.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = CHUNK_SUMMARY
            else:
                kwargs["max_tokens"] = CHUNK_SUMMARY
                kwargs["temperature"] = 1
            resp = oai.chat.completions.create(**kwargs)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            log.warning(f"final summarize fail {m}: {e}")
    return "（汇总失败）"

# =========================================================
# SerpAPI 搜索
# =========================================================
async def serp_search(q: str) -> str:
    if not (WEB_SEARCH_ENABLE and SERPAPI_API_KEY):
        return "（未配置 SerpAPI，或已关闭联网）"
    params = {"engine": "google", "q": q, "api_key": SERPAPI_API_KEY, "num": 5, "hl": "zh-cn"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get("https://serpapi.com/search", params=params)
        data = r.json()
    links = []
    for item in (data.get("organic_results") or [])[:5]:
        title = item.get("title") or item.get("displayed_link") or "结果"
        link = item.get("link") or ""
        links.append(f"[{title}]\n{link}")
    return "🔎 已联网搜索（serpapi）：\n" + ("\n\n".join(links) if links else "（没有搜到可用结果）")

# =========================================================
# Chat（含兜底链）
# =========================================================
async def chat_with_models(messages: List[dict], max_out: int = 512) -> str:
    for m in (MODEL_PRIMARY, MODEL_BACKUP, MODEL_FALLBACK):
        try:
            kwargs = dict(model=m, messages=messages)
            if m.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_out
            else:
                kwargs["max_tokens"] = max_out
                kwargs["temperature"] = 1
            resp = oai.chat.completions.create(**kwargs)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
            log.warning(f"model {m} returned empty content")
        except Exception as e:
            log.warning(f"model {m} call fail: {e}")
    return "（模型临时没有返回内容，建议换个说法或稍后重试）"

# =========================================================
# 工具：清洗/重建 XML 以处理 ExpatError
# =========================================================
def clean_xml_payload(raw: str) -> str:
    if not raw:
        return ""
    s = raw.lstrip("\ufeff \t\r\n")
    i = s.find("<")
    if i > 0:
        s = s[i:]
    # 只保留到最后一个 '>'（去尾部脏字节）
    j = s.rfind(">")
    if j >= 0:
        s = s[:j+1]
    return s

# =========================================================
# 路由
# =========================================================
@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "service": "WeCom + ChatGPT",
        "model": f"{MODEL_PRIMARY},{MODEL_BACKUP},{MODEL_FALLBACK}",
        "memory": MEMORY_BACKEND,
        "web_search": "on/serpapi" if WEB_SEARCH_ENABLE else "off",
        "pdf_image": "enabled",
        "local_ocr": LOCAL_OCR_ENABLE,
    })

# 企业微信 URL 验证（GET）
@app.get("/wecom/callback")
async def wecom_verify(
    msg_signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query("")
):
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORP_ID)
        echo = crypto.decrypt_message(msg_signature, timestamp, nonce, echostr)
        echo = to_text(xmltodict.parse(to_text(echo))["xml"]["EchoStr"])
        return PlainTextResponse(echo)
    except Exception as e:
        log.error(f"URL verify decrypt failed: {e}")
        return PlainTextResponse("error", status_code=400)

# 企业微信消息回调（POST）
@app.post("/wecom/callback")
async def wecom_callback(
    request: Request,
    msg_signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query("")
):
    body_bytes = await request.body()
    xml_text = body_bytes.decode("utf-8", errors="ignore") if SAFE_MODE else body_bytes  # safe-mode 走 str

    # 先尝试解密
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORP_ID)
        decrypted_xml = crypto.decrypt_message(msg_signature, timestamp, nonce, xml_text)
        msg = xmltodict.parse(to_text(decrypted_xml))["xml"]
    except Exception as e:
        # 解密失败：尝试清洗后的明文 XML（企业微信偶发明文/前后带脏字节）
        log.error(f"ERROR:wecom-app:decrypt fail (safe-mode): {e}")
        try:
            cleaned = clean_xml_payload(xml_text)
            log.warning(f"safe-mode: using cleaned payload; head='{cleaned[:120]}'")
            msg = xmltodict.parse(to_text(cleaned))["xml"]
        except Exception as e2:
            log.error(f"decrypt retry fail: {e2}")
            return PlainTextResponse("success")  # 一律返回 success，避免企业微信重试风暴

    to_user = msg.get("ToUserName", "")
    from_user = msg.get("FromUserName", "")
    msg_type = (msg.get("MsgType") or "").lower()

    # ---------- 文本 ----------
    if msg_type == "text":
        content = (msg.get("Content") or "").strip()

        if content == "/ping":
            info = (
                f"当前活跃模型：{MODEL_PRIMARY}\n"
                f"候选列表：{MODEL_PRIMARY}, {MODEL_BACKUP}, {MODEL_FALLBACK}\n"
                f"组织ID：{OPENAI_ORG_ID or '-'}\n"
                f"记忆：{MEMORY_BACKEND}\n"
                f"联网搜索：{'on/serpapi' if (WEB_SEARCH_ENABLE and SERPAPI_API_KEY) else 'off'}\n"
                f"PDF/图片解析：已启用（LOCAL_OCR={'on' if LOCAL_OCR_ENABLE else 'off'}）"
            )
            await send_text(from_user, info)
            return PlainTextResponse("success")

        if content.startswith("/web "):
            q = content[5:].strip()
            result = await serp_search(q)
            await send_text(from_user, result)
            return PlainTextResponse("success")

        # 普通对话（带记忆）
        history = memory.load(from_user)
        history.append({"role": "user", "content": content})
        reply = await chat_with_models(history, max_out=512)
        history.append({"role": "assistant", "content": reply})
        memory.save(from_user, history)
        await send_text(from_user, reply)
        return PlainTextResponse("success")

    # ---------- 图片 ----------
    if msg_type == "image":
        media_id = msg.get("MediaId")
        await send_text(from_user, "已收到图片，正在识别…")
        try:
            file_bytes, ct = await download_wecom_media(media_id)
            ocr_txt = await ocr_image(file_bytes)
            if not ocr_txt:
                await send_text(from_user, "（未识别到文字，或图片质量较低）")
                return PlainTextResponse("success")
            summary = await summarize_long_text(ocr_txt, "你是图像文字识别与摘要助手。")
            await send_text(from_user, f"【图片要点摘要】\n{summary}")
        except Exception as e:
            log.warning(f"image handle fail: {e}")
            await send_text(from_user, "（图片解析失败）")
        return PlainTextResponse("success")

    # ---------- 文件（PDF / 图片等） ----------
    if msg_type == "file":
        fname = (msg.get("FileName") or "").strip().lower()
        media_id = msg.get("MediaId")
        await send_text(from_user, f"已收到文件：{fname}，正在处理…")
        try:
            file_bytes, ct = await download_wecom_media(media_id)
            summary = ""
            if fname.endswith(".pdf") or "pdf" in ct:
                text = extract_text_from_pdf(file_bytes)
                summary = await summarize_long_text(text, "你是专业文档助理。")
            elif any(fname.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")) or ("image/" in ct):
                ocr_txt = await ocr_image(file_bytes)
                summary = await summarize_long_text(ocr_txt, "你是图像文字识别与摘要助手。")
            else:
                summary = "（暂不支持该格式，仅支持 PDF / PNG / JPG / WEBP）"

            await send_text(from_user, f"【文件摘要】\n{summary}")
        except Exception as e:
            log.warning(f"file handle fail: {e}")
            await send_text(from_user, "（文件解析失败）")
        return PlainTextResponse("success")

    # 其它类型
    await send_text(from_user, f"（暂未支持的消息类型：{msg_type}）")
    return PlainTextResponse("success")
