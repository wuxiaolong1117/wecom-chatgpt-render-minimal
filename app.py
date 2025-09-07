import os
import io
import json
import base64
import time
import math
import asyncio
import logging
from typing import Optional, Tuple, List

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------- WeCom 依赖 ----------
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.utils import to_text
import xmltodict

# ---------- 文档/图像处理 ----------
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
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")  # 显式组织
MODEL_PRIMARY = os.getenv("MODEL_PRIMARY", "gpt-5")
MODEL_BACKUP = os.getenv("MODEL_BACKUP", "gpt-5-mini")
MODEL_FALLBACK = os.getenv("MODEL_FALLBACK", "gpt-4o-mini")

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

LOCAL_OCR_ENABLE = os.getenv("LOCAL_OCR_ENABLE", "false").lower() == "true"
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
SEND_MAX_LEN = int(os.getenv("SEND_MAX_LEN", "1900"))  # 单条消息最大字数，避免企业微信超限
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# =========================================================
# OpenAI Client
# =========================================================
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=(OPENAI_ORG_ID or None),
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
        self.mem = {}
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

    # 分片发送，避免超限
    chunks = []
    text = content.strip() or "（空）"
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

            # 44004 空文本保护（极少数情况下模型回空）
            if data.get("errcode") == 44004:
                payload["text"]["content"] = "（模型未输出文本，建议换个问法或稍后重试）"
                r = await client.post(url, json=payload)
                log.warning("WeCom empty content retry once.")

async def download_wecom_media(media_id: str) -> Tuple[bytes, str]:
    """
    返回 (bytes, content_type)
    """
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
        return text.strip()
    except Exception as e:
        log.warning(f"local ocr fail: {e}")
        return ""

async def _vision_ocr(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    # 使用 chat.completions Vision 输入
    try:
        resp = oai.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "从图片中**提取所有可见文字**，保持自然段顺序，不要额外解释。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            max_completion_tokens=1024,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt
    except Exception as e:
        log.warning(f"vision ocr fail: {e}")
        return ""

async def ocr_image(image_bytes: bytes) -> str:
    if LOCAL_OCR_ENABLE:
        text = _local_ocr(image_bytes)
        if text:
            return text
        # 本地识别失败则回退云端
    return await _vision_ocr(image_bytes)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    先用 PyMuPDF 提取文本；若页面文本稀少，则转图后本地 OCR（需要 tesseract），最后拼接。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i, page in enumerate(doc):
        txt = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        txt = (txt or "").strip()
        if len(txt) < 20 and LOCAL_OCR_ENABLE:
            # 回退为图片 OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 提高分辨率
            img_bytes = pix.tobytes("png")
            txt = _local_ocr(img_bytes)
        parts.append(txt)
    text = "\n\n".join([p for p in parts if p])
    return text[:MAX_INPUT_CHARS]

# =========================================================
# 分块摘要
# =========================================================
async def summarize_long_text(raw_text: str, system_prompt: str = "你是专业的文档助理。") -> str:
    text = (raw_text or "").strip()
    if not text:
        return "（未从文件中提取到可读文字）"

    if len(text) <= CHUNK_SIZE:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请用要点+结论总结这段文字（150-300字）：\n{text}"}]
        for model in (SUMMARIZER_MODEL, MODEL_FALLBACK):
            try:
                resp = oai.chat.completions.create(
                    model=model,
                    messages=msgs,
                    max_completion_tokens=CHUNK_SUMMARY,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
            except Exception as e:
                log.warning(f"summarize single fail with {model}: {e}")
        return "（摘要失败）"

    # 分块
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE

    bullets = []
    for idx, ck in enumerate(chunks, 1):
        prompt = f"第{idx}/{len(chunks)}段，请提取3-5条要点，保留关键数字/专有名词：\n{ck}"
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
        summary_piece = ""
        for model in (SUMMARIZER_MODEL, MODEL_FALLBACK):
            try:
                resp = oai.chat.completions.create(
                    model=model,
                    messages=msgs,
                    max_completion_tokens=CHUNK_SUMMARY,
                )
                summary_piece = (resp.choices[0].message.content or "").strip()
                if summary_piece:
                    break
            except Exception as e:
                log.warning(f"chunk summarize fail with {model}: {e}")
        if summary_piece:
            bullets.append(f"- {summary_piece}")

    # 汇总
    join_text = "\n".join(bullets)[:MAX_INPUT_CHARS]
    final_prompt = f"以下是分段要点，请合并去重并给出‘结论/建议’：\n{join_text}"
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}]
    for model in (SUMMARIZER_MODEL, MODEL_FALLBACK):
        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=msgs,
                max_completion_tokens=CHUNK_SUMMARY,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            log.warning(f"final summarize fail with {model}: {e}")
    return "（汇总失败）"

# =========================================================
# 搜索（SerpAPI）
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
    if not links:
        return "（没有搜到可用结果）"
    return "🔎 已联网搜索（serpapi）：\n" + "\n\n".join(links)

# =========================================================
# OpenAI 对话（含兜底）
# =========================================================
async def chat_with_models(messages: List[dict], max_out: int = 512) -> str:
    chain = (MODEL_PRIMARY, MODEL_BACKUP, MODEL_FALLBACK)
    for m in chain:
        try:
            kwargs = dict(model=m, messages=messages)
            # gpt-5 系列不支持 temperature/max_tokens，需要用 max_completion_tokens
            if m.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_out
            else:
                kwargs["max_tokens"] = max_out
                kwargs["temperature"] = 1
            resp = oai.chat.completions.create(**kwargs)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
            log.warning(f"primary model {m} failed: empty content")
        except Exception as e:
            log.warning(f"model {m} call fail: {e}")
    return "（模型临时没有返回内容，建议换个说法或稍后重试）"

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
    xml_text = (await request.body()).decode("utf-8", errors="ignore") if SAFE_MODE else (await request.body())

    # 先尝试解密
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORP_ID)
        decrypted_xml = crypto.decrypt_message(msg_signature, timestamp, nonce, xml_text)
        msg = xmltodict.parse(to_text(decrypted_xml))["xml"]
    except Exception as e:
        # 解析异常，按明文 xml 尝试一次（企业微信偶发回明文）
        log.error(f"ERROR:wecom-app:decrypt fail (safe-mode): {e}")
        try:
            log.warning(f"safe-mode: using payload (xml); head='{xml_text[:120]}'")
            msg = xmltodict.parse(to_text(xml_text))["xml"]
        except Exception as e2:
            log.error(f"decrypt retry fail: {e2}")
            return PlainTextResponse("success")  # 返回 success 不重试

    to_user = msg.get("ToUserName", "")
    from_user = msg.get("FromUserName", "")
    msg_type = (msg.get("MsgType") or "").lower()

    # ---------- 文本 ----------
    if msg_type == "text":
        content = (msg.get("Content") or "").strip()

        # 指令：/ping
        if content == "/ping":
            info = (
                f"当前活跃模型：{MODEL_PRIMARY}\n"
                f"候选列表：{MODEL_PRIMARY}, {MODEL_BACKUP}, {MODEL_FALLBACK}\n"
                f"组织ID： {OPENAI_ORG_ID or '-'}\n"
                f"记忆：{MEMORY_BACKEND}\n"
                f"联网搜索：{'on/serpapi' if (WEB_SEARCH_ENABLE and SERPAPI_API_KEY) else 'off'}\n"
                f"PDF/图片解析：已启用（LOCAL_OCR={'on' if LOCAL_OCR_ENABLE else 'off'}）"
            )
            await send_text(from_user, info)
            return PlainTextResponse("success")

        # 指令：/web xxx
        if content.startswith("/web "):
            q = content[5:].strip()
            result = await serp_search(q)
            await send_text(from_user, result)
            return PlainTextResponse("success")

        # 普通对话：带记忆
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

    # ---------- 文件（含 PDF、图片当附件等） ----------
    if msg_type == "file":
        fname = (msg.get("FileName") or "").strip()
        media_id = msg.get("MediaId")
        await send_text(from_user, f"已收到文件：{fname}，正在处理…")
        try:
            file_bytes, ct = await download_wecom_media(media_id)
            name_low = fname.lower()
            summary = ""

            if name_low.endswith(".pdf") or "pdf" in ct:
                text = extract_text_from_pdf(file_bytes)
                summary = await summarize_long_text(text, "你是专业文档助理。")
            elif any(name_low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")) or ("image/" in ct):
                ocr_txt = await ocr_image(file_bytes)
                summary = await summarize_long_text(ocr_txt, "你是图像文字识别与摘要助手。")
            else:
                summary = "（暂不支持该格式，仅支持 PDF / PNG / JPG / WEBP）"

            await send_text(from_user, f"【文件摘要】\n{summary}")
        except Exception as e:
            log.warning(f"file handle fail: {e}")
            await send_text(from_user, "（文件解析失败）")
        return PlainTextResponse("success")

    # 其它类型：直接忽略
    await send_text(from_user, f"（暂未支持的消息类型：{msg_type}）")
    return PlainTextResponse("success")
