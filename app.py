# app.py —— WeCom ↔ OpenAI Chat 完整版
# 功能：文本对话、/web 联网提示、PDF 全文解析（分块总结）、图片多模态识图
# 兼容：企业微信安全模式 echostr 校验 & 消息解密（不同 wechatpy 版本）

import os
import io
import base64
import logging
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.utils import to_text

# ============ 日志 ============
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("wecom-app")

# ============ 环境变量 ============
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-5")  # 主模型
OPENAI_FALLBACKS = [m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-5-mini,gpt-4o-mini").split(",") if m.strip()]
OPENAI_ORG_ID    = os.getenv("OPENAI_ORG_ID", "")

WEWORK_CORP_ID   = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID  = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET    = os.getenv("WEWORK_SECRET", "")

WECOM_TOKEN      = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY    = os.getenv("WECOM_AES_KEY", "")

# /web 开关：仅提示联网（不做真实抓取）
WEB_SEARCH_HINT  = os.getenv("WEB_SEARCH_HINT", "1") == "1"

# 媒体解析相关
ALLOW_MEDIA      = os.getenv("ALLOW_MEDIA", "1") == "1"
MAX_MEDIA_BYTES  = int(os.getenv("MAX_MEDIA_BYTES", "15728640"))  # 15 MB 保护
LOCAL_OCR        = os.getenv("LOCAL_OCR", "0") == "1"             # 默认关闭，本机通常没装 tesseract

# ============ OpenAI ============
import openai
openai.base_url = OPENAI_BASE_URL
openai.api_key  = OPENAI_API_KEY
if OPENAI_ORG_ID:
    openai.organization = OPENAI_ORG_ID

# ============ HTTP Client ============
async_client = httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10))

# ============ WeCom 发送 ============
async def get_wecom_token() -> str:
    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    r = await async_client.get(url)
    r.raise_for_status()
    data = r.json()
    if data.get("errcode") != 0:
        raise RuntimeError(f"gettoken error: {data}")
    return data["access_token"]

async def send_text(to_user: str, content: str) -> Dict[str, Any]:
    if not content or not content.strip():
        content = "（模型未输出文本，建议换个问法或稍后再试）"
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "agentid": int(WEWORK_AGENT_ID),
        "msgtype": "text",
        "text": {"content": content[:2048]},
        "safe": 0,
    }
    last_err = None
    for _ in range(3):
        try:
            r = await async_client.post(url, json=payload)
            data = r.json()
            log.warning("WeCom send result -> to=%s payload_len=%s resp=%s",
                        to_user, len(payload['text']['content']), data)
            if data.get("errcode") == 0:
                return data
            last_err = data
        except Exception as e:
            last_err = {"exception": str(e)}
    log.error("WeCom send failed: %s", last_err)
    raise RuntimeError(f"WeCom send err: {last_err}")

# ============ WeCom 解密（兼容） ============
crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)

def compat_decrypt_echostr(echostr: str, msg_signature: str, timestamp: str, nonce: str) -> str:
    try:  # 有些版本有 decrypt
        return to_text(crypto.decrypt(echostr, msg_signature, timestamp, nonce))
    except AttributeError:
        # 构造最小 XML 通过 decrypt_message
        fake_xml = (
            "<xml>"
            f"<ToUserName><![CDATA[{WEWORK_CORP_ID}]]></ToUserName>"
            f"<Encrypt><![CDATA[{echostr}]]></Encrypt>"
            "</xml>"
        )
        plain = crypto.decrypt_message(fake_xml, msg_signature, timestamp, nonce)
        return to_text(plain)

# ============ OpenAI 工具 ============
async def call_openai_chat(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    use_model = model or OPENAI_MODEL
    families = [use_model] + [m for m in OPENAI_FALLBACKS if m != use_model]
    last_err = None
    for m in families:
        try:
            req: Dict[str, Any] = {"model": m, "messages": messages}
            if max_tokens is not None:
                req["max_completion_tokens"] = max_tokens
            if temperature is not None and not m.startswith("gpt-5"):
                req["temperature"] = temperature

            resp = await async_client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    **({"OpenAI-Organization": OPENAI_ORG_ID} if OPENAI_ORG_ID else {}),
                },
                json=req,
            )
            if resp.status_code != 200:
                last_err = {"status": resp.status_code, "body": resp.text}
                raise RuntimeError(f"HTTP {resp.status_code}")

            data = resp.json()
            # content 可能是字符串或多模态（列表）生成的字符串，这里按普通文本取
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            ).strip()
            if text and not (text.startswith("（模型未输出文本") and len(text) < 60):
                return text
            raise RuntimeError("empty content")
        except Exception as e:
            log.warning("wecom-app:model %s fail: %s", m, str(e))
            last_err = str(e)
            continue
    log.error("OpenAI call failed after fallbacks: %s", last_err)
    return "（模型未输出文本，建议换个问法或稍后再试）"

# ============ PDF 提取 & 总结 ============
def _is_pdf(data: bytes, ctype: str, filename: str) -> bool:
    fn = (filename or "").lower()
    return ctype == "application/pdf" or fn.endswith(".pdf") or data[:4] == b"%PDF"

def _extract_pdf_text(data: bytes, max_chars: int = 300_000) -> str:
    # 用 pypdf，依赖：pypdf
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        log.warning("pypdf open fail: %s", e)
        return ""
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            parts.append(f"\n\n# Page {i+1}\n{txt.strip()}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "".join(parts)[:max_chars]

def _chunk_text(s: str, chunk_chars: int = 6000) -> List[str]:
    s = s.replace("\x00", "")
    chunks = []
    pos = 0
    n = len(s)
    while pos < n:
        chunk = s[pos:pos+chunk_chars]
        chunks.append(chunk)
        pos += chunk_chars
    return chunks

async def summarize_long_text(text: str, file_title: str = "") -> str:
    if not text.strip():
        return "这份 PDF 没有可直接提取的文本（可能是扫描件）。请提供可复制文本的 PDF，或开启本地 OCR 后再试。"
    chunks = _chunk_text(text, 6000)
    notes: List[str] = []
    sys_prompt = "你是专业的文档助理。请将用户提供的文档片段进行要点笔记化梳理。简洁、分条、保留关键事实/数字/结论。"
    for idx, ch in enumerate(chunks, 1):
        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"《{file_title or 'PDF'}》第 {idx}/{len(chunks)} 段内容：\n{ch}\n\n请输出本段要点（短句分条）。"}
        ]
        part = await call_openai_chat(msg, max_tokens=700, temperature=None)
        notes.append(f"## 第 {idx} 段要点\n{part}")
        # 适度控制请求量
        if idx >= 30:  # 最多 30 段，避免极端长文档烧费
            notes.append("（超长文档，已截断到前 30 段进行总结）")
            break

    # 合并总结
    merge_msg = [
        {"role": "system", "content": "你是专业的文档汇总助手。请把多段笔记合并为一份结构化摘要，包含：概要、关键要点、数据/结论、可能的风险/限制。"},
        {"role": "user", "content": "\n\n".join(notes)}
    ]
    final = await call_openai_chat(merge_msg, max_tokens=900, temperature=None)
    return final

# ============ 图片识别（多模态） ============
def _is_image(ctype: str, filename: str) -> bool:
    fn = (filename or "").lower()
    return ctype.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))

async def vision_analyze_image(image_bytes: bytes, mime: str) -> str:
    # 用 data:URL 直接传给 gpt-4o-mini，不需要公网 URL
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "请识别并简要说明图片关键信息，若有文字也尽可能读出。"},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
    }]
    return await call_openai_chat(msgs, model="gpt-4o-mini", max_tokens=700, temperature=None)

# ============ WeCom 媒体下载 ============
async def download_media(media_id: str) -> Tuple[bytes, str, str]:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    r = await async_client.get(url)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    disp  = r.headers.get("Content-disposition", "") or r.headers.get("Content-Disposition", "")
    filename = ""
    if "filename=" in disp:
        filename = disp.split("filename=")[-1].strip('"\'')
    data = r.content
    if len(data) > MAX_MEDIA_BYTES:
        raise RuntimeError(f"file too large: {len(data)} bytes")
    return data, ctype, filename

# ============ FastAPI ============
app = FastAPI(title="WeCom + ChatGPT")

@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "status": "ok",
        "service": "WeCom + ChatGPT",
        "model": OPENAI_MODEL + ("," + ",".join(OPENAI_FALLBACKS) if OPENAI_FALLBACKS else ""),
        "org": OPENAI_ORG_ID or "",
        "web_hint": WEB_SEARCH_HINT,
        "media": ALLOW_MEDIA,
        "pdf_support": True,
        "local_ocr": LOCAL_OCR,
    }

# ============ 回调 ============
@app.api_route("/wecom/callback", methods=["GET", "POST"])
async def wecom_callback(request: Request):
    msg_signature = request.query_params.get("msg_signature", "")
    timestamp = request.query_params.get("timestamp", "")
    nonce = request.query_params.get("nonce", "")

    # ---- GET：URL 验证 ----
    if request.method == "GET":
        echostr = request.query_params.get("echostr", "")
        if not echostr:
            return PlainTextResponse("missing echostr", status_code=400)
        if msg_signature:
            try:
                echo = compat_decrypt_echostr(echostr, msg_signature, timestamp, nonce)
                return PlainTextResponse(echo)
            except Exception as e:
                log.error("decrypt fail: %s", e)
                return PlainTextResponse("bad request", status_code=400)
        else:
            return PlainTextResponse(echostr)

    # ---- POST：收消息 ----
    raw = (await request.body()).decode("utf-8")
    xml_text = raw
    if msg_signature:
        try:
            xml_text = crypto.decrypt_message(xml_text, msg_signature, timestamp, nonce)
        except Exception as e:
            log.error("decrypt fail: %s", e)
            return PlainTextResponse("bad request", status_code=400)

    # 极简解析（生产建议用 xmltodict）
    def _extract(tag: str, text: str) -> str:
        # 先找 <![CDATA[]]>
        c_tag = f"<{tag}><![CDATA["
        s = text.find(c_tag)
        if s >= 0:
            e = text.find("]]>", s)
            return text[s+len(c_tag):e] if e > s else ""
        s = text.find(f"<{tag}>")
        if s >= 0:
            e = text.find(f"</{tag}>", s)
            return text[s+len(tag)+2:e] if e > s else ""
        return ""

    from_user = _extract("FromUserName", xml_text)
    msg_type  = _extract("MsgType", xml_text).lower()
    content   = _extract("Content", xml_text).strip()
    media_id  = _extract("MediaId", xml_text)
    file_name = _extract("FileName", xml_text)  # file 消息会有
    pic_url   = _extract("PicUrl", xml_text)    # image 消息可能带

    if not from_user:
        return PlainTextResponse("ok")

    # -------- 文本命令 --------
    if msg_type == "text":
        # /ping
        if content.lower() == "/ping":
            await send_text(from_user, "pong")
            return PlainTextResponse("ok")

        # /web xxx
        use_web_hint = False
        if content.lower().startswith("/web"):
            use_web_hint = True
            content = content[4:].strip()

        sys_prompt = "你是企业微信里的智能助手，回复请简洁、准确。"
        msgs = [{"role": "system", "content": sys_prompt}]

        if use_web_hint and WEB_SEARCH_HINT:
            hint = (
                "（联网提示）请结合权威新闻站点、政府/机构发布与近24-48小时信息为主，"
                "并在可能时给出简短出处名称（不要长链接）。如果不确定，请如实说明。"
            )
            user_text = f"{hint}\n\n用户问题：{content}"
        else:
            user_text = content
        msgs.append({"role": "user", "content": user_text})

        reply = await call_openai_chat(msgs, max_tokens=800, temperature=None)
        await send_text(from_user, reply)
        return PlainTextResponse("ok")

    # -------- 图片 / 文件 --------
    if ALLOW_MEDIA and media_id:
        try:
            data, ctype, filename = await download_media(media_id)
        except Exception as e:
            log.error("download media fail: %s", e)
            await send_text(from_user, "文件下载失败，请稍后再试。")
            return PlainTextResponse("ok")

        # 图片
        if msg_type == "image" or _is_image(ctype, file_name or filename):
            try:
                mime = ctype if ctype.startswith("image/") else "image/png"
                reply = await vision_analyze_image(data, mime)
            except Exception as e:
                log.error("vision analyze fail: %s", e)
                reply = "图片识别失败，请稍后再试。"
            await send_text(from_user, reply)
            return PlainTextResponse("ok")

        # 文件（PDF）
        if msg_type == "file" or _is_pdf(data, ctype, file_name or filename):
            title = (file_name or filename or "PDF")[:60]
            text = _extract_pdf_text(data)
            if not text.strip():
                if LOCAL_OCR:
                    # 可在此加入 OCR 路径（pdf → images → pytesseract），此处留空以避免依赖冲突
                    reply = "这份 PDF 似乎是扫描件，当前服务未启用本地 OCR。请开启 LOCAL_OCR 或提供可复制文本的 PDF。"
                else:
                    reply = "这份 PDF 似乎是扫描件（无法提取文本）。请提供可复制文本的 PDF，或开启本地 OCR 后再试。"
            else:
                reply = await summarize_long_text(text, title)
                reply = f"已解析《{title}》：\n{reply}"
            await send_text(from_user, reply)
            return PlainTextResponse("ok")

        # 其他类型
        await send_text(from_user, "目前仅支持**图片**与**PDF**文件的解析。")
        return PlainTextResponse("ok")

    # 其他消息类型
    await send_text(from_user, "暂不支持此类消息，发送文字、图片或 PDF 试试。")
    return PlainTextResponse("ok")
