import os
import io
import base64
import json
import time
import logging
import traceback
from typing import List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.background import BackgroundTask

import httpx

# --- optional wechatpy: 仅安全模式解密会用到；明文模式可以没有 ---
try:
    from wechatpy.enterprise.crypto import WeChatCrypto  # type: ignore
    from wechatpy.utils import to_text as _wechat_to_text  # type: ignore
except Exception:
    WeChatCrypto = None
    _wechat_to_text = None

def to_text(val):
    """
    wechatpy.utils.to_text 的轻量兜底实现：
    - wechatpy 存在：直接用官方实现
    - wechatpy 不存在：尽量把 bytes/None 转成 str
    """
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

import xmltodict
# 仅用于 xml->dict（轻量），如果你已用其它库也可替换
# 如果没有 wechatpyrepl，可改为: import xmltodict

from openai import OpenAI

from pypdf import PdfReader

# ====== 环境变量 ======
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 候选模型（按优先级）
PRIMARY_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5")
FALLBACK_MODELS  = [m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-5-mini,gpt-4o-mini").split(",") if m.strip()]
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")  # 用于 PDF/图片摘要
VISION_MODEL     = os.getenv("VISION_MODEL", "gpt-4o-mini")     # 图片 OCR 默认走视觉

OPENAI_ORG_ID    = os.getenv("OPENAI_ORG_ID", "")               # 显式组织（你已经验证通过）

# WeCom
WECOM_TOKEN      = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY    = os.getenv("WECOM_AES_KEY", "")
WEWORK_CORP_ID   = os.getenv("WEWORK_CORP_ID", "")
WEWORK_SECRET    = os.getenv("WEWORK_SECRET", "")
WEWORK_AGENT_ID  = os.getenv("WEWORK_AGENT_ID", "")

# 解析策略
LOCAL_OCR_ENABLE = os.getenv("LOCAL_OCR_ENABLE", "false").lower() in ("1", "true", "yes")
MAX_INPUT_CHARS  = int(os.getenv("MAX_INPUT_CHARS", "120000"))      # 单次最大抽取长度
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "8000"))             # 分块大小
CHUNK_SUMMARY    = int(os.getenv("CHUNK_SUMMARY", "1200"))          # 每块汇总目标字数（提示参考）

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wecom-app")

# OpenAI 客户端（显式组织）
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,
)

app = FastAPI()


# ====== 工具函数 ======
async def get_wecom_token() -> str:
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
    if data.get("errcode") != 0:
        raise RuntimeError(f"gettoken failed: {data}")
    return data["access_token"]


async def send_text(to_user: str, text: str):
    """
    发送文本给单人；自动避免 44004（空内容）：
    - 为空时，发送一个最小兜底文本
    """
    payload_text = text.strip() or "（解析成功，但本段内容过短，请换个问题或重试）"
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(WEWORK_AGENT_ID),
        "text": {"content": payload_text},
        "safe": 0
    }
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.post(url, json=payload)
        data = r.json()
        logger.warning("WeCom send result -> to=%s payload_len=%s resp=%s", to_user, len(payload_text), data)
        if data.get("errcode") != 0:
            raise RuntimeError(f"WeCom send err: {data}")


async def download_media(media_id: str) -> Tuple[bytes, Optional[str]]:
    """下载临时素材，返回(字节, content_type)"""
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"download media failed: {r.status_code}")
        return r.content, r.headers.get("Content-Type")


def _ext_from_content_type(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    ct = ct.lower()
    if "pdf" in ct: return ".pdf"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "heic" in ct: return ".heic"
    return None


def _chunk_text(s: str, size: int) -> List[str]:
    return [s[i:i+size] for i in range(0, len(s), size)]


# ====== 文档解析 ======
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """使用 PyPDF 抽取全文"""
    text_parts: List[str] = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages:
        t = page.extract_text() or ""
        text_parts.append(t)
    text = "\n".join(text_parts).strip()
    # 控长
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    return text


def local_ocr_image(img_bytes: bytes) -> str:
    """本地 OCR（需要系统有 Tesseract；Render 常见镜像没有，默认不用）"""
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(img_bytes))
        txt = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return (txt or "").strip()
    except Exception as e:
        logger.warning("local OCR failed: %s", e)
        return ""


def vision_ocr_image(img_bytes: bytes) -> str:
    """OpenAI 视觉 OCR。把图片转成 data URL 走 vision 模型"""
    b64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"
    try:
        resp = oai.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR engine. Extract all readable text from the image faithfully. Output plain text only."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text in this image. Keep reading order as much as possible."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                }
            ],
            # 视觉模型通常不要求这些参数；这里不设 temperature 等，避免不被支持
            max_completion_tokens=5000,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt
    except Exception as e:
        logger.error("vision OCR failed: %s\n%s", e, traceback.format_exc())
        return ""


def summarize_long_text(raw: str, goal_len: int = 400) -> str:
    """
    超长文本递归汇总：
    1) >CHUNK_SIZE 分块 -> 每块生成要点
    2) 把要点合并再总结
    """
    if not raw.strip():
        return "（没有可总结的文本内容）"

    # 单块直接总结
    if len(raw) <= CHUNK_SIZE:
        prompt = (
            f"请根据以下原文做结构化中文摘要，面向商务读者，尽量包含关键数据/方法/结论。\n"
            f"- 目标长度：约 {goal_len} 字\n"
            f"- 输出格式：三要点 + 一句结论\n"
            f"- 若原文非中文，请先翻译再总结\n\n"
            f"【原文】\n{raw}"
        )
        try:
            r = oai.chat.completions.create(
                model=SUMMARIZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=goal_len * 2
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("summarize failed: %s", e)
            return "（摘要失败，请稍后重试）"

    # 多块：先分块要点 -> 汇总
    chunks = _chunk_text(raw, CHUNK_SIZE)
    points: List[str] = []
    for idx, ch in enumerate(chunks, 1):
        sub_prompt = (
            f"以下是第 {idx}/{len(chunks)} 段文本，请提取3-5个要点（中文，保留关键数据/术语），每个要点一行：\n\n{ch}"
        )
        try:
            r = oai.chat.completions.create(
                model=SUMMARIZER_MODEL,
                messages=[{"role": "user", "content": sub_prompt}],
                max_completion_tokens=CHUNK_SUMMARY
            )
            pts = (r.choices[0].message.content or "").strip()
            points.append(pts)
        except Exception as e:
            logger.warning("chunk summarize failed: %s", e)

    all_pts = "\n".join(points)
    final_prompt = (
        "根据下列分块要点，整合为一份完整中文摘要：\n"
        f"- 总体长度：约 {goal_len} 字\n"
        "- 结构：三要点 + 一句结论\n"
        "- 保留关键数据与方法\n\n"
        f"【分块要点汇总】\n{all_pts}"
    )
    try:
        r = oai.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            max_completion_tokens=goal_len * 2
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("final summarize failed: %s", e)
        return all_pts[:goal_len] or "（摘要失败，请稍后重试）"


# ====== OpenAI 调用（文本对话，带回退与空输出兜底）======
async def ask_openai_with_fallback(user_text: str) -> str:
    candidates = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m]
    last_err = None

    for m in candidates:
        try:
            # 注意：gpt-5 系列不接受 temperature/max_tokens 参数，用 max_completion_tokens
            resp = oai.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": user_text}],
                max_completion_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            logger.warning("primary model %s failed: empty content from primary model", m)
        except Exception as e:
            last_err = e
            logger.warning("model %s failed: %s", m, e)
            continue

    # 所有候选失败 => 兜底
    logger.error("all models failed, last error: %s", last_err)
    return "（模型临时没有返回内容，建议换个问法或稍后再试）"


# ====== WeCom 回调 ======
@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "status": "ok",
        "mode": "safe",
        "service": "WeCom + ChatGPT",
        "model": f"{PRIMARY_MODEL}",
        "candidates": [PRIMARY_MODEL] + FALLBACK_MODELS,
        "memory": "memory",
        "pdf_support": True,
        "local_ocr": LOCAL_OCR_ENABLE,
    }


@app.get("/wecom/callback")
async def wecom_verify(request: Request):
    """
    明文/安全模式兼容：GET 验签后直接把 echostr 原样返回
    """
    echostr = request.query_params.get("echostr", "")
    return PlainTextResponse(echostr or "ok")


from fastapi import Request
# 顶部 import（如已存在可跳过）
import re, json, gzip, zlib
from starlette.responses import PlainTextResponse

def _head_preview(s: str) -> str:
    return re.sub(r"[\r\n\t ]+", " ", (s or "")[:120])

def _ensure_xml_for_wechatpy(body_text: str) -> tuple[str, str]:
    """
    让任意形态的 body 变成 wechatpy.decrypt_message 所需 XML。
    返回: (xml_string, how)；xml_string 为空说明失败。
    """
    t = (body_text or "").lstrip("\ufeff").strip()
    if not t:
        return "", "empty"

    # 1) JSON -> 构造 xml
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

    # 2) 原本就是 XML
    if t.startswith("<"):
        # 已包含 Encrypt 就直接交给 wechatpy
        if ("<Encrypt>" in t) or ("<Encrypt><![CDATA[" in t):
            return t, "xml"
        # 没有 Encrypt，尝试从 XML 内抢救
        m = re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", t, re.S)
        if m:
            enc = m.group(1)
            xml = f"<xml><ToUserName><![CDATA[]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
            return xml, "xml-salvaged"

    # 3) 杂形态：urlencoded / 代理前缀/文本包裹等，尽力提取 Encrypt
    m = (re.search(r'"Encrypt"\s*:\s*"([^"]+)"', t) or
         re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", t, re.S) or
         re.search(r"encrypt=([A-Za-z0-9+/=]+)", t))
    if m:
        enc = m.group(1)
        xml = f"<xml><ToUserName><![CDATA[]]></ToUserName><Encrypt><![CDATA[{enc}]]></Encrypt></xml>"
        return xml, "regex-salvaged"

    return "", "no-encrypt-field"

@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    # 1) query
    msg_signature = request.query_params.get("msg_signature")
    timestamp     = request.query_params.get("timestamp")
    nonce         = request.query_params.get("nonce")

    # 2) 读 body（只读一次） + 解压 + 去 BOM
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

    # 是否安全模式：有签名 + AES_KEY
    is_safe_mode = bool(msg_signature and (WECOM_AES_KEY or "").strip())

    data = {}
    if is_safe_mode:
        if WeChatCrypto is None:
            logger.error("decrypt fail: wechatpy not installed")
            return PlainTextResponse("success")

        # 将 body 统一整理成 wechatpy 需要的 XML
        xml_for_wechatpy, how = _ensure_xml_for_wechatpy(text)
        if not xml_for_wechatpy:
            logger.error(
                "safe-mode: cannot convert body to xml (%s); head=%r; ct=%s",
                how, _head_preview(text), request.headers.get("Content-Type")
            )
            return PlainTextResponse("success")

        logger.info("safe-mode: using payload (%s); head=%r", how, _head_preview(xml_for_wechatpy))

        # 解密
        try:
            crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
            decrypted_xml = crypto.decrypt_message(
                msg_signature, timestamp, nonce, xml_for_wechatpy
            )
        except Exception:
            logger.exception("ERROR:wecom-app:decrypt fail (safe-mode)")
            return PlainTextResponse("success")

        # 解析解密后 XML
        try:
            data = xmltodict.parse(to_text(decrypted_xml)).get("xml", {})
        except Exception:
            logger.exception("safe-mode: parse decrypted xml fail. head=%r", _head_preview(to_text(decrypted_xml)))
            return PlainTextResponse("success")

    else:
        # 明文模式
        if not text or not text.startswith("<"):
            logger.warning("plain-mode: body not xml. head=%r", _head_preview(text))
            return PlainTextResponse("success")
        try:
            data = xmltodict.parse(text).get("xml", {})
        except Exception:
            logger.exception("plain-mode: parse xml fail. head=%r", _head_preview(text))
            return PlainTextResponse("success")

    # ---------- 以下继续你原来的业务逻辑 ----------
    msg_type  = (data.get("MsgType") or "").lower()
    from_user = data.get("FromUserName") or ""
    content   = (data.get("Content") or "").strip()
    pic_url   = data.get("PicUrl") or ""
    media_id  = data.get("MediaId") or ""

    if content.lower() == "/ping":
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    # .... 继续你的指令/LLM/PDF/图片逻辑
    # reply_text = await handle_message(...)
    # await send_text(from_user, reply_text)
    return PlainTextResponse("success")
