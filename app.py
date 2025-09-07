# -*- coding: utf-8 -*-
import os
import re
import json
import time
import asyncio
import logging
import base64
import hashlib
from typing import Dict, List, Tuple, Optional

import httpx
import xmltodict
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

# === OpenAI v1 SDK ===
from openai import OpenAI

# === WeCom 加解密（仅保留 GET 校验用）===
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.utils import to_text

# === AES-CBC 直解依赖 ===
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# === PDF / OCR 依赖（按需使用）===
from pypdf import PdfReader
from io import BytesIO
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("wecom-app")

# ========== 环境变量 ==========
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")  # 组织 ID

# 模型与回退
PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-5").strip()
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-5-mini,gpt-4o-mini").split(",") if m.strip()
]

# 记忆
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()  # redis/memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# WeCom
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")

# 安全模式兼容与解密降级
WECOM_SAFE_MODE = os.getenv("WECOM_SAFE_MODE", "true").lower() == "true"

# PDF/图片解析与摘要
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")  # 用于长文摘要（避免 gpt-5 的参数限制）
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")          # 视觉 / OCR
LOCAL_OCR_ENABLE = os.getenv("LOCAL_OCR_ENABLE", "false").lower() == "true"
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))     # 输入给模型前的最大字符
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))                 # 分块大小（字符）
CHUNK_SUMMARY = int(os.getenv("CHUNK_SUMMARY", "400"))            # 每块摘要建议长度（字符）

# 联网搜索（SerpAPI）
WEB_SEARCH_ENABLE = os.getenv("WEB_SEARCH_ENABLE", "false").lower() == "true"
WEB_PROVIDER = os.getenv("WEB_PROVIDER", "serpapi").lower()  # serpapi / cse
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "")

# ========== OpenAI 客户端 ==========
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,
)

# ========== FastAPI ==========
app = FastAPI()

# ========== 记忆实现（内存 + Redis 兼容） ==========
class MemoryBase:
    async def load(self, uid: str) -> List[Dict]:
        raise NotImplementedError

    async def append(self, uid: str, role: str, content: str):
        raise NotImplementedError


class InMemory(MemoryBase):
    def __init__(self, limit=12):
        self.store: Dict[str, List[Dict]] = {}
        self.limit = limit

    async def load(self, uid: str) -> List[Dict]:
        return self.store.get(uid, [])

    async def append(self, uid: str, role: str, content: str):
        arr = self.store.setdefault(uid, [])
        arr.append({"role": role, "content": content})
        if len(arr) > self.limit:
            self.store[uid] = arr[-self.limit :]


class RedisMemory(MemoryBase):
    def __init__(self, url: str, limit=12, namespace="mem:"):
        self.url = url
        self.limit = limit
        self.ns = namespace
        try:
            import redis.asyncio as redis  # type: ignore
            self.r = redis.from_url(self.url, decode_responses=True)
        except Exception as e:
            logger.warning("Redis init failed: %s", e)
            self.r = None

    async def load(self, uid: str) -> List[Dict]:
        if not self.r:
            raise RuntimeError("Redis not ready")
        key = f"{self.ns}{uid}"
        data = await self.r.get(key)
        return json.loads(data) if data else []

    async def append(self, uid: str, role: str, content: str):
        if not self.r:
            raise RuntimeError("Redis not ready")
        key = f"{self.ns}{uid}"
        data = await self.r.get(key)
        arr = json.loads(data) if data else []
        arr.append({"role": role, "content": content})
        if len(arr) > self.limit:
            arr = arr[-self.limit :]
        await self.r.set(key, json.dumps(arr), ex=60 * 60 * 24 * 7)


if MEMORY_BACKEND == "redis":
    memory: MemoryBase = RedisMemory(REDIS_URL)
else:
    memory = InMemory()

# ========== WeCom 发送/鉴权 ==========
_token_cache = {"value": "", "expire": 0}


async def get_wecom_token() -> str:
    now = time.time()
    if _token_cache["value"] and _token_cache["expire"] > now + 30:
        return _token_cache["value"]
    url = (
        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        tok = data.get("access_token", "")
        exp = int(data.get("expires_in", 7200))
        _token_cache["value"] = tok
        _token_cache["expire"] = now + exp - 60
        return tok


async def send_text(to_user: str, content: str):
    # 避免 44004 空文本
    txt = (content or "").strip()
    if not txt:
        txt = "（模型未输出可读文本，建议换个问法，或发送 /ping 自检出站链路。）"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(WEWORK_AGENT_ID),
        "text": {"content": txt[:4096]},  # 企微单条上限
        "safe": 0,
    }
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    async with httpx.AsyncClient(timeout=12.0) as client:
        r = await client.post(url, json=payload)
        try:
            data = r.json()
        except Exception:
            data = {"errcode": -1, "errmsg": "json decode error"}
        logger.warning("WeCom send result -> to=%s payload_len=%s resp=%s", to_user, len(txt), data)
        if data.get("errcode") != 0:
            # 再试一次（常见 44004）
            if data.get("errcode") == 44004:
                payload["text"]["content"] = "（消息过短或被过滤，已替换为占位文本。）"
                r2 = await client.post(url, json=payload)
                logger.warning("WeCom send first attempt failed: %s, retrying...", data)
                return r2.json()
            raise RuntimeError(f"WeCom send err: {data}")
        return data


# ========== 文本清洗与触发 ==========
def _normalize_text(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("／", "/").replace("：", ":")
    t = re.sub(r"\s+", " ", t)
    return t


def _want_web_route(t: str) -> Tuple[bool, str]:
    """
    触发词：
      /web
      web <query>
      search: <query>
      search <query>
      搜索: <query> / 搜索：<query>
    """
    s = _normalize_text(t).lower()
    raw = _normalize_text(t)
    prefixes = ["/web", "web ", "search:", "search ", "搜索:", "搜索："]
    for p in prefixes:
        if s.startswith(p):
            q = raw[len(p):].strip()
            return True, q
    return False, raw


async def _web_search_serpapi(query: str, k: int = 5):
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY missing")
    params = {
        "engine": "google",
        "q": query,
        "hl": "zh-cn",
        "gl": "cn",
        "num": k,
        "api_key": SERPAPI_KEY,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get("https://serpapi.com/search.json", params=params)
        r.raise_for_status()
        items = r.json().get("organic_results", [])
        return [
            {
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", ""),
            }
            for it in items
        ]


def _render_search(items, provider="serpapi") -> str:
    if not items:
        return "（联网搜索没有检索到结果，换个关键词或加上地点/时间再试）"
    out = [f"🔎 已联网搜索（{provider}）："]
    for i, it in enumerate(items[:5], 1):
        title = it.get("title") or ""
        url = it.get("url") or ""
        out.append(f"[{i}] {title}\n{url}")
    return "\n".join(out)


# ========== OpenAI 对话（主模型 + 回退） ==========
async def ask_models(messages: List[Dict], models: List[str]) -> Tuple[str, str]:
    """
    依次尝试模型，返回 (reply_text, used_model)
    - gpt-5 / gpt-5-mini 不要传 temperature / max_tokens（只用默认）
    """
    for m in models:
        try:
            completion = oai.chat.completions.create(
                model=m,
                messages=messages,
            )
            text = (completion.choices[0].message.content or "").strip()
            if text:
                return text, m
            else:
                logger.warning("primary model %s failed: empty content from primary model", m)
        except Exception as e:
            logger.error("OpenAI call failed for %s: %s", m, e)
    return "（模型临时没有返回内容，建议换个说法或稍后再试）", models[-1] if models else "unknown"


# ========== 长文处理（PDF/图片 OCR）==========
def _smart_truncate(s: str, limit: int = 3500) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."

def _guess_ext_from_ct(ct: str) -> str:
    if not ct:
        return ""
    ct = ct.lower()
    if "pdf" in ct:
        return ".pdf"
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    return ""

def _is_image_filename(name: str) -> bool:
    name = (name or "").lower()
    return any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"])

async def _download_wecom_media(media_id: str) -> Tuple[bytes, str, Optional[str]]:
    """
    下载企微临时素材
    return: (bytes, content_type, suggested_filename)
    """
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        # Content-Disposition: attachment; filename="xxx.pdf"
        disp = r.headers.get("Content-Disposition", "")
        filename = None
        m = re.search(r'filename="?([^";]+)"?', disp)
        if m:
            filename = m.group(1)
        return r.content, ct, filename

def _pdf_extract_text(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    基于 pypdf 的纯文本抽取（不做 OCR）；返回 (全文文本, 页数)
    """
    text_parts: List[str] = []
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = len(reader.pages)
    for i in range(pages):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            text_parts.append(txt.strip())
    return "\n\n".join(text_parts).strip(), pages

async def _summarize_long_text(raw_text: str, filename: str = "") -> str:
    """
    Map-Reduce 式摘要：分块 -> 各块摘要 -> 汇总
    不传 max_tokens/temperature，兼容 gpt-5 家族。
    """
    text = raw_text.strip()
    if not text:
        return "（未提取到可读文本，可能是扫描版或加密PDF）"

    # 截断上限，避免送入过长
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    # 只有一块，直接摘要
    if len(text) <= CHUNK_SIZE:
        sys = "你是文档助理，请用中文给出要点摘要，列出3-6条要点，并给出一句话结论。"
        user = f"文件名：{filename or '(未命名)'}\n请在 {CHUNK_SUMMARY} 字内概括以下内容：\n\n{text}"
        c = oai.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        )
        return c.choices[0].message.content.strip()

    # 多块 map
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    part_summaries: List[str] = []
    for idx, ck in enumerate(chunks, 1):
        prompt = (
            f"下面是文档的第 {idx}/{len(chunks)} 段内容，请用 {CHUNK_SUMMARY} 字以内归纳3-5点要点：\n\n{ck}"
        )
        c = oai.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "system", "content": "中文输出；客观精炼。"},
                      {"role": "user", "content": prompt}],
        )
        part_summaries.append((c.choices[0].message.content or "").strip())

    # reduce 汇总
    combined = "\n".join(f"- {s}" for s in part_summaries if s)
    final_prompt = (
        f"这是一份文档的分段要点，请在 600 字以内汇总为一份条理清晰的中文摘要，给出：\n"
        f"1) 关键要点列表（5-8条）\n2) 关键结论（1-2句）\n\n分段要点：\n{combined}"
    )
    c2 = oai.chat.completions.create(
        model=SUMMARIZER_MODEL,
        messages=[{"role": "system", "content": "中文输出；保留事实细节，避免主观猜测。"},
                  {"role": "user", "content": final_prompt}],
    )
    return (c2.choices[0].message.content or "").strip()

def _to_data_url(image_bytes: bytes, content_type: str) -> str:
    """
    以 data URL 形式给视觉模型（无需外链）
    """
    ct = content_type or "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{ct};base64,{b64}"

async def _ocr_image_with_vision(image_bytes: bytes, content_type: str) -> str:
    """
    使用 OpenAI 视觉模型做 OCR+理解
    """
    data_url = _to_data_url(image_bytes, content_type)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "请识别图片中的所有可读文字，并在需要时做简要总结；保留关键信息（人名、金额、时间）。中文输出。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    c = oai.chat.completions.create(model=VISION_MODEL, messages=messages)
    return (c.choices[0].message.content or "").strip()

def _ocr_image_local(image_bytes: bytes) -> str:
    """
    本地 pytesseract OCR（需系统安装 tesseract，可在 Render 私有镜像中预装）
    """
    if not pytesseract or not Image:
        return ""
    try:
        img = Image.open(BytesIO(image_bytes))
        txt = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return txt.strip()
    except Exception as e:
        logger.warning("local OCR failed: %s", e)
        return ""

async def _process_pdf_and_reply(from_user: str, pdf_bytes: bytes, filename: str):
    text, pages = _pdf_extract_text(pdf_bytes)
    if not text or len(text) < 50:
        msg = (
            f"收到 PDF：{filename or '(未命名)'}（{pages} 页）。"
            f"\n但未能抽取到足够文本，可能是**扫描版**或加密 PDF。"
            f"\n可尝试：\n- 发送每页截图/图片；或\n- 开启本地 OCR（LOCAL_OCR_ENABLE=true，并在系统安装 tesseract）。"
        )
        await send_text(from_user, msg)
        return

    summary = await _summarize_long_text(text, filename=filename or "")
    # 附上基本信息
    head = f"📄 《{filename or '未命名'}》 | 页数：{pages} | 抽取字数：{len(text)}"
    out = head + "\n\n" + summary
    await send_text(from_user, _smart_truncate(out, 3800))

async def _process_image_and_reply(from_user: str, image_bytes: bytes, content_type: str, filename: Optional[str]):
    if LOCAL_OCR_ENABLE:
        txt = _ocr_image_local(image_bytes)
        if txt and len(txt) > 20:
            # 对 OCR 文本再做一次精炼
            s = await _summarize_long_text(txt[:MAX_INPUT_CHARS], filename=filename or "")
            out = f"🖼️ 图片（{filename or '未命名'}）OCR+摘要：\n\n{s}"
            await send_text(from_user, _smart_truncate(out, 3800))
            return
        # 本地 OCR 不足时，回退到云端视觉
        logger.warning("local OCR produced little text, fallback to vision model")

    try:
        v = await _ocr_image_with_vision(image_bytes, content_type or "image/png")
        await send_text(from_user, _smart_truncate(f"🖼️ 图片解析：\n\n{v}", 3800))
    except Exception as e:
        logger.exception("vision ocr failed: %s", e)
        await send_text(from_user, "（图片解析失败，可稍后再试，或关闭本地OCR并使用云端视觉）")

# ========== 状态接口 ==========
@app.get("/")
async def root():
    return JSONResponse(
        {
            "status": "ok",
            "mode": "safe" if WECOM_SAFE_MODE else "plain",
            "service": "WeCom + ChatGPT",
            "model": PRIMARY_MODEL,
            "fallbacks": FALLBACK_MODELS,
            "organization": OPENAI_ORG_ID or "",
            "memory": MEMORY_BACKEND,
            "pdf_support": True,
            "local_ocr": LOCAL_OCR_ENABLE,
            "web_search": {"enabled": WEB_SEARCH_ENABLE, "provider": WEB_PROVIDER},
        }
    )

# ========== GET 校验（仍用 wechatpy）==========
@app.get("/wecom/callback")
async def wecom_verify(request: Request):
    """
    企业微信“接收消息服务器配置”校验；GET 验证回显
    """
    params = dict(request.query_params)
    msg_signature = params.get("msg_signature", "")
    timestamp = params.get("timestamp", "")
    nonce = params.get("nonce", "")
    echostr = params.get("echostr", "")

    crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
    try:
        echo = crypto.decrypt_message(msg_signature, timestamp, nonce, echostr)
        return PlainTextResponse(echo)
    except Exception as e:
        logger.error("wecom-app:URL verify decrypt failed: %s", e)
        return PlainTextResponse("invalid")

# ========== 签名/解密工具 ==========
def wecom_sign(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    """
    企业微信签名算法：对 [token, timestamp, nonce, encrypt] 字符串数组做字典序排序后拼接，取 SHA1 十六进制。
    """
    raw = "".join(sorted([token, str(timestamp), str(nonce), encrypt]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def wecom_decrypt_raw(encrypt_b64: str, aes_key43: str, corp_id_or_suiteid: str) -> str:
    """
    直接按企业微信规范对 Encrypt 字段做 AES-CBC 解密：
    - key = Base64_Decode(aes_key43 + "=")
    - iv  = key[:16]
    - 明文结构 = 16字节随机 + 4字节网络序msg_len + msg_xml + corp_id/suite_id
    """
    if not aes_key43 or len(aes_key43) != 43:
        logger.warning("WECOM_AES_KEY length is not 43, actual=%s", len(aes_key43) if aes_key43 else 0)
    key = base64.b64decode((aes_key43 or "") + "=")
    iv = key[:16]
    ct = base64.b64decode(encrypt_b64)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ct) + decryptor.finalize()

    # PKCS#7 去填充
    pad = padded[-1]
    if pad < 1 or pad > 32:
        raise ValueError(f"bad padding: {pad}")
    plaintext = padded[:-pad]

    # 拆明文
    msg_len = int.from_bytes(plaintext[16:20], "big")
    xml_bytes = plaintext[20:20 + msg_len]
    tail = plaintext[20 + msg_len:].decode("utf-8", "ignore")

    # 校验 corp/suite id（放宽为包含）
    if corp_id_or_suiteid and (corp_id_or_suiteid not in tail):
        logger.warning("wecom decrypt: corp/suite id mismatch: in-xml=%s expected-like=%s", tail, corp_id_or_suiteid)

    return xml_bytes.decode("utf-8", "ignore")

# ========== POST 业务处理（稳健解密 + PDF/图片解析）==========
@app.post("/wecom/callback")
async def wecom_callback(request: Request):
    """
    业务消息处理（POST）
    安全模式：正则抽取 Encrypt -> 签名校验 -> AES-CBC 直解
    兼容明文模式：没有 Encrypt 时，直接解析原文
    支持消息类型：text / image / file
    """
    params = dict(request.query_params)
    msg_signature = params.get("msg_signature", "")
    timestamp = params.get("timestamp", "")
    nonce = params.get("nonce", "")

    # 读取原始 body
    raw = await request.body()
    if not raw:
        logger.error("wecom-app: empty body")
        return PlainTextResponse("success")

    # 兼容 JSON 或 XML，从 body 中抽取 Encrypt
    m = re.search(
        rb"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>|<Encrypt>([^<]+)</Encrypt>|\"Encrypt\"\s*:\s*\"(.*?)\"",
        raw, re.S,
    )

    if m:
        enc_bytes = next(g for g in m.groups() if g)
        encrypt = enc_bytes.decode("utf-8", "ignore")

        # 签名校验（失败仅告警）
        calc_sig = wecom_sign(WECOM_TOKEN, timestamp, nonce, encrypt)
        if calc_sig != msg_signature:
            logger.warning("wecom-app: msg_signature mismatch: got=%s calc=%s", msg_signature, calc_sig)

        try:
            decrypted_xml = wecom_decrypt_raw(encrypt, WECOM_AES_KEY, WEWORK_CORP_ID)
        except Exception:
            head = raw[:120].decode("utf-8", "ignore")
            logger.exception("wecom-app: decrypt failed via raw aes-cbc, head=%r", head)
            return PlainTextResponse("success")
    else:
        # 无 Encrypt：当作明文模式
        decrypted_xml = raw.decode("utf-8", "ignore").strip()

    # 解析“明文 XML”
    try:
        d = xmltodict.parse(decrypted_xml).get("xml", {})
    except Exception:
        logger.exception("wecom-app: parse decrypted xml failed, xml_head=%r", decrypted_xml[:120])
        return PlainTextResponse("success")

    msg_type = (d.get("MsgType") or "").lower()
    from_user = d.get("FromUserName") or ""
    content = (d.get("Content") or "").strip()

    # ---- ping 自检 ----
    if msg_type == "text" and content.strip().lower().startswith("/ping"):
        info = [
            f"当前活跃模型：{PRIMARY_MODEL}",
            f"候选列表：{', '.join([PRIMARY_MODEL] + FALLBACK_MODELS)}",
            f"组织ID：{OPENAI_ORG_ID or '(未设)'}",
            f"记忆：{MEMORY_BACKEND}",
            f"联网搜索：{'on' if WEB_SEARCH_ENABLE else 'off'} / {WEB_PROVIDER}",
            f"PDF/图片解析：已启用（LOCAL_OCR={'on' if LOCAL_OCR_ENABLE else 'off'}）",
        ]
        await send_text(from_user, "\n".join(info))
        return PlainTextResponse("success")

    # ---- Image 图片 ----
    if msg_type == "image":
        media_id = d.get("MediaId") or ""
        if not media_id:
            await send_text(from_user, "（未拿到图片 MediaId）")
            return PlainTextResponse("success")
        try:
            data, ct, fn = await _download_wecom_media(media_id)
        except Exception as e:
            logger.exception("download image failed: %s", e)
            await send_text(from_user, "（下载图片失败）")
            return PlainTextResponse("success")

        await _process_image_and_reply(from_user, data, ct or "image/png", fn)
        return PlainTextResponse("success")

    # ---- File 文件（含 PDF）----
    if msg_type == "file":
        media_id = d.get("MediaId") or ""
        filename = d.get("FileName") or ""
        if not media_id:
            await send_text(from_user, "（未拿到文件 MediaId）")
            return PlainTextResponse("success")

        try:
            data, ct, suggest = await _download_wecom_media(media_id)
            if not filename:
                filename = suggest or ("file" + _guess_ext_from_ct(ct))
        except Exception as e:
            logger.exception("download file failed: %s", e)
            await send_text(from_user, "（下载文件失败）")
            return PlainTextResponse("success")

        # PDF
        if filename.lower().endswith(".pdf") or "pdf" in (ct or "").lower():
            await _process_pdf_and_reply(from_user, data, filename)
            return PlainTextResponse("success")

        # 图片类文件（用户可能从文件选择里发图）
        if _is_image_filename(filename):
            await _process_image_and_reply(from_user, data, ct or "image/png", filename)
            return PlainTextResponse("success")

        await send_text(from_user, f"已收到文件：{filename}（暂只支持 PDF 与常见图片格式）。")
        return PlainTextResponse("success")

    # ---- 文本消息：搜索或闲聊 ----
    if msg_type == "text":
        # 联网搜索
        should_web, web_q = _want_web_route(content)
        if should_web:
            if not WEB_SEARCH_ENABLE:
                await send_text(from_user, "（联网搜索未启用：请把 WEB_SEARCH_ENABLE=true 并配置 SERPAPI_KEY）")
                return PlainTextResponse("success")
            try:
                items = await _web_search_serpapi(web_q, k=5)
                reply_text = _render_search(items, "serpapi")
            except Exception as e:
                logger.exception("web search failed")
                reply_text = f"（联网搜索出错：{e}）"
            await send_text(from_user, reply_text)
            return PlainTextResponse("success")

        # 普通对话（记忆 + 回退）
        try:
            history = []
            try:
                history = await memory.load(from_user)
            except Exception as e:
                logger.warning("load memory failed: %s", e)

            messages = [{"role": "system", "content": "你是企业微信里的智能助手，回答要简洁、直给。"}]
            messages.extend(history[-8:])
            messages.append({"role": "user", "content": content})

            models_try = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m]
            reply_text, used_model = await ask_models(messages, models_try)

            await send_text(from_user, reply_text)

            try:
                await memory.append(from_user, "user", content)
                await memory.append(from_user, "assistant", reply_text)
            except Exception as e:
                logger.warning("append memory failed: %s", e)

            return PlainTextResponse("success")
        except Exception:
            logger.exception("biz error")
            await send_text(from_user, "（服务端异常，可稍后再试或 /ping 自检）")
            return PlainTextResponse("success")

    # ---- 其它消息类型（语音/视频/事件等）----
    await send_text(from_user, "已收到消息（当前支持：文本、图片、PDF）。")
    return PlainTextResponse("success")
