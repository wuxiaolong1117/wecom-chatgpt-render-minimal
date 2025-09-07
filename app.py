import os
import time
import json
import base64
import asyncio
import logging
import threading
import hashlib
from typing import Optional, Tuple, List
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from dotenv import load_dotenv
from lxml import etree
from openai import OpenAI

from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# ---------- 可选：PDF 解析 ----------
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

# ---------- 可选：本地 OCR（默认关闭，不建议在 Render 上启用） ----------
USE_LOCAL_OCR = False
try:
    if USE_LOCAL_OCR:
        from PIL import Image
        import pytesseract
        LOCAL_OCR = True
    else:
        LOCAL_OCR = False
except Exception:
    LOCAL_OCR = False

# ---------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wecom-app")

app = FastAPI(title="WeCom + ChatGPT (files/images/groups)")

# ----- OpenAI -----
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_MODEL_LIST = os.getenv("OPENAI_MODEL_LIST", "").strip()
OPENAI_ORG_ID   = os.getenv("OPENAI_ORG_ID", "").strip()

MODEL_CANDIDATES: List[str] = [m.strip() for m in OPENAI_MODEL_LIST.split(",") if m.strip()] or [OPENAI_MODEL]
app_state = {"active_model": MODEL_CANDIDATES[0]}

oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID or None,  # 显式绑定到已验证组织
)

# ----- WeCom -----
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
# Access Token 缓存 + 【① 重试与兜底】
# ---------------------------------------------------------------------
_TOKEN_CACHE = {"value": None, "exp": 0}

async def get_wecom_token() -> str:
    now = int(time.time())
    if _TOKEN_CACHE["value"] and _TOKEN_CACHE["exp"] > now + 30:
        return _TOKEN_CACHE["value"]

    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORP_ID}&corpsecret={SECRET}"
    last_err = None
    for attempt in range(3):  # 指数退避：0,1,2 -> sleep 1s/2s/4s
        try:
            timeout = httpx.Timeout(15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                data = r.json()
            access_token = data.get("access_token", "")
            if access_token:
                _TOKEN_CACHE["value"] = access_token
                _TOKEN_CACHE["exp"] = now + 7000
                return access_token
            else:
                last_err = RuntimeError(f"gettoken no token: {data}")
        except Exception as e:
            last_err = e
        await asyncio.sleep(2 ** attempt)

    # 兜底：有旧 token 先用，避免丢回复
    if _TOKEN_CACHE["value"]:
        log.warning("gettoken failed (%s), fallback to cached token", last_err)
        return _TOKEN_CACHE["value"]

    raise last_err or RuntimeError("gettoken failed")

# ---------------------------------------------------------------------
# 发送消息到 WeCom 【② send_text 失效重试 + 空内容兜底】
# ---------------------------------------------------------------------
async def send_text(to_user: str, content: str) -> dict:
    # ---- 兜底，避免空内容触发 44004 ----
    if not content or not str(content).strip():
        content = "（空内容占位：模型未返回文本，请重试或换个问法）"

    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "agentid": int(AGENT_ID),
        "text": {"content": content[:2048]},
        "safe": 0,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        r = await client.post(url, json=payload)
        try:
            data = r.json()
        except Exception:
            data = {"status_code": r.status_code, "text": r.text}

    if isinstance(data, dict) and data.get("errcode") not in (0, None):
        log.warning("WeCom send first attempt failed: %s, retrying...", data)
        # 强制刷新 token 再发一次
        _TOKEN_CACHE["exp"] = 0
        token = await get_wecom_token()
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            r = await client.post(url, json=payload)
            try:
                data = r.json()
            except Exception:
                data = {"status_code": r.status_code, "text": r.text}

    log.warning("WeCom send result -> to=%s payload_len=%s resp=%s", to_user, len(content), data)
    return data

async def send_long_text(to_user: str, content: str, part_len: int = 1800, max_parts: int = 20):
    if not content:
        return
    parts = [content[i:i+part_len] for i in range(0, len(content), part_len)]
    for idx, p in enumerate(parts[:max_parts], start=1):
        suffix = f"\n\n({idx}/{len(parts)})" if len(parts) > 1 else ""
        await send_text(to_user, p + suffix)
    if len(parts) > max_parts:
        await send_text(to_user, f"……内容较长，已发送前 {max_parts} 段，共 {len(parts)} 段。")

# ---------------------------------------------------------------------
# 群发：Webhook + appchat（可选启用）
# ---------------------------------------------------------------------
ENABLE_GROUP_WEBHOOK = os.getenv("ENABLE_GROUP_WEBHOOK", "false").lower() == "true"
WEBHOOK_KEYS = [k.strip() for k in os.getenv("WECOM_GROUP_WEBHOOK_KEYS", "").split(",") if k.strip()]

async def send_group_webhook_text(content: str) -> list:
    if not (ENABLE_GROUP_WEBHOOK and WEBHOOK_KEYS):
        return []
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for key in WEBHOOK_KEYS:
            url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
            payload = {"msgtype": "text", "text": {"content": content[:2048]}}
            r = await client.post(url, json=payload)
            try:
                results.append(r.json())
            except Exception:
                results.append({"status_code": r.status_code, "text": r.text})
    return results

ENABLE_APPCHAT = os.getenv("ENABLE_APPCHAT", "false").lower() == "true"
APPCHAT_DEFAULT_CHATID = os.getenv("APPCHAT_DEFAULT_CHATID", "").strip()

async def appchat_send_text(chatid: str, content: str) -> dict:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/appchat/send?access_token={token}"
    payload = {"chatid": chatid, "msgtype": "text", "text": {"content": content[:2048]}, "safe": 0}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, json=payload)
        return r.json()

async def appchat_create(name: str, owner_userid: str, userlist: List[str]) -> dict:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/appchat/create?access_token={token}"
    payload = {"name": name, "owner": owner_userid, "userlist": userlist}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, json=payload)
        return r.json()

# ---------------------------------------------------------------------
# XML & 媒体下载
# ---------------------------------------------------------------------
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
        "MediaId": get("MediaId"),
        "PicUrl": get("PicUrl"),
        "FileName": get("FileName"),
        "MsgId": get("MsgId"),
        "AgentID": get("AgentID"),
        "Event": get("Event"),
    }

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
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(url)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
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
# PDF / 图片
# ---------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    if not PDF_SUPPORT:
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

async def analyze_image_with_openai(path: str, content_type: Optional[str] = None, task: str = "请提取图片中的文字，并简要说明图像要点。"):
    data_url = encode_image_to_data_url(path, content_type)
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": task},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }]
        resp = create_chat_with_tokens(app_state.get("active_model", MODEL_CANDIDATES[0]), messages, max_new_tokens=800)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"OpenAI 视觉解析失败：{e}"

def ocr_image_locally(path: str) -> str:
    if not LOCAL_OCR:
        return "本地 OCR 未启用。建议使用多模态模型（已默认启用）。"
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text.strip()
    except Exception as e:
        return f"OCR 失败：{e}"

# ---------------------------------------------------------------------
# Memory：内存 + Redis
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
# 幂等去重：优先 Redis (SETNX+EX)；无 Redis 用本地字典 + TTL
# ---------------------------------------------------------------------
DEDUP_LOCAL = {}  # msg_key -> expire_ts

async def is_duplicated_and_mark(msg_key: str, ttl: int = 300) -> bool:
    """
    返回 True 表示这条消息已处理过（重复），False 表示第一次看到并已标记。
    """
    now = int(time.time())

    # 优先使用 Redis
    try:
        if REDIS_MEMORY:
            cli = await REDIS_MEMORY.client()
            key = f"wecom:dedup:{msg_key}"
            added = await cli.setnx(key, "1")
            if added:
                await cli.expire(key, ttl)
                return False
            else:
                return True
    except Exception as e:
        log.warning("dedup via redis failed: %s; fallback to local", e)

    # 本地兜底：清理过期
    if DEDUP_LOCAL and len(DEDUP_LOCAL) % 128 == 0:
        exp_keys = [k for k, ex in DEDUP_LOCAL.items() if ex <= now]
        for k in exp_keys:
            DEDUP_LOCAL.pop(k, None)

    if DEDUP_LOCAL.get(msg_key, 0) > now:
        return True

    DEDUP_LOCAL[msg_key] = now + ttl
    return False

def make_msg_key(msg: dict) -> str:
    """
    生成去重用的 key：
    - 优先使用 MsgId
    - 无 MsgId（如 Event）时用字段指纹
    """
    raw = (
        msg.get("MsgId")
        or "|".join([
            msg.get("FromUserName") or "",
            msg.get("CreateTime") or "",
            msg.get("MsgType") or "",
            msg.get("Event") or "",
            msg.get("Content") or "",
            msg.get("MediaId") or "",
            msg.get("FileName") or "",
        ])
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# GPT-5/4o 参数兼容 + 模型降级
# ---------------------------------------------------------------------
def create_chat_with_tokens(model: str, messages, max_new_tokens=300, temp_for_non_gpt5=0.3):
    """
    gpt-5: 使用 max_completion_tokens，且不传 temperature
    其它：使用 max_tokens，并可设置 temperature
    """
    kwargs = dict(model=model, messages=messages)
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_new_tokens
        # 不传 temperature
    else:
        kwargs["max_tokens"] = max_new_tokens
        kwargs["temperature"] = temp_for_non_gpt5
    return oai.chat.completions.create(**kwargs)

def _should_downgrade(err: Exception) -> bool:
    s = str(err).lower()
    return ("model_not_found" in s) or ("must be verified" in s) or ("404" in s and "not found" in s)

def chat_with_fallback(messages):
    last_err = None
    for mdl in MODEL_CANDIDATES:
        try:
            resp = create_chat_with_tokens(mdl, messages, max_new_tokens=300)
            app_state["active_model"] = mdl
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if _should_downgrade(e):
                log.warning("Model %s unavailable, trying next...", mdl)
                continue
            raise
    raise last_err if last_err else RuntimeError("no model available")

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
# 消息回调（含文本/图片/文件） 【③ 失败返回 500 触发企微重试 + 幂等去重】
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

    # -------- 幂等去重：重复投递直接丢弃，避免多次回复 --------
    msg_key = make_msg_key(msg)
    if await is_duplicated_and_mark(msg_key, ttl=300):
        log.info("dup message dropped: key=%s from=%s type=%s", msg_key, msg.get("FromUserName"), msg.get("MsgType"))
        return PlainTextResponse("success")

    from_user = msg.get("FromUserName") or ""
    msg_type = (msg.get("MsgType") or "").lower()
    content = msg.get("Content") or ""
    media_id = msg.get("MediaId")
    file_name = msg.get("FileName") or ""

    # 简单测试命令：绕过 LLM，验证出站链路
    if (content or "").strip().lower() in ("/ping", "ping", "测试"):
        try:
            ret = await send_text(from_user, "pong")
            if isinstance(ret, dict) and ret.get("errcode", 0) != 0:
                raise RuntimeError(f"WeCom send err: {ret}")
            return PlainTextResponse("success")
        except Exception as e:
            log.exception("WeCom send failed: %s", e)
            return PlainTextResponse("retry later", status_code=500)

    # ====== 图片 ======
    if msg_type == "image" and media_id:
        path, ctype = await download_media_to_tmp(media_id)
        if USE_LOCAL_OCR and LOCAL_OCR:
            text = ocr_image_locally(path)
        else:
            text = await analyze_image_with_openai(path, ctype)
        try:
            await send_long_text(from_user, f"图片解析结果：\n{text}")
            return PlainTextResponse("success")
        except Exception as e:
            log.exception("WeCom send failed: %s", e)
            return PlainTextResponse("retry later", status_code=500)

    # ====== 文件（PDF 优先）======
    if msg_type == "file" and media_id:
        path, ctype = await download_media_to_tmp(media_id)
        ext = os.path.splitext(file_name or path)[-1].lower()
        try:
            if ext == ".pdf" or "pdf" in (ctype or "").lower():
                pdf_text = extract_text_from_pdf(path)
                if not pdf_text.strip():
                    await send_text(from_user, "该 PDF 提取不到可搜索文本（可能是扫描版）。可改为图片解析（OpenAI 视觉）或启用本地 OCR。")
                else:
                    await send_long_text(from_user, f"《{file_name or 'PDF 文件'}》全文提取：\n{pdf_text}", part_len=1800, max_parts=20)
            else:
                await send_text(from_user, f"已收到文件：{file_name or os.path.basename(path)}（类型：{ctype or '未知'}）。当前仅对 PDF 做全文提取，其它类型可先转为 PDF 或图片。")
            return PlainTextResponse("success")
        except Exception as e:
            log.exception("WeCom send failed: %s", e)
            return PlainTextResponse("retry later", status_code=500)

    # ====== 群发命令解析（可选）======
    text_for_llm = content or (msg.get("Event") or "")
    broadcast_flag = False
    new_chat_req = None  # (name, members)

    low = (text_for_llm or "").strip()
    if low.startswith(("群发", "/broadcast", "broadcast")):
        broadcast_flag = True
        for p in ("群发：", "群发:", "群发 ", "/broadcast ", "broadcast:", "broadcast "):
            if low.startswith(p):
                text_for_llm = low[len(p):].strip() or text_for_llm
                break

    if low.startswith(("建群 ", "/mkchat ")):
        try:
            body = low.split(" ", 1)[1].strip()
            name, members = body.split(" ", 1)
            members = [m.strip().strip("@") for m in members.split(",") if m.strip()]
            if from_user and from_user not in members:
                members = [from_user] + members
            new_chat_req = (name, members)
        except Exception:
            new_chat_req = None

    # ====== 构造上下文 ======
    base_system = {"role": "system", "content": "You are a helpful assistant for WeCom users."}
    messages = [base_system]

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

    messages.append({"role": "user", "content": text_for_llm})

    # ====== 调 OpenAI（含降级） ======
    try:
        reply_text = chat_with_fallback(messages)
        # 模型文本兜底，避免空字符串
        if not reply_text or not reply_text.strip():
            reply_text = "（模型未输出可读文本。建议换个问法，或发送 /ping 自检出站链路。）"
    except Exception as e:
        log.exception("OpenAI call failed: %s", e)
        # 给用户也反馈错误
        try:
            ret = await send_text(from_user, f"OpenAI 调用失败：{e}")
            if isinstance(ret, dict) and ret.get("errcode", 0) != 0:
                raise RuntimeError(f"WeCom send err: {ret}")
            return PlainTextResponse("success")
        except Exception as e2:
            log.exception("WeCom send failed: %s", e2)
            return PlainTextResponse("retry later", status_code=500)

    # ====== 写记忆 ======
    if ENABLE_MEMORY and from_user:
        try:
            if REDIS_MEMORY:
                await REDIS_MEMORY.append_turn(from_user, content, reply_text)
            elif MEMORY:
                MEMORY.append_turn(from_user, content, reply_text)
        except Exception as e:
            log.warning("append memory failed: %s", e)

    # ====== 建群 / 群发 执行 ======
    try:
        if new_chat_req and ENABLE_APPCHAT:
            name, members = new_chat_req
            ret = await appchat_create(name=name, owner_userid=from_user, userlist=members)
            await send_text(from_user, f"建群结果：{ret}")
            if ret.get("errcode") == 0 and ret.get("chatid"):
                await send_text(from_user, f"新群 chatid：{ret['chatid']}。可设置为 APPCHAT_DEFAULT_CHATID 用于群发。")

        if broadcast_flag:
            sent = False
            if ENABLE_APPCHAT and APPCHAT_DEFAULT_CHATID:
                try:
                    ret = await appchat_send_text(APPCHAT_DEFAULT_CHATID, reply_text)
                    await send_text(from_user, f"(appchat) 群发结果：{ret}")
                    sent = True
                except Exception as e:
                    await send_text(from_user, f"(appchat) 群发失败：{e}")
            if ENABLE_GROUP_WEBHOOK and WEBHOOK_KEYS:
                try:
                    rets = await send_group_webhook_text(reply_text)
                    await send_text(from_user, f"(webhook) 群发结果：{rets}")
                    sent = True
                except Exception as e:
                    await send_text(from_user, f"(webhook) 群发失败：{e}")
            if not sent:
                await send_text(from_user, "未配置群聊通道：请设置 APPCHAT_DEFAULT_CHATID 或 WECOM_GROUP_WEBHOOK_KEYS。")

        # ====== 最终回复用户 ======
        ret = await send_text(from_user, reply_text)
        if isinstance(ret, dict) and ret.get("errcode", 0) != 0:
            raise RuntimeError(f"WeCom send err: {ret}")
        return PlainTextResponse("success")
    except Exception as e:
        log.exception("WeCom send failed: %s", e)
        # 返回 500 让企微重试
        return PlainTextResponse("retry later", status_code=500)

# ---------------------------------------------------------------------
# 健康 & 调试
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "mode": "safe" if crypto else "plain",
        "service": "WeCom + ChatGPT",
        "model_candidates": MODEL_CANDIDATES,
        "active_model": app_state.get("active_model", MODEL_CANDIDATES[0]),
        "memory": "redis" if REDIS_MEMORY else ("memory" if ENABLE_MEMORY else "disabled"),
        "pdf_support": PDF_SUPPORT,
        "local_ocr": LOCAL_OCR,
    }

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/debug/ping")
async def debug_ping(user: str):
    ret = await send_text(user, "调试：这是一条来自 /debug/ping 的消息。")
    return ret
