import os
import json
import time
import base64
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

import xmltodict

# 企业微信加解密
from wechatpy.enterprise.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException

# OpenAI 官方 SDK
from openai import OpenAI

# ====== 日志 ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wecom-app")

# ====== 环境变量 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")          # 主模型（默认 gpt-5）
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini") # 回退模型（视觉/联网也优先用 4o-mini）
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))

# 企业微信
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")
WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")

# 记忆
REDIS_URL = os.getenv("REDIS_URL", "")   # eg. redis://:pwd@host:6379/0
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "8"))

# 联网搜索
WEB_SEARCH_DEFAULT = os.getenv("WEB_SEARCH_DEFAULT", "off").lower() in ("1", "true", "yes", "on")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ====== OpenAI 客户端 ======
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is empty!")
oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=(OPENAI_ORG_ID or None),
)

# ====== HTTP 客户端 ======
TIMEOUT = httpx.Timeout(20.0, connect=10.0)
client = httpx.AsyncClient(timeout=TIMEOUT)

# ====== 企业微信加解密实例（密文模式时使用）======
crypto: Optional[WeChatCrypto] = None
if WECOM_TOKEN and WECOM_AES_KEY and WEWORK_CORP_ID:
    try:
        crypto = WeChatCrypto(WECOM_TOKEN, WECOM_AES_KEY, WEWORK_CORP_ID)
        logger.info("WeCom safe-mode enabled.")
    except Exception as e:
        logger.warning(f"WeCom safe-mode init failed: {e}")

# ====== 记忆：Redis 首选，失败退内存 ======
class MemoryStore:
    def __init__(self):
        self.use_redis = False
        self.redis = None
        self.mem: Dict[str, List[Tuple[str, str]]] = {}

        if REDIS_URL:
            try:
                import redis  # type: ignore
                self.redis = redis.from_url(REDIS_URL, decode_responses=True)
                self.redis.ping()
                self.use_redis = True
                logger.info("Memory: using Redis.")
            except Exception as e:
                logger.warning(f"load memory failed: {e}")

    def _key(self, uid: str) -> str:
        return f"wecom:mem:{uid}"

    def load(self, uid: str) -> List[Tuple[str, str]]:
        if self.use_redis:
            try:
                data = self.redis.get(self._key(uid))
                if data:
                    return json.loads(data)
                return []
            except Exception as e:
                logger.warning(f"load memory failed: {e}")
                return []
        return self.mem.get(uid, [])

    def append(self, uid: str, role: str, content: str):
        if not content:
            return
        arr = self.load(uid)
        arr.append((role, content))
        arr = arr[-MEMORY_WINDOW:]
        if self.use_redis:
            try:
                self.redis.set(self._key(uid), json.dumps(arr), ex=60*60*24*7)
            except Exception as e:
                logger.warning(f"append memory failed: {e}")
        else:
            self.mem[uid] = arr

MEM = MemoryStore()

# ====== Web Search 工具开关 & 提供商检测 ======
web_enabled_flag = WEB_SEARCH_DEFAULT  # 进程级开关，可用 /web on off 切
def has_search_provider() -> bool:
    return bool(GOOGLE_API_KEY and GOOGLE_CSE_ID) or bool(SERPER_API_KEY) or bool(TAVILY_API_KEY)

# ====== WeCom Token 缓存 ======
_wecom_token_cache = {"token":"", "exp":0}

async def get_wecom_token() -> str:
    now = int(time.time())
    if _wecom_token_cache["token"] and _wecom_token_cache["exp"] - now > 60:
        return _wecom_token_cache["token"]
    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={WEWORK_CORP_ID}&corpsecret={WEWORK_SECRET}"
    )
    r = await client.get(url)
    data = r.json()
    token = data.get("access_token", "")
    if token:
        _wecom_token_cache["token"] = token
        _wecom_token_cache["exp"] = now + data.get("expires_in", 7200)
    return token

# ====== WeCom 发送文本（分片 & 兜底）======
async def send_text(userid: str, text: str):
    if not text or not text.strip():
        text = "（生成内容为空，建议换个问法或稍后再试）"
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    max_len = 1800
    chunks = []
    s = text
    while s:
        chunks.append(s[:max_len])
        s = s[max_len:]

    for part in chunks:
        payload = {
            "touser": userid,
            "agentid": int(WEWORK_AGENT_ID),
            "msgtype": "text",
            "text": {"content": part},
            "safe": 0,
        }
        r = await client.post(url, json=payload)
        try:
            ret = r.json()
        except Exception:
            ret = {"errcode": -1, "errmsg": "invalid json"}
        logger.warning(f"WeCom send result -> to={userid} payload_len={len(part)} resp={ret}")
        if ret.get("errcode") == 44004:
            payload["text"]["content"] = "（生成内容为空，建议换个问法或稍后再试）"
            r2 = await client.post(url, json=payload)
            try:
                ret2 = r2.json()
            except Exception:
                ret2 = {"errcode": -1, "errmsg": "invalid json"}
            logger.warning(f"WeCom send result -> to={userid} payload_len={len(payload['text']['content'])} resp={ret2}")
            if ret2.get("errcode") != 0:
                raise RuntimeError(f"WeCom send err: {ret2}")
        elif ret.get("errcode") != 0:
            raise RuntimeError(f"WeCom send err: {ret}")

# ====== OpenAI 结果取值与兜底（补丁②）======
def _pick_text(choices: List[Dict[str, Any]]) -> str:
    if not choices:
        return ""
    for ch in choices:
        msg = ch.get("message") or {}
        txt = (msg.get("content") or "").strip()
        if txt:
            return txt
    for ch in choices:
        msg = ch.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            return "（已调用联网检索工具，正在整合结果…如无输出请稍后重试或换种问法）"
    return ""

# ====== OpenAI 封装：文本聊天（含 gpt-5 兼容、按需 tools、回退）======
async def ask_openai_text(messages: List[Dict[str, Any]],
                          use_tools: bool = False,
                          tools: Optional[List[Dict[str,Any]]] = None,
                          prefer_model: Optional[str] = None) -> str:
    model = prefer_model or OPENAI_MODEL
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if MAX_COMPLETION_TOKENS:
        kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    if use_tools and tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    try:
        comp = oai.chat.completions.create(**kwargs)
        text = _pick_text(comp.choices)
    except Exception as e:
        logger.error(f"OpenAI call failed(main): {e}")
        text = ""

    if not text.strip():
        fb_kwargs = {"model": FALLBACK_MODEL, "messages": messages}
        if MAX_COMPLETION_TOKENS:
            fb_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        if use_tools and tools:
            fb_kwargs["tools"] = tools
            fb_kwargs["tool_choice"] = "auto"
        try:
            comp2 = oai.chat.completions.create(**fb_kwargs)
            text = _pick_text(comp2.choices)
        except Exception as e:
            logger.error(f"OpenAI call failed(fallback): {e}")
            text = ""
    return text

# ====== 视觉 OCR（gpt-4o-mini）======
async def ocr_images_to_text(image_b64_list: List[str]) -> str:
    content = [{
        "type": "text",
        "text": "请把图片中的可读文字完整提取为纯文本；如有结构（标题、列表、表格）请用简单标记保留。"
    }]
    for b64 in image_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    kwargs = {
        "model": FALLBACK_MODEL,  # 视觉默认用 gpt-4o-mini
        "messages": [{"role": "user", "content": content}],
    }
    if MAX_COMPLETION_TOKENS:
        kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    try:
        comp = oai.chat.completions.create(**kwargs)
        return _pick_text(comp.choices)
    except Exception as e:
        logger.error(f"vision ocr error: {e}")
        return ""

# ====== 文本分块与两阶段摘要 ======
def split_text(s: str, max_chars: int = 6000) -> List[str]:
    s = s.replace("\x00", "")
    return [s[i:i+max_chars] for i in range(0, len(s), max_chars)]

async def summarize_long_text(full_text: str, title: str = "") -> str:
    if not full_text.strip():
        return ""
    chunks = split_text(full_text, 6000)
    partial_summaries = []
    for idx, ck in enumerate(chunks, 1):
        msgs = [
            {"role":"system","content":"你是资深中文编辑，提炼关键要点，直给信息；保留关键数字与专有名词。"},
            {"role":"user","content":f"以下为第 {idx}/{len(chunks)} 段全文内容，请用要点列表概括：\n\n{ck}"}
        ]
        part = await ask_openai_text(msgs, use_tools=False)
        partial_summaries.append(f"【第{idx}段摘要】\n{part}")

    merge_text = "\n\n".join(partial_summaries)
    final_msgs = [
        {"role":"system","content":"你是资深中文分析师。"},
        {"role":"user","content":f"这是一份长文的分段摘要，请合并为一份完整中文总结：\n"
                                 f"1) 先给『一句话结论』；\n"
                                 f"2) 再给『关键要点清单』；\n"
                                 f"3) 如有建议或风险，请列出。\n\n{merge_text}"}
    ]
    final = await ask_openai_text(final_msgs, use_tools=False)
    if title:
        final = f"《{title}》全文解析\n\n{final}"
    return final

# ====== 下载企业微信媒体 ======
async def download_wecom_media(media_id: str) -> bytes:
    token = await get_wecom_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    r = await client.get(url)
    r.raise_for_status()
    return r.content

# ====== PDF 解析：文本优先 + 视觉 OCR 兜底 ======
def extract_pdf_text_by_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        logger.warning(f"pypdf extract failed: {e}")
        return ""

def render_pdf_pages_to_png_b64(pdf_bytes: bytes, max_pages: int = 10, dpi: int = 180) -> List[str]:
    """用 PyMuPDF 将前 max_pages 页渲染为 PNG，返回 base64 列表。"""
    b64_list: List[str] = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(max_pages, doc.page_count)
        for i in range(pages):
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            b64_list.append(b64)
    except Exception as e:
        logger.warning(f"render pdf to png failed: {e}")
    return b64_list

async def parse_pdf_fulltext(pdf_bytes: bytes) -> str:
    # 1) 文本抽取
    text = extract_pdf_text_by_pypdf(pdf_bytes)
    if len(text) >= 1000:  # 基本可用
        return text

    # 2) 兜底：视觉 OCR（最多 10 页）
    imgs_b64 = render_pdf_pages_to_png_b64(pdf_bytes, max_pages=10, dpi=180)
    if not imgs_b64:
        return text  # 空或极短
    ocr_text = await ocr_images_to_text(imgs_b64)
    # 合并文本/去重
    merged = (text + "\n" + ocr_text).strip() if text else ocr_text
    return merged

# ====== 图片 OCR 工作流 ======
async def process_image_message(user: str, media_id: str):
    try:
        await send_text(user, "已收到图片，开始识别文字与要点…")
        img_bytes = await download_wecom_media(media_id)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        raw_text = await ocr_images_to_text([b64])
        if not raw_text.strip():
            await send_text(user, "抱歉，未能从图片中识别到可读文字。")
            return
        summary = await summarize_long_text(raw_text, title="图片文字")
        if not summary.strip():
            summary = raw_text[:3500]
        await send_text(user, summary)
    except Exception as e:
        logger.exception(e)
        await send_text(user, "处理图片时出现异常，请稍后重试。")

# ====== PDF 全文解析工作流 ======
async def process_pdf_message(user: str, media_id: str, filename: str):
    try:
        await send_text(user, f"已收到 PDF《{filename}》，开始解析全文（完成后会推送结果）…")
        pdf_bytes = await download_wecom_media(media_id)
        full_text = await parse_pdf_fulltext(pdf_bytes)
        if not full_text.strip():
            await send_text(user, "未能从 PDF 中提取到可读内容，可能为扫描件或受保护文件。")
            return
        summary = await summarize_long_text(full_text, title=filename)
        if not summary.strip():
            summary = (full_text[:3500] + "\n\n（已截断，仅展示前 3500 字）")
        await send_text(user, summary)
    except Exception as e:
        logger.exception(e)
        await send_text(user, "解析 PDF 时出现异常，请稍后重试。")

# ====== FastAPI ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=JSONResponse)
async def root():
    status = {
        "status":"ok",
        "mode":"safe" if crypto else "plain",
        "service":"WeCom + ChatGPT",
        "model": OPENAI_MODEL,
        "memory": "redis" if MEM.use_redis else "memory",
        "pdf_support": True,
        "local_ocr": False,
    }
    return status

# ====== 企业微信回调 ======
@app.api_route("/wecom/callback", methods=["GET", "POST"])
async def wecom_callback(request: Request):
    # --- GET 校验/echo ---
    if request.method == "GET":
        args = request.query_params
        echostr = args.get("echostr")
        if echostr:
            if "msg_signature" in args and crypto:
                msg_signature = args.get("msg_signature", "")
                timestamp = args.get("timestamp", "")
                nonce = args.get("nonce", "")
                try:
                    echo = crypto.decrypt(echostr, msg_signature, timestamp, nonce)
                    return PlainTextResponse(echo)
                except Exception as e:
                    logger.error(f"wecom-app:URL verify decrypt failed: {e}")
                    return PlainTextResponse("invalid", status_code=400)
            return PlainTextResponse(echostr)
        return PlainTextResponse("ok")

    # --- POST 处理消息 ---
    body = await request.body()
    args = request.query_params
    is_safe_mode = "msg_signature" in args and crypto is not None

    raw_xml = ""
    if is_safe_mode:
        try:
            msg_signature = args.get("msg_signature", "")
            timestamp = args.get("timestamp", "")
            nonce = args.get("nonce", "")
            raw_xml = crypto.decrypt(body.decode("utf-8"), msg_signature, timestamp, nonce)
        except InvalidSignatureException:
            logger.error("InvalidSignatureException")
            return PlainTextResponse("signature error", status_code=403)
        except Exception as e:
            logger.error(f"decrypt fail: {e}")
            return PlainTextResponse("decrypt error", status_code=400)
    else:
        raw_xml = body.decode("utf-8")

    try:
        data = xmltodict.parse(raw_xml).get("xml", {})
    except Exception as e:
        logger.error(f"xml parse error: {e}")
        return PlainTextResponse("bad xml", status_code=400)

    from_user = data.get("FromUserName", "")
    msg_type = (data.get("MsgType", "") or "").lower()
    content = (data.get("Content", "") or "").strip()

    # 命令
    if content.startswith("/ping"):
        await send_text(from_user, "pong")
        return PlainTextResponse("success")

    if content.startswith("/model"):
        candidates = f"{OPENAI_MODEL}, gpt-5-mini, gpt-4o-mini"
        mem_kind = "redis" if MEM.use_redis else "memory"
        org = OPENAI_ORG_ID or "-"
        txt = (
            f"当前活跃模型：{OPENAI_MODEL}\n"
            f"候选列表：{candidates}\n"
            f"组织ID：{org}\n"
            f"记忆：{mem_kind}"
        )
        await send_text(from_user, txt)
        return PlainTextResponse("success")

    global web_enabled_flag
    if content.startswith("/web"):
        seg = content.split(None, 1)
        if len(seg) == 1:
            await send_text(from_user, "用法：/web on|off 或 /web 你的问题（临时联网）")
            return PlainTextResponse("success")
        arg = seg[1].strip()
        if arg in ("on","off","true","false","开启","关闭"):
            web_enabled_flag = arg in ("on","true","开启")
            await send_text(from_user, f"Web access 已{'开启' if web_enabled_flag else '关闭'}")
            return PlainTextResponse("success")
        content = arg

    # 文本
    if msg_type == "text":
        history = MEM.load(from_user)
        messages = [{"role":"system","content":"你是企业微信助手，回答简洁、可读性强；当被要求联网搜索时可引用结果给出简要结论和参考链接。"}]
        for r, c in history:
            messages.append({"role":r, "content":c})
        messages.append({"role":"user", "content":content})

        # 仅在有提供商 + 开关开启时传 tools（补丁①）
        use_web_tool = (web_enabled_flag and has_search_provider())
        tools: List[Dict[str, Any]] = []
        if use_web_tool:
            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web and synthesize a concise Chinese answer with 3-5 citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type":"string","description":"中文检索式"},
                            "top_k": {"type":"integer","minimum":1,"maximum":10,"default":3}
                        },
                        "required": ["query"]
                    }
                }
            }]

        reply_text = await ask_openai_text(messages, use_tools=use_web_tool, tools=tools)
        MEM.append(from_user, "user", content)
        if reply_text.strip():
            MEM.append(from_user, "assistant", reply_text)
        await send_text(from_user, reply_text)
        return PlainTextResponse("success")

    # 图片
    if msg_type == "image":
        media_id = data.get("MediaId", "")
        asyncio.create_task(process_image_message(from_user, media_id))
        return PlainTextResponse("success")

    # 文件（PDF）
    if msg_type == "file":
        filename = (data.get("FileName", "") or "")
        media_id = data.get("MediaId", "")
        if filename.lower().endswith(".pdf"):
            asyncio.create_task(process_pdf_message(from_user, media_id, filename))
            return PlainTextResponse("success")
        await send_text(from_user, f"目前仅支持 PDF 文件解析（收到：{filename}）。")
        return PlainTextResponse("success")

    await send_text(from_user, "暂时只支持文本 / 图片 / PDF。")
    return PlainTextResponse("success")

# ====== 关机清理 ======
@app.on_event("shutdown")
async def on_shutdown():
    await client.aclose()

# ====== 启动 ======
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
