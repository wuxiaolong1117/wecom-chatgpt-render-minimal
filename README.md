
# WeCom + Render + ChatGPT (Minimal, Python/FastAPI)

A tiny webhook that connects **WeCom (企业微信)** to **OpenAI ChatGPT** and deploys on **Render**.

> ✅ For first deploy, set your WeCom app **Message Encryption** to **明文模式 (Plaintext)**. You can switch to 安全模式 later.
>
> ✅ This repo handles URL verification (GET echo) and incoming text messages, calls OpenAI, and replies back to the user.

---

## 1) What you get

- `/wecom/callback` **GET**: echoes `echostr` for WeCom URL verification
- `/wecom/callback` **POST**: receives WeCom XML, calls OpenAI, and sends a text reply
- Minimal caching of WeCom `access_token`
- Ready for **Render** deploy (Dockerfile or build/start cmd)

---

## 2) Environment variables

Create `.env` from `.env.example` and fill values:

```bash
OPENAI_API_KEY=sk-...
WEWORK_CORP_ID=ww...
WEWORK_AGENT_ID=1000002
WEWORK_SECRET=...
```

Optional (defaults provided):
```bash
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 3) Local run (optional)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

---

## 4) Deploy on Render

1. Push this repo to GitHub.
2. In **Render Dashboard → New → Web Service**:
   - Select your repo
   - Runtime: **Docker** (or use Build Command/Start Command without Dockerfile)
   - Region: closest to your WeCom users
   - Add **Environment Variables**: `OPENAI_API_KEY, WEWORK_CORP_ID, WEWORK_AGENT_ID, WEWORK_SECRET`
   - If not using Dockerfile, use:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. After deploy, you’ll have a public URL like `https://xxx.onrender.com`

---

## 5) Configure WeCom (企业微信)

1. 管理后台 → **应用管理 → 自建应用 → 功能 → 接收消息**（或消息回调）
2. 回调 URL：`https://<your-render-domain>/wecom/callback`
3. **消息加密方式**：选择 **明文模式**（最简）。
4. 点“保存”。WeCom 会向回调 URL 发送 **GET 验证**，本服务会原样返回 `echostr`。

> 若选择“安全模式”，需实现加解密（AES+签名）。最小模板未包含，后续可扩展。

---

## 6) Test

在企业微信里向你的自建应用发送文字消息，几秒后将收到 ChatGPT 的回复。

---

## 7) Extend ideas

- 🔁 **会话记忆**：将每位用户的最近消息保存在 SQLite/Redis 并追加到 `messages` 中。
- 🧠 **模型选择**：用 `OPENAI_MODEL` 切换到更强模型（如 `gpt-5-mini`, `gpt-5`）。
- 🎤 **语音**：接入语音转文字（ASR）与 TTS。
- 👥 **群聊**：仅在被 @ 时响应。
- 🔒 **安全模式**：接入企业微信官方加解密 SDK。

---

## 8) References

- OpenAI API docs (Chat Completions / Responses). See official docs for latest Python usage.
- WeCom message send/gettoken API — see WeCom official developer docs.
