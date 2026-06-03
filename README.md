# Asset Analysis API

Backend for asset validation (`multiple-image-validation-backend`).

Production-ready FastAPI service that accepts **one** image of a physical asset, applies light preprocessing (EXIF, resize), and extracts structured metadata via a single Google Gemini call.

## Features

- **Direct image analysis**: One photo preprocessed and sent to Gemini (no server-side collage or TAG ZOOM)
- **Single Gemini call**: Asset understanding and tag/barcode OCR in one request
- **Async endpoint**: Optional job-based analysis at `POST /v1/assets/analyze/async`
- **Observability**: Structured JSON logs, Prometheus metrics at `/metrics`

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
copy .env.example .env     # Set GEMINI_API_KEY

# LAN + local (required for other devices on same Wi‑Fi)
python serve.py
# or: .\run.ps1

# Local-only (other devices CANNOT connect — do not use for Wi‑Fi sharing)
# uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for interactive API documentation.

### Use from other devices on the same Wi‑Fi

1. Start the server with **`.\run.ps1`** (binds `0.0.0.0`, not only `127.0.0.1`).
2. On the PC, open http://localhost:8000/ — the JSON shows **`lan_base_url`** (e.g. `http://192.168.1.42:8000`).
3. On phones/tablets on the same network, use that URL in the UI or Swagger: `http://<your-pc-ip>:8000/docs`.
4. **Windows Firewall:** if requests fail, allow **Python** or **port 8000** for private networks when prompted.
5. In your UI JavaScript, set the API base to the LAN URL, not `localhost`:
   ```javascript
   const API_BASE = "http://192.168.1.42:8000"; // your PC's Wi‑Fi IP from GET /
   ```
6. CORS is already `allow_origins=["*"]`, so browsers on other devices can call the API.

### UI contract

**File upload:**

```javascript
const form = new FormData();
form.append("image", file);
await fetch(`${API_BASE}/v1/assets/analyze`, { method: "POST", body: form });
```

**Base64** (send one of `image` or `image_base64`, not both):

```javascript
const form = new FormData();
form.append("image_base64", dataUrlOrRawBase64);
await fetch(`${API_BASE}/v1/assets/analyze`, { method: "POST", body: form });
```

Data URLs are supported, e.g. `data:image/jpeg;base64,/9j/4AAQ...`. **`image_base64` has no length cap for now** (file upload still respects `MAX_IMAGE_SIZE_MB`).

### "Refused to connect" from phone

| Check | Fix |
|--------|-----|
| Server only on `127.0.0.1` | Stop uvicorn (Ctrl+C). Double-click **`START-API.bat`** or run `python serve.py` |
| Wrong URL on phone | Use `http://192.168.x.x:8000` from `GET /` on the PC — not `localhost` |
| Windows Firewall | Run **`scripts\open-firewall.ps1`** as Administrator |
| Wi‑Fi is "Public" | Settings → Network → Wi‑Fi → your network → **Private** |
| Guest Wi‑Fi | Router guest networks often block device-to-device; use main Wi‑Fi |

Verify on the PC after starting with `python serve.py`:

```powershell
netstat -an | findstr ":8000"
```

Must show **`0.0.0.0:8000`** LISTENING. If you only see **`127.0.0.1:8000`**, LAN will not work.

## API Endpoints (Swagger: http://localhost:8000/docs)

| Endpoint | Purpose |
|---|---|
| `POST /v1/assets/analyze` | Upload one image → preprocess → single Gemini extraction |
| `POST /v1/assets/analyze/async` | Same as analyze, returns job id for polling |
| `GET /v1/health` | Health check |

### Swagger upload

1. Open **/docs**
2. Expand **Analysis** → `POST /v1/assets/analyze`
3. Click **Try it out**
4. Choose one file for **image**
5. Execute — JSON includes asset fields and image dimensions (no image file in response)

## API Usage (curl)

```bash
# File upload
curl -X POST "http://localhost:8000/v1/assets/analyze" \
  -F "image=@photo.jpg"

# Base64 (raw or data URL)
curl -X POST "http://localhost:8000/v1/assets/analyze" \
  -F "image_base64=$(base64 -w0 photo.jpg)"
```

## Deploy to Vercel

1. Push this repo to GitHub (see below).
2. Import project in [Vercel](https://vercel.com) → **Add New Project** → select `multiple-image-validation-backend`.
3. **Environment variables** (Project → Settings → Environment Variables):
   - `GEMINI_API_KEY` — required
   - `GEMINI_MODEL` — e.g. `gemini-2.0-flash`
   - Other keys from [`.env.example`](.env.example) as needed
4. Deploy. API base URL: `https://<your-project>.vercel.app`

**Note:** `/v1/assets/analyze` can take 20–60 seconds. In Vercel → Project → Settings → Functions, set **Max Duration** (e.g. 60s) and **Memory** (max 2048 MB on Hobby). If requests time out, use Docker/Railway/Render instead.

**Deploy uses** `pyproject.toml` → `[tool.vercel] entrypoint = "app.main:app"`. Do not add `api/index.py` in `vercel.json` `functions` — that causes build errors on current Vercel CLI.

```bash
# CLI deploy (after npm i -g vercel)
vercel link
vercel env add GEMINI_API_KEY
vercel --prod
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: asset analysis API"
git remote add origin https://github.com/aniketv31/multiple-image-validation-backend.git
git branch -M main
git push -u origin main
```

Never commit `.env` — it is in `.gitignore`.

## Docker

```bash
docker compose up --build
```

## Configuration

See [.env.example](.env.example) for all settings.

| Setting | Default | Purpose |
|---|---|---|
| `MAX_PREPROCESS_EDGE_PX` | 2048 | Max longest edge before Gemini |
| `MAX_IMAGE_SIZE_MB` | 7 | Upload size limit |
| `GEMINI_ANALYZE_TEMPERATURE` | 0.0 | Temperature for analyze call |

### Prompts

All Gemini instructions for `/assets/analyze` live in [`app/prompts/analysis.txt`](app/prompts/analysis.txt). It covers asset identification, condition, and barcode OCR.

## Development

```bash
pytest
```

## License

MIT
