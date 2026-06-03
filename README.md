# Multi-Image Asset Analysis API

Production-ready FastAPI service that accepts 2–10 images of a physical asset, builds a high-resolution analysis contact sheet (HRC), and extracts structured metadata via a single Google Gemini call.

## Features

- **Collage contact sheet (HRC)**: One canvas with labeled angle boxes (aspect-fit, no warp) plus a TAG ZOOM row (raw + enhanced ROI) — one JPEG sent to Gemini
- **Single Gemini call**: Asset understanding, tag/barcode OCR, and optional name validation in one API request (no second tag pass)
- **Panorama preview**: `/panorama` defaults to the same collage layout; optional `layout=stitch` or `grid`. Side/tag angles are not stitched (avoids 160×120 warp)
- **Tag view selection**: Optional `tag_image_index` form field, angle hints (`tag`, `barcode`, …), or edge-density heuristic for TAG ZOOM crops
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
| `POST /v1/assets/panorama` | Upload multiple images → unified panorama/grid preview only |
| `POST /v1/assets/analyze` | Upload multiple images → HRC contact sheet + single Gemini extraction |

### Swagger multi-image upload

1. Open **/docs**
2. Expand **Panorama** or **Analysis** endpoint
3. Click **Try it out**
4. For **images**, click **Add item** once per photo (2–10 rows)
5. Optionally set **angles** e.g. `Front,Back,Left,Right`
6. Execute — response includes `image_base64` for inline preview (images are not stored on disk)

## API Usage (curl)

```bash
# Preview panorama only
curl -X POST "http://localhost:8000/v1/assets/panorama" \
  -F "images=@front.jpg" \
  -F "images=@back.jpg" \
  -F "images=@left.jpg" \
  -F "angles=Front,Back,Left" \
  -F "layout=collage" \
  -F "include_image_base64=true"

# Full Gemini analysis (single composite + single call)
curl -X POST "http://localhost:8000/v1/assets/analyze" \
  -F "images=@front.jpg" \
  -F "images=@back.jpg" \
  -F "images=@left.jpg" \
  -F "angles=Front,Back,Left" \
  -F "tag_image_index=2"
```

## Deploy to Vercel

1. Push this repo to GitHub (see below).
2. Import project in [Vercel](https://vercel.com) → **Add New Project** → select `multiple-image-validation-backend`.
3. **Environment variables** (Project → Settings → Environment Variables):
   - `GEMINI_API_KEY` — required
   - `GEMINI_MODEL` — e.g. `gemini-2.0-flash`
   - Other keys from [`.env.example`](.env.example) as needed
4. Deploy. API base URL: `https://<your-project>.vercel.app`

**Note:** `/v1/assets/analyze` can take 20–60 seconds. `vercel.json` sets `maxDuration: 60` (requires Vercel **Pro** on many plans; Hobby is often 10s). If requests time out, use Docker/Railway/Render instead.

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
git commit -m "Initial commit: multi-image asset analysis API"
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

Key analysis composite settings:

| Setting | Default | Purpose |
|---|---|---|
| `MAX_GEMINI_COMPOSITE_BYTES` | 6500000 | JPEG size budget (under Gemini 7MB limit) |
| `MAX_COMPOSITE_WIDTH_PX` | 4096 | Max canvas width |
| `ANALYSIS_MIN_CELL_PX` | 640 | Minimum panel size |
| `ANALYSIS_MAX_CELL_PX` | 1280 | Maximum panel size |
| `COLLAGE_TAG_SLOT_SCALE` | 1.35 | Tag collage cell width multiplier |
| `TAG_ZOOM_ROW_MIN_PX` | 400 | Minimum height for TAG ZOOM row |
| `GEMINI_TAG_MAX_EDGE_PX` | 2048 | Max edge for tag ROI before zoom row |
| `PANORAMA_DEFAULT_LAYOUT` | collage | Panorama preview layout |
| `GEMINI_ANALYZE_TEMPERATURE` | 0.0 | Temperature for analyze call |

### Prompts

All Gemini instructions for `/assets/analyze` live in a single file: [`app/prompts/analysis.txt`](app/prompts/analysis.txt). It covers asset identification, condition, barcode OCR, and optional name validation (injected at runtime when `asset_name` is provided).

## Development

```bash
pytest
```

## License

MIT
