# AI Image Enhancement

FastAPI + OpenCV service that enhances images. Used by the interactive demo in the
[portfolio](https://github.com/mmeraj1/portfolio) (`/projects/ai-image-enhancement`).

## Methods
| id | description |
|----|-------------|
| `auto` | Gamma brightness correction, then CLAHE, then a gentle unsharp mask |
| `clahe` | CLAHE contrast enhancement on the LAB lightness channel |
| `gamma` | Gamma correction; gamma is chosen automatically to move mean brightness to mid-grey |
| `sharpen` | Unsharp-mask sharpening |

## API
- `POST /enhance` — multipart form: `file` (PNG/JPEG/WebP) and `method`. Returns `image/png`.
  Errors: `400` bad image/method, `413` file too large, `415` unsupported type, `422` bad dimensions.
- `GET /methods`, `GET /health`

Limits (env-overridable): `MAX_UPLOAD_BYTES` (10 MB), `MAX_IMAGE_SIDE` (4096), `MAX_IMAGE_PIXELS` (12 MP); minimum 16 px per side.
CORS: `ALLOWED_ORIGINS` (comma separated, default `https://mmeraj1.github.io`) plus any `localhost`/`127.0.0.1` port.

## Run locally
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests -q
uvicorn app.main:app --reload
```

## Docker
```bash
docker build -t ai-image-enhancement . && docker run -p 7860:7860 ai-image-enhancement
```
The image works on any container host (Hugging Face Spaces Docker SDK, Render, Fly.io, Cloud Run).
