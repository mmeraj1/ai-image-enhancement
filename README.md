---
title: AI Image Enhancement
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI Image Enhancement

FastAPI image enhancement service used by the interactive demo in the
[portfolio](https://github.com/mmeraj1/portfolio) (`/projects/ai-image-enhancement`).
It includes lightweight OpenCV filters and optional local GAN inference.

## Methods
| id | description |
|----|-------------|
| `auto` | Gamma brightness correction, then CLAHE, then a gentle unsharp mask |
| `clahe` | CLAHE contrast enhancement on the LAB lightness channel |
| `gamma` | Gamma correction; gamma is chosen automatically to move mean brightness to mid-grey |
| `sharpen` | Unsharp-mask sharpening |
| `deblur` | DeblurGAN-v2 generator for motion blur; processes 256px tiles with overlap |
| `colorize` | DeOldify Stable GAN for plausible black-and-white photo colorization |

GAN inference runs inside this service rather than being forwarded to a third-party model provider. Uploaded images are still sent to the configured backend host. GANs may alter or invent details; the colorizer estimates colors and cannot recover historically accurate colors.

## Set up local GAN inference

GAN dependencies and weights are optional. The ordinary OpenCV API remains lightweight.
The local setup requires Python 3.10, Git, and several GB of free disk space. The
DeOldify checkpoint is about 874 MB; model files are stored outside the repository in
`~/.cache/ai-image-enhancement` and are not committed.

From the portfolio root:

```bash
cd ai-image-enhancement
python -m venv .venv-gan
. .venv-gan/bin/activate
python -m pip install -r requirements.txt -r requirements-gan.txt
python scripts/setup_gan_models.py
python -m uvicorn app.main:app --reload
```

To keep models somewhere else, set `AI_IMAGE_MODEL_CACHE` before setup and startup.
The setup script verifies SHA-256 hashes for downloaded weights and pins the DeOldify
source revision. If GAN dependencies or weights are missing, GAN requests return
`503` with setup guidance; the OpenCV methods continue to work.

Model sources: [DeblurGAN-V2 Keras checkpoint](https://huggingface.co/TJ-27/Deblur-GAN-V2),
[DeOldify source](https://github.com/jantic/DeOldify), and a
[DeOldify Stable checkpoint mirror](https://huggingface.co/databuzzword/deoldify-stable).
Weights are not committed; review the checkpoint mirror's terms before redistributing
or hosting the models.

Test a local image without uploading it to any hosted API:

```bash
python scripts/enhance_image.py ../family_photo.jpeg \
  --method deblur --output test_outputs/family_photo_deblurred.png
python scripts/enhance_image.py ../family_photo.jpeg \
  --method colorize --grayscale --output test_outputs/family_photo_colorized.png
python scripts/restore_framed_photo.py
```

`restore_framed_photo.py` rectifies and crops the photographed print to its four
image edges, then reduces scan grain and gently evens the dark background. It saves
`test_outputs/family_photo_cropped_restored.png` without changing the source file.

## API
- `POST /enhance` — multipart form: `file` (PNG/JPEG/WebP) and `method`. Returns `image/png`.
  Errors: `400` bad image/method, `413` file too large, `415` unsupported type, `422` bad dimensions, `503` optional GAN model not set up.
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
The default image includes only the lightweight OpenCV methods. To deploy GAN inference,
install the optional dependencies and weights on a host with enough memory and storage.

## Hosting on Hugging Face Spaces (Docker)
The `Dockerfile` builds a CPU image (Python 3.11, `tensorflow-cpu`) that serves **DeblurGAN-v2** only:
the 46 MB weights are downloaded and SHA-256 verified at build time, so nothing large is committed.
DeOldify colorization is excluded from the hosted image (it needs PyTorch and ~870 MB of weights);
`/methods` and `/health` report which models are actually available.

1. Create a Space at https://huggingface.co/new-space, SDK **Docker**, hardware **CPU basic** (free: 2 vCPU, 16 GB RAM).
2. Push this repository to the Space's git remote (the README front matter configures `app_port: 7860`).
3. In the Space settings add the variable `ALLOWED_ORIGINS=https://mmeraj1.github.io`.
4. Set the portfolio's GitHub Actions variable `VITE_ENHANCE_API_URL` to `https://<user>-<space>.hf.space` and re-run the deploy workflow.
