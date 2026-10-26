from __future__ import annotations

import io
import os
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from .enhance import METHODS, enhance

MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MIN_SIDE = 16
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", 4096))
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 12_000_000))
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}

DEFAULT_ORIGINS = ["https://mmeraj1.github.io"]
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",") if o.strip()]

app = FastAPI(title="AI Image Enhancement", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Enhancement-Method"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/methods")
def methods() -> dict[str, list[str]]:
    return {"methods": list(METHODS)}


def _read_limited(upload: UploadFile) -> bytes:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    return data


def _decode(data: bytes) -> np.ndarray:
    # Pillow reads only the header here, so oversized images are rejected before full decoding.
    try:
        with PILImage.open(io.BytesIO(data)) as probe:
            fmt, (width, height) = probe.format, probe.size
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a valid image.") from None
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, "Unsupported image format. Use PNG, JPEG or WebP.")
    if min(width, height) < MIN_SIDE:
        raise HTTPException(422, f"Image is too small. Minimum size is {MIN_SIDE}px per side.")
    if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise HTTPException(422, f"Image is too large. Maximum is {MAX_SIDE}px per side and {MAX_PIXELS // 1_000_000} megapixels.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Image could not be decoded.")
    return image


@app.post("/enhance")
def enhance_image(file: Annotated[UploadFile, File()], method: Annotated[str, Form()] = "auto") -> Response:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type. Use PNG, JPEG or WebP.")
    method = method.strip().lower()
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}.")

    image = _decode(_read_limited(file))
    try:
        result = enhance(image, method)
        ok, buffer = cv2.imencode(".png", result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    except cv2.error:
        raise HTTPException(500, "Image processing failed.") from None
    if not ok:
        raise HTTPException(500, "Could not encode the enhanced image.")
    return Response(content=buffer.tobytes(), media_type="image/png", headers={"X-Enhancement-Method": method})
