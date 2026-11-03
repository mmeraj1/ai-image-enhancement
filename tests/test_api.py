import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.enhance import METHODS, ModelSetupError

client = TestClient(main.app)


def encode(ext: str = ".png", size=(64, 48), dtype_max=80) -> bytes:
    rng = np.random.default_rng(0)
    img = rng.integers(0, dtype_max, (size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def post(data: bytes, method="auto", content_type="image/png", name="x.png"):
    return client.post("/enhance", files={"file": (name, data, content_type)}, data={"method": method})


def test_health_and_methods():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/methods").json()["methods"] == list(METHODS)


@pytest.mark.parametrize("method", ("auto", "clahe", "gamma", "sharpen"))
def test_enhance_each_method_returns_png_of_same_size(method):
    r = post(encode(), method)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-enhancement-method"] == method
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (48, 64, 3)


def test_gan_model_setup_error_is_reported_as_service_unavailable(monkeypatch):
    def missing_model(image, method):
        raise ModelSetupError("GAN weights are not installed.")

    monkeypatch.setattr(main, "enhance", missing_model)
    r = post(encode(), "deblur")
    assert r.status_code == 503
    assert r.json()["detail"] == "GAN weights are not installed."


def test_jpeg_and_webp_accepted():
    assert post(encode(".jpg"), content_type="image/jpeg", name="a.jpg").status_code == 200
    assert post(encode(".webp"), content_type="image/webp", name="a.webp").status_code == 200


def test_unknown_method_rejected():
    r = post(encode(), "magic")
    assert r.status_code == 400 and "Unknown method" in r.json()["detail"]


def test_unsupported_content_type_rejected():
    assert post(b"hello", content_type="text/plain", name="a.txt").status_code == 415


def test_gif_declared_as_png_rejected_by_content_sniffing():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, "GIF")
    assert post(buf.getvalue()).status_code == 415


def test_corrupt_image_rejected():
    assert post(b"not an image at all").status_code == 400


def test_empty_file_rejected():
    assert post(b"").status_code == 400


def test_too_large_file_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 100)
    assert post(encode()).status_code == 413


def test_dimensions_too_large_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_SIDE", 32)
    assert post(encode()).status_code == 422


def test_dimensions_too_small_rejected():
    assert post(encode(size=(8, 8))).status_code == 422


def test_missing_file_rejected():
    assert client.post("/enhance", data={"method": "auto"}).status_code == 422


def test_cors_allows_portfolio_and_localhost_only():
    for origin in ("https://mmeraj1.github.io", "http://localhost:5173"):
        r = client.options("/enhance", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") == origin
    r = client.options("/enhance", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
