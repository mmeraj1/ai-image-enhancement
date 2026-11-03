from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

TILE_SIZE = 256
TILE_OVERLAP = 32
COLORIZE_RENDER_FACTOR = 35


class ModelSetupError(RuntimeError):
    pass


def _cache_dir() -> Path:
    return Path(os.getenv("AI_IMAGE_MODEL_CACHE", Path.home() / ".cache" / "ai-image-enhancement")).expanduser()


_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_deblur_model: Any = None
_colorizer: Any = None


def _resnet101_without_pretraining(pretrained: bool = True) -> Any:
    from torchvision.models import resnet101

    return resnet101(weights=None)


_resnet101_without_pretraining.__name__ = "resnet101"


def _load_deblurgan_model() -> Any:
    global _deblur_model
    with _model_lock:
        if _deblur_model is not None:
            return _deblur_model

        model_path = _cache_dir() / "deblurgan_generator.keras"
        if not model_path.is_file():
            raise ModelSetupError("DeblurGAN-v2 weights are missing. Run `python scripts/setup_gan_models.py` in the backend repository.")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ModelSetupError("TensorFlow is not installed. Install `requirements-gan.txt` to enable DeblurGAN-v2.") from exc

        _deblur_model = tf.keras.models.load_model(model_path, compile=False)
        return _deblur_model


def _load_deoldify_colorizer() -> Any:
    global _colorizer
    with _model_lock:
        if _colorizer is not None:
            return _colorizer

        cache_dir = _cache_dir()
        source_dir = cache_dir / "DeOldify"
        weights_path = cache_dir / "models" / "ColorizeStable_gen.pth"
        if not source_dir.joinpath("deoldify").is_dir() or not weights_path.is_file():
            raise ModelSetupError("DeOldify source or weights are missing. Run `python scripts/setup_gan_models.py` in the backend repository.")

        source = str(source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from deoldify.filters import ColorizerFilter
            from deoldify.generators import gen_inference_wide
        except ImportError as exc:
            raise ModelSetupError("DeOldify dependencies are unavailable. Install `requirements-gan.txt` to enable colorization.") from exc

        learner = gen_inference_wide(
            root_folder=cache_dir,
            weights_name="ColorizeStable_gen",
            arch=_resnet101_without_pretraining,
        )
        _colorizer = ColorizerFilter(learn=learner)
        return _colorizer


def _tile_starts(length: int) -> list[int]:
    if length <= TILE_SIZE:
        return [0]
    stride = TILE_SIZE - TILE_OVERLAP
    starts = list(range(0, length - TILE_SIZE + 1, stride))
    final_start = length - TILE_SIZE
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _deblur_tiled(image_rgb: np.ndarray, model: Any) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    padded_height = max(height, TILE_SIZE)
    padded_width = max(width, TILE_SIZE)
    padded = cv2.copyMakeBorder(
        image_rgb,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REFLECT_101,
    )

    weight_1d = np.maximum(np.hanning(TILE_SIZE), 0.05).astype(np.float32)
    weights = np.outer(weight_1d, weight_1d)[:, :, None]
    merged = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    total_weight = np.zeros((padded_height, padded_width, 1), dtype=np.float32)

    for top in _tile_starts(padded_height):
        for left in _tile_starts(padded_width):
            tile = padded[top:top + TILE_SIZE, left:left + TILE_SIZE].astype(np.float32)
            batch = np.expand_dims(tile / 127.5 - 1.0, axis=0)
            prediction = model.predict(batch, verbose=0)
            if isinstance(prediction, (list, tuple)):
                if len(prediction) != 1:
                    raise RuntimeError("DeblurGAN-v2 returned an unexpected number of outputs.")
                prediction = prediction[0]
            prediction = np.asarray(prediction)
            if prediction.shape != (1, TILE_SIZE, TILE_SIZE, 3):
                raise RuntimeError("DeblurGAN-v2 returned an unexpected image shape.")
            tile_result = (prediction[0] + 1.0) * 127.5
            merged[top:top + TILE_SIZE, left:left + TILE_SIZE] += tile_result * weights
            total_weight[top:top + TILE_SIZE, left:left + TILE_SIZE] += weights

    result = merged / np.maximum(total_weight, 1e-6)
    return np.clip(result[:height, :width], 0, 255).astype(np.uint8)


def deblur(image_bgr: np.ndarray) -> np.ndarray:
    model = _load_deblurgan_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with _inference_lock:
        result_rgb = _deblur_tiled(image_rgb, model)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def colorize(image_bgr: np.ndarray) -> np.ndarray:
    colorizer = _load_deoldify_colorizer()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(image_rgb)
    with _inference_lock:
        result = colorizer.filter(
            source,
            source,
            render_factor=COLORIZE_RENDER_FACTOR,
            post_process=True,
        )
    return cv2.cvtColor(np.asarray(result.convert("RGB")), cv2.COLOR_RGB2BGR)
