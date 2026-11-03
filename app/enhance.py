"""OpenCV image enhancement methods. All functions take and return BGR uint8 arrays."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .gan import ModelSetupError, colorize, deblur

Image = np.ndarray


def clahe(image: Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image:
    """Contrast Limited Adaptive Histogram Equalisation on the lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    equaliser = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    merged = cv2.merge((equaliser.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma(image: Image, value: float | None = None) -> Image:
    """Gamma correction. With no value, gamma is chosen so the mean brightness moves towards mid-grey."""
    if value is None:
        mean = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) / 255.0
        mean = min(max(mean, 0.02), 0.98)
        value = float(np.clip(np.log(mean) / np.log(0.5), 0.4, 2.5))
    table = (np.linspace(0, 1, 256) ** (1.0 / value) * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def sharpen(image: Image, amount: float = 1.0, sigma: float = 1.5) -> Image:
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def auto(image: Image) -> Image:
    """Brightness correction, then local contrast, then a gentle sharpen."""
    return sharpen(clahe(gamma(image), clip_limit=1.5), amount=0.5)


METHODS: dict[str, Callable[[Image], Image]] = {
    "auto": auto,
    "clahe": clahe,
    "gamma": gamma,
    "sharpen": sharpen,
    "deblur": deblur,
    "colorize": colorize,
}


def enhance(image: Image, method: str) -> Image:
    try:
        return METHODS[method](image)
    except KeyError:
        raise ValueError(f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}") from None
