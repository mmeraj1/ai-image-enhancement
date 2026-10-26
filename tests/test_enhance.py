import numpy as np
import pytest

from app.enhance import METHODS, auto, clahe, enhance, gamma, sharpen


def dark_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 60, (64, 80, 3), dtype=np.uint8)


def test_all_methods_preserve_shape_and_dtype():
    img = dark_image()
    for name in METHODS:
        out = enhance(img, name)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_gamma_brightens_dark_image():
    img = dark_image()
    assert gamma(img).mean() > img.mean() + 20


def test_gamma_darkens_bright_image():
    img = 255 - dark_image()
    assert gamma(img).mean() < img.mean() - 20


def test_clahe_increases_contrast():
    rng = np.random.default_rng(1)
    img = rng.integers(100, 130, (64, 64, 3), dtype=np.uint8)
    assert clahe(img).std() > img.std()


def test_sharpen_increases_edge_strength():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, 20:] = 120
    out = sharpen(img).astype(int)
    assert out.max() > 120 and out.min() == 0
    assert np.abs(np.diff(out[:, :, 0], axis=1)).max() >= np.abs(np.diff(img[:, :, 0].astype(int), axis=1)).max()


def test_auto_changes_image():
    img = dark_image()
    assert not np.array_equal(auto(img), img)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        enhance(dark_image(), "nope")
