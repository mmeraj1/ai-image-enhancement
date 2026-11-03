import numpy as np

from app.gan import TILE_SIZE, _deblur_tiled, _tile_starts


class IdentityModel:
    def predict(self, batch, verbose=0):
        return batch


def test_tile_starts_cover_short_exact_and_overlapping_dimensions():
    assert _tile_starts(120) == [0]
    assert _tile_starts(TILE_SIZE) == [0]
    starts = _tile_starts(TILE_SIZE + 50)
    assert starts[0] == 0 and starts[-1] == 50
    assert all(right - left <= TILE_SIZE for left, right in zip(starts, starts[1:]))


def test_deblur_tiling_preserves_dimensions_and_near_identity_input():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (315, 410, 3), dtype=np.uint8)
    result = _deblur_tiled(image, IdentityModel())
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.abs(result.astype(np.int16) - image.astype(np.int16)).max() <= 1


def test_deblur_tiling_handles_images_smaller_than_model_tile():
    image = np.full((80, 96, 3), 127, dtype=np.uint8)
    result = _deblur_tiled(image, IdentityModel())
    assert result.shape == image.shape
    assert np.allclose(result, image, atol=1)
