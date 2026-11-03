import numpy as np
import cv2

from scripts.restore_framed_photo import (
    FRAME_CORNERS,
    crop_to_photo,
    improve_photo,
)


def test_crop_removes_outside_frame_and_rectifies_to_photo_size():
    image = np.full((1200, 1600, 3), 10, dtype=np.uint8)
    cv2.fillConvexPoly(image, FRAME_CORNERS.astype(np.int32), (100, 100, 100))
    result = crop_to_photo(image)
    assert result.shape == (1200, 1600, 3)
    assert np.abs(result[600, 800].astype(int) - 100).max() <= 1
    assert np.abs(result[0, 0].astype(int) - 100).max() <= 1


def test_photo_cleanup_preserves_dimensions_and_dtype():
    image = np.full((120, 160, 3), 120, dtype=np.uint8)
    result = improve_photo(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
