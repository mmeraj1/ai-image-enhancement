from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SIZE = (1600, 1200)
FRAME_CORNERS = np.array(
    [(158, 45), (1432, 6), (1493, 993), (115, 1001)],
    dtype=np.float32,
)
RESTORED_SIZE = (1600, 1200)


def crop_to_photo(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    reference_width, reference_height = REFERENCE_SIZE
    corners = FRAME_CORNERS * np.array(
        [width / reference_width, height / reference_height],
        dtype=np.float32,
    )
    target_width, target_height = RESTORED_SIZE
    target = np.array(
        [
            (0, 0),
            (target_width - 1, 0),
            (target_width - 1, target_height - 1),
            (0, target_height - 1),
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(
        image,
        transform,
        RESTORED_SIZE,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def improve_photo(image: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=8,
        hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(lightness)
    balanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    softened_background = cv2.bilateralFilter(
        balanced,
        d=9,
        sigmaColor=60,
        sigmaSpace=60,
    )
    dark_area = np.clip((110 - lightness.astype(np.float32)) / 45, 0, 1)
    background_mask = cv2.GaussianBlur(dark_area, (0, 0), sigmaX=12)[..., None]
    blended = (
        balanced.astype(np.float32) * (1 - 0.75 * background_mask)
        + softened_background.astype(np.float32) * (0.75 * background_mask)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop the family photo to its print and gently clean up the scan."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "family_photo.jpeg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "test_outputs"
        / "family_photo_cropped_restored.png",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read input image: {args.input}")

    restored = improve_photo(crop_to_photo(image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), restored):
        raise SystemExit(f"Could not write restored image: {args.output}")
    print(f"Saved cropped and cleaned photo to {args.output}")


if __name__ == "__main__":
    main()
