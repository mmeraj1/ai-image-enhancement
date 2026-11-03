from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enhance import METHODS, enhance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an image enhancement method locally.")
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grayscale", action="store_true", help="Convert the input to grayscale before colorization.")
    args = parser.parse_args()

    if args.grayscale and args.method != "colorize":
        parser.error("--grayscale can only be used with --method colorize")

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read input image: {args.input}")
    if args.grayscale:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    result = enhance(image, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise SystemExit(f"Could not write output image: {args.output}")
    print(f"Saved {args.method} result to {args.output}")


if __name__ == "__main__":
    main()
