"""Generate a high-contrast digit 7 that the bundled CNN classifies as 7."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


APP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).with_name("demo_digit_7.png")
sys.path.insert(0, str(APP_DIR))

from app import load_model, predict_probabilities, preprocess_image  # noqa: E402


def candidate_images():
    for width in (12, 16, 20, 24, 28, 32):
        for left in (45, 60, 75):
            for diagonal_end in (95, 115, 135):
                image = Image.new("L", (280, 280), 255)
                draw = ImageDraw.Draw(image)
                draw.line((left, 55, 235, 55), fill=0, width=width)
                draw.line((230, 55, diagonal_end, 235), fill=0, width=width)
                yield image

                crossed = image.copy()
                ImageDraw.Draw(crossed).line(
                    (125, 145, 190, 145),
                    fill=0,
                    width=max(8, width // 2),
                )
                yield crossed


def main() -> None:
    model, _ = load_model()
    best_image = None
    best_score = -1.0
    best_prediction = None

    for image in candidate_images():
        tensor, _ = preprocess_image(image)
        probabilities = predict_probabilities(model, tensor)
        score = float(probabilities[7])
        if score > best_score:
            best_image = image
            best_score = score
            best_prediction = int(probabilities.argmax())

    if best_image is None or best_prediction != 7:
        raise RuntimeError(
            f"Could not generate a reliable digit 7 sample (best score={best_score:.3f})."
        )

    best_image.save(OUTPUT_PATH)
    print(f"saved {OUTPUT_PATH.name}: predicted=7, confidence={best_score:.3f}")


if __name__ == "__main__":
    main()
