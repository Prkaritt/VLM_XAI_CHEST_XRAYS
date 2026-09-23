"""Generate reusable blank image controls.

The generated controls are model-input-sized blank images that can be reused
across studies and questions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


DEFAULT_OUTPUT_DIR = Path("runs/controls")
DEFAULT_IMAGE_SIZE = 512
CONTROL_COLORS = {
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def generate_blank_controls(output_dir: Path, image_size: int) -> list[dict[str, str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for control_type, color in CONTROL_COLORS.items():
        filename = f"blank_{image_size}_{control_type}.png"
        image_path = output_dir / filename
        image = Image.new("RGB", (image_size, image_size), color=color)
        image.save(image_path)

        rows.append(
            {
                "control_type": control_type,
                "image_size": image_size,
                "red": color[0],
                "green": color[1],
                "blue": color[2],
                "image_path": str(image_path),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["control_type", "image_size", "red", "green", "blue", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable blank image controls.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = generate_blank_controls(
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
    print(f"Generated {len(rows)} blank control image(s).")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
