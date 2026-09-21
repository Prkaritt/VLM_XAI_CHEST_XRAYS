"""Generate patch-occluded image variants for causal mapping.

This module only prepares image perturbations. It does not run a VLM or compute
causal scores. The first target is CheXagent, whose visual pipeline resizes
images to 512x512 before encoding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_IMAGE_SIZE = 512
DEFAULT_GRID_SIZE = 16
DEFAULT_BLUR_RADIUS = 3
DEFAULT_FILL_COLOR = (128, 128, 128)


@dataclass(frozen=True)
class PatchRecord:
    patch_index: int
    row: int
    col: int
    grid_size: int
    image_size: int
    x1: int
    y1: int
    x2: int
    y2: int
    fill_color: tuple[int, int, int]
    blur_radius: int
    image_path: str


def parse_fill_color(fill: str) -> tuple[int, int, int]:
    """Parse a named or comma-separated RGB fill color."""
    normalized = fill.strip().lower()
    if normalized == "gray":
        return DEFAULT_FILL_COLOR
    if normalized == "black":
        return (0, 0, 0)
    if normalized == "white":
        return (255, 255, 255)

    parts = normalized.split(",")
    if len(parts) != 3:
        raise ValueError("Fill must be gray, black, white, or R,G,B.")

    rgb = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("Fill RGB values must be in [0, 255].")
    return rgb


def patch_indices(grid_size: int, limit: int | None = None) -> Iterable[tuple[int, int, int]]:
    """Yield row-major patch indices."""
    count = 0
    for row in range(grid_size):
        for col in range(grid_size):
            if limit is not None and count >= limit:
                return
            yield count, row, col
            count += 1


def build_patch_alpha(
    row: int,
    col: int,
    grid_size: int,
    image_size: int,
    blur_radius: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build alpha mask for one occluded patch.

    Returns alpha where 1 keeps the original pixel and 0 applies the fill.
    """
    if image_size % grid_size != 0:
        raise ValueError("image_size must be divisible by grid_size.")

    cell_size = image_size // grid_size
    x1 = col * cell_size
    y1 = row * cell_size
    x2 = x1 + cell_size
    y2 = y1 + cell_size

    alpha = np.ones((image_size, image_size), dtype=np.float32)
    alpha[y1:y2, x1:x2] = 0.0

    if blur_radius > 0:
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha_uint8, mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha = np.array(alpha_img).astype(np.float32) / 255.0

    return alpha, (x1, y1, x2, y2)


def apply_alpha_fill(
    image: Image.Image,
    alpha: np.ndarray,
    fill_color: tuple[int, int, int],
) -> Image.Image:
    """Blend an image with a solid fill using alpha."""
    image_arr = np.array(image).astype(np.float32)
    fill_arr = np.full_like(image_arr, fill_color, dtype=np.float32)
    alpha3 = alpha[:, :, None]
    out = image_arr * alpha3 + fill_arr * (1.0 - alpha3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def prepare_image(image_path: Path, image_size: int) -> Image.Image:
    """Load and resize an image to the target square size."""
    image = Image.open(image_path).convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def generate_patch_occlusions(
    image_path: Path,
    output_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    fill_color: tuple[int, int, int] = DEFAULT_FILL_COLOR,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    limit: int | None = None,
) -> list[PatchRecord]:
    """Generate patch-occluded images and a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    occluded_dir = output_dir / "occluded"
    occluded_dir.mkdir(parents=True, exist_ok=True)

    image = prepare_image(image_path, image_size)
    resized_path = output_dir / "original_resized.png"
    image.save(resized_path)

    records: list[PatchRecord] = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for patch_index, row, col in patch_indices(grid_size, limit=limit):
            alpha, (x1, y1, x2, y2) = build_patch_alpha(
                row=row,
                col=col,
                grid_size=grid_size,
                image_size=image_size,
                blur_radius=blur_radius,
            )
            occluded = apply_alpha_fill(image, alpha, fill_color)
            patch_name = f"patch_{patch_index:03d}_r{row:02d}_c{col:02d}.png"
            patch_path = occluded_dir / patch_name
            occluded.save(patch_path)

            record = PatchRecord(
                patch_index=patch_index,
                row=row,
                col=col,
                grid_size=grid_size,
                image_size=image_size,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                fill_color=fill_color,
                blur_radius=blur_radius,
                image_path=str(patch_path),
            )
            records.append(record)
            manifest.write(json.dumps(asdict(record)) + "\n")

    metadata = {
        "source_image": str(image_path),
        "resized_image": str(resized_path),
        "manifest": str(manifest_path),
        "occluded_dir": str(occluded_dir),
        "image_size": image_size,
        "grid_size": grid_size,
        "patch_count": len(records),
        "total_possible_patches": grid_size * grid_size,
        "fill_color": fill_color,
        "blur_radius": blur_radius,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patch-occluded image variants for causal mapping."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument("--grid-size", default=DEFAULT_GRID_SIZE, type=int)
    parser.add_argument("--fill", default="gray", help="gray, black, white, or R,G,B.")
    parser.add_argument("--blur-radius", default=DEFAULT_BLUR_RADIUS, type=int)
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Generate only the first N row-major patches for visual debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    fill_color = parse_fill_color(args.fill)
    records = generate_patch_occlusions(
        image_path=image_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        fill_color=fill_color,
        blur_radius=args.blur_radius,
        limit=args.limit,
    )

    print(f"Generated {len(records)} patch-occluded image(s).")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
