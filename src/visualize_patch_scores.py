"""Visualize patch-occlusion support drops as a causal heatmap."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUTPUT_DIR = Path("runs/visualizations/patch_scores")
GRID_FILENAME = "causal_grid.png"
OVERLAY_FILENAME = "causal_overlay.png"


@dataclass(frozen=True)
class PatchScore:
    patch_index: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    support_drop: float
    patch_image_path: Path


def read_scores(scores_csv: Path) -> list[PatchScore]:
    with scores_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        scores = [
            PatchScore(
                patch_index=int(row["patch_index"]),
                row=int(row["row"]),
                col=int(row["col"]),
                x1=int(row["x1"]),
                y1=int(row["y1"]),
                x2=int(row["x2"]),
                y2=int(row["y2"]),
                support_drop=float(row["support_drop"]),
                patch_image_path=Path(row["patch_image_path"]),
            )
            for row in reader
        ]
    if not scores:
        raise ValueError(f"No patch scores found in: {scores_csv}")
    return scores


def infer_original_image(scores: list[PatchScore]) -> Path:
    first_patch_path = scores[0].patch_image_path.expanduser().resolve()
    candidate = first_patch_path.parent.parent / "original_resized.png"
    if not candidate.exists():
        raise FileNotFoundError(
            "Could not infer original resized image. Pass --image explicitly."
        )
    return candidate


def color_for_value(value: float, max_abs: float) -> tuple[int, int, int]:
    """Map negative values to blue, zero to white, positive values to red."""
    if max_abs <= 0:
        return (255, 255, 255)

    scaled = max(-1.0, min(1.0, value / max_abs))
    if scaled >= 0:
        intensity = int(round(255 * scaled))
        return (255, 255 - intensity, 255 - intensity)

    intensity = int(round(255 * abs(scaled)))
    return (255 - intensity, 255 - intensity, 255)


def readable_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def make_grid_image(scores: list[PatchScore], output_path: Path, cell_size: int = 120) -> None:
    max_row = max(score.row for score in scores)
    max_col = max(score.col for score in scores)
    grid_rows = max_row + 1
    grid_cols = max_col + 1
    max_abs = max(abs(score.support_drop) for score in scores)

    image = Image.new("RGB", (grid_cols * cell_size, grid_rows * cell_size), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    small_font = load_font(14)

    for score in scores:
        x1 = score.col * cell_size
        y1 = score.row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        color = color_for_value(score.support_drop, max_abs)
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=(40, 40, 40), width=2)
        text_color = readable_text_color(color)
        draw_centered_text(
            draw,
            (x1, y1 + 12, x2, y2 - 18),
            f"{score.support_drop:.3f}",
            font,
            text_color,
        )
        draw_centered_text(
            draw,
            (x1, y2 - 32, x2, y2 - 6),
            f"r{score.row} c{score.col}",
            small_font,
            text_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_overlay_image(
    scores: list[PatchScore],
    image_path: Path,
    output_path: Path,
    alpha: float,
) -> None:
    max_abs = max(abs(score.support_drop) for score in scores)
    max_x = max(score.x2 for score in scores)
    max_y = max(score.y2 for score in scores)

    base = Image.open(image_path).convert("RGB").resize((max_x, max_y), Image.Resampling.BICUBIC)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(18)

    for score in scores:
        color = color_for_value(score.support_drop, max_abs)
        fill = (*color, int(round(255 * alpha)))
        draw.rectangle((score.x1, score.y1, score.x2, score.y2), fill=fill)
        draw.rectangle((score.x1, score.y1, score.x2, score.y2), outline=(255, 255, 255, 180), width=2)
        draw_centered_text(
            draw,
            (score.x1, score.y1, score.x2, score.y2),
            f"{score.support_drop:.2f}",
            font,
            (255, 255, 255, 230),
        )

    blended = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grid and overlay visualizations for patch-occlusion scores."
    )
    parser.add_argument("--scores-csv", required=True, type=Path)
    parser.add_argument(
        "--image",
        default=None,
        type=Path,
        help="Original/resized image for overlay. If omitted, infer original_resized.png from patch paths.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--overlay-alpha", default=0.45, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores_csv = args.scores_csv.expanduser().resolve()
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV does not exist: {scores_csv}")

    scores = read_scores(scores_csv)
    image_path = (
        args.image.expanduser().resolve()
        if args.image is not None
        else infer_original_image(scores)
    )

    if args.overlay_alpha < 0 or args.overlay_alpha > 1:
        raise ValueError("--overlay-alpha must be in [0, 1].")
    if not image_path.exists():
        raise FileNotFoundError(f"Overlay image does not exist: {image_path}")

    output_dir = args.output_dir.expanduser()
    grid_path = output_dir / GRID_FILENAME
    overlay_path = output_dir / OVERLAY_FILENAME
    make_grid_image(scores=scores, output_path=grid_path)
    make_overlay_image(
        scores=scores,
        image_path=image_path,
        output_path=overlay_path,
        alpha=args.overlay_alpha,
    )

    print(f"Wrote grid heatmap: {grid_path}")
    print(f"Wrote overlay heatmap: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
