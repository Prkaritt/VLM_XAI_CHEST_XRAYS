"""Run CheXagent baseline inference for rows in a manifest CSV.

The manifest should contain one image per row and include the columns produced
by the pilot manifest workflow. This script writes model responses and Yes/No
score margins for downstream patch-occlusion experiments.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

from src.chexagent_inference import (
    DEFAULT_MODEL_ID,
    DEFAULT_OFFLOAD_FOLDER,
    DtypeName,
    LocalCheXagent,
    build_yes_no_prompt,
    parse_yes_no_response,
    resolve_device,
)


DEFAULT_MANIFEST = Path("runs/manifests/pleural_effusion_pilot_manifest.csv")
DEFAULT_IMAGE_ROOT = Path("mimic-cxr-jpg/2.1.0")
DEFAULT_OUTPUT = Path("runs/baselines/chexagent_pleural_effusion_pilot_results.csv")


OUTPUT_COLUMNS = [
    "study_id",
    "subject_id",
    "dicom_id",
    "image_path",
    "local_image_path",
    "view_position",
    "split",
    "pleural_effusion_label",
    "cohort",
    "question",
    "model_id",
    "raw_response",
    "parsed_answer",
    "score_predicted_answer",
    "yes_score",
    "no_score",
    "yes_probability",
    "no_probability",
    "yes_no_margin",
]


def read_manifest(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


def resolve_image_path(image_root: Path, manifest_image_path: str) -> Path:
    image_path = image_root / manifest_image_path
    return image_path.expanduser().resolve()


def result_row(
    manifest_row: dict[str, str],
    local_image_path: Path,
    raw_response: str,
    parsed_answer: str,
    score,
) -> dict[str, str | float]:
    return {
        "study_id": manifest_row["study_id"],
        "subject_id": manifest_row["subject_id"],
        "dicom_id": manifest_row["dicom_id"],
        "image_path": manifest_row["image_path"],
        "local_image_path": str(local_image_path),
        "view_position": manifest_row["view_position"],
        "split": manifest_row["split"],
        "pleural_effusion_label": manifest_row["pleural_effusion_label"],
        "cohort": manifest_row["cohort"],
        "question": manifest_row["question"],
        "model_id": DEFAULT_MODEL_ID,
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "score_predicted_answer": score.predicted_answer,
        "yes_score": score.yes_score,
        "no_score": score.no_score,
        "yes_probability": score.yes_probability,
        "no_probability": score.no_probability,
        "yes_no_margin": score.margin,
    }


def write_results(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CheXagent baseline inference for a manifest CSV."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        type=Path,
        help="Root prepended to manifest image_path values.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device. Use 'auto' for cuda -> mps -> cpu selection.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. Auto uses bfloat16 on CUDA, float16 on MPS, and float32 on CPU.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=DEFAULT_OFFLOAD_FOLDER,
        help="Disk offload folder used when device_map='auto'.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Process only the first N manifest rows for smoke testing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a full traceback when baseline inference fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.exists():
        print(f"Manifest does not exist: {manifest_path}", file=sys.stderr)
        return 1

    rows = read_manifest(manifest_path, limit=args.limit)
    if not rows:
        print(f"Manifest has no rows to process: {manifest_path}", file=sys.stderr)
        return 1

    resolved_device = resolve_device(args.device)
    try:
        agent = LocalCheXagent(
            model_name=DEFAULT_MODEL_ID,
            device=resolved_device,
            dtype=args.dtype,
            offload_folder=args.offload_folder,
        )

        output_rows: list[dict[str, str | float]] = []
        for index, row in enumerate(rows, start=1):
            local_image_path = resolve_image_path(args.image_root, row["image_path"])
            if not local_image_path.exists():
                raise FileNotFoundError(f"Image does not exist: {local_image_path}")

            question = row["question"]
            prompt = build_yes_no_prompt(question)
            raw_response = agent.generate([str(local_image_path)], prompt)
            score = agent.score_yes_no([str(local_image_path)], prompt)
            parsed_answer = parse_yes_no_response(raw_response)
            output_rows.append(
                result_row(
                    manifest_row=row,
                    local_image_path=local_image_path,
                    raw_response=raw_response,
                    parsed_answer=parsed_answer,
                    score=score,
                )
            )
            print(
                f"[{index}/{len(rows)}] study_id={row['study_id']} "
                f"label={row['pleural_effusion_label']} parsed={parsed_answer} "
                f"score={score.predicted_answer} margin={score.margin:.6f}"
            )

        write_results(output_rows, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    print(f"Wrote baseline results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
