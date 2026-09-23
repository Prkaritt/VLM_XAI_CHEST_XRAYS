"""Run visual input sanity controls for a yes/no VLM question."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

from src.chexagent_inference import DEFAULT_OFFLOAD_FOLDER, DtypeName
from src.vlm_runner import ModelName, VLMYesNoResult, create_vlm_runner


DEFAULT_CONTROLS_MANIFEST = Path("runs/controls/manifest.csv")
DEFAULT_OUTPUT_DIR = Path("runs/visual_controls")
DEFAULT_QUESTION = "Is there pleural effusion?"


OUTPUT_COLUMNS = [
    "case_id",
    "control_type",
    "image_path",
    "question",
    "model_name",
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


def read_controls_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result_to_row(
    case_id: str,
    control_type: str,
    result: VLMYesNoResult,
) -> dict[str, str | float | None]:
    return {
        "case_id": case_id,
        "control_type": control_type,
        "image_path": str(result.image_path),
        "question": result.question,
        "model_name": result.model_name,
        "model_id": result.model_id,
        "raw_response": result.raw_response,
        "parsed_answer": result.parsed_answer,
        "score_predicted_answer": result.score_predicted_answer,
        "yes_score": result.yes_score,
        "no_score": result.no_score,
        "yes_probability": result.yes_probability,
        "no_probability": result.no_probability,
        "yes_no_margin": result.yes_no_margin,
    }


def write_results(rows: list[dict[str, str | float | None]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a real image and reusable blank controls with a VLM."
    )
    parser.add_argument("--real-image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--controls-manifest", default=DEFAULT_CONTROLS_MANIFEST, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--model", default="chexagent", choices=["chexagent"])
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
    parser.add_argument("--offload-folder", default=DEFAULT_OFFLOAD_FOLDER, type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    real_image = args.real_image.expanduser().resolve()
    controls_manifest = args.controls_manifest.expanduser().resolve()

    if not real_image.exists():
        print(f"Real image does not exist: {real_image}", file=sys.stderr)
        return 1
    if not controls_manifest.exists():
        print(f"Controls manifest does not exist: {controls_manifest}", file=sys.stderr)
        return 1

    try:
        runner = create_vlm_runner(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            offload_folder=args.offload_folder,
        )

        rows: list[dict[str, str | float | None]] = []
        inputs = [{"control_type": "real", "image_path": str(real_image)}]
        inputs.extend(read_controls_manifest(controls_manifest))

        for index, item in enumerate(inputs, start=1):
            control_type = item["control_type"]
            image_path = Path(item["image_path"]).expanduser().resolve()
            result = runner.answer_yes_no(image_path=image_path, question=args.question)
            rows.append(
                result_to_row(
                    case_id=args.case_id,
                    control_type=control_type,
                    result=result,
                )
            )
            print(
                f"[{index}/{len(inputs)}] control={control_type} "
                f"parsed={result.parsed_answer} score={result.score_predicted_answer} "
                f"margin={result.yes_no_margin:.6f}"
            )

        output_path = (
            args.output_dir
            / args.case_id
            / f"{args.model}_visual_control_scores.csv"
        )
        write_results(rows, output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    print(f"Wrote visual control results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
