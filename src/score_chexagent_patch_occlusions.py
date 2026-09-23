"""Score CheXagent responses for patch-occluded images.

This is the first bridge between generated patch occlusions and causal scoring.
It compares each occluded image's Yes/No support against the original image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Literal

from src.chexagent_inference import (
    DEFAULT_MODEL_ID,
    DEFAULT_OFFLOAD_FOLDER,
    DtypeName,
    LocalCheXagent,
    ParsedAnswer,
    build_yes_no_prompt,
    parse_yes_no_response,
    resolve_device,
)


TargetAnswer = Literal["auto", "yes", "no"]
DEFAULT_OUTPUT = Path("runs/patch_scores/patch_scores.csv")
DEFAULT_QUESTION = "Is there pleural effusion?"


OUTPUT_COLUMNS = [
    "patch_index",
    "row",
    "col",
    "x1",
    "y1",
    "x2",
    "y2",
    "patch_image_path",
    "question",
    "target_answer",
    "original_raw_response",
    "original_parsed_answer",
    "original_score_predicted_answer",
    "original_yes_score",
    "original_no_score",
    "original_yes_probability",
    "original_no_probability",
    "original_yes_no_margin",
    "original_support",
    "occluded_raw_response",
    "occluded_parsed_answer",
    "occluded_score_predicted_answer",
    "occluded_yes_score",
    "occluded_no_score",
    "occluded_yes_probability",
    "occluded_no_probability",
    "occluded_yes_no_margin",
    "occluded_support",
    "support_drop",
]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def read_metadata(patch_manifest: Path) -> dict:
    metadata_path = patch_manifest.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Patch metadata file does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def support_from_margin(margin: float, target_answer: ParsedAnswer) -> float:
    if target_answer == "yes":
        return margin
    if target_answer == "no":
        return -margin
    raise ValueError(f"Target answer must be yes or no, got: {target_answer}")


def resolve_target_answer(requested: TargetAnswer, parsed_answer: ParsedAnswer) -> ParsedAnswer:
    if requested == "auto":
        if parsed_answer not in {"yes", "no"}:
            raise ValueError(
                "Could not infer target answer from original response. "
                "Pass --target-answer yes or --target-answer no."
            )
        return parsed_answer
    return requested


def write_rows(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score CheXagent Yes/No support drops for patch-occluded images."
    )
    parser.add_argument("--patch-manifest", required=True, type=Path)
    parser.add_argument(
        "--original-image",
        default=None,
        type=Path,
        help="Original image path. Defaults to source_image from patch metadata.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--target-answer",
        default="auto",
        choices=["auto", "yes", "no"],
        help="Answer whose support drop should be measured. Auto uses the original parsed answer.",
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
    parser.add_argument("--offload-folder", default=DEFAULT_OFFLOAD_FOLDER, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_manifest = args.patch_manifest.expanduser().resolve()
    if not patch_manifest.exists():
        print(f"Patch manifest does not exist: {patch_manifest}", file=sys.stderr)
        return 1

    try:
        metadata = read_metadata(patch_manifest)
        original_image = (
            args.original_image.expanduser().resolve()
            if args.original_image is not None
            else Path(metadata["source_image"]).expanduser().resolve()
        )
        if not original_image.exists():
            raise FileNotFoundError(f"Original image does not exist: {original_image}")

        patch_records = read_jsonl(patch_manifest, limit=args.limit)
        if not patch_records:
            raise ValueError(f"No patch records found in: {patch_manifest}")

        prompt = build_yes_no_prompt(args.question)
        resolved_device = resolve_device(args.device)
        agent = LocalCheXagent(
            model_name=DEFAULT_MODEL_ID,
            device=resolved_device,
            dtype=args.dtype,
            offload_folder=args.offload_folder,
        )

        original_raw_response = agent.generate([str(original_image)], prompt)
        original_score = agent.score_yes_no([str(original_image)], prompt)
        original_parsed_answer = parse_yes_no_response(original_raw_response)
        target_answer = resolve_target_answer(args.target_answer, original_parsed_answer)
        original_support = support_from_margin(original_score.margin, target_answer)

        output_rows: list[dict[str, str | float | int]] = []
        for index, patch_record in enumerate(patch_records, start=1):
            patch_image_path = Path(patch_record["image_path"]).expanduser().resolve()
            if not patch_image_path.exists():
                raise FileNotFoundError(f"Patch image does not exist: {patch_image_path}")

            occluded_raw_response = agent.generate([str(patch_image_path)], prompt)
            occluded_score = agent.score_yes_no([str(patch_image_path)], prompt)
            occluded_parsed_answer = parse_yes_no_response(occluded_raw_response)
            occluded_support = support_from_margin(occluded_score.margin, target_answer)
            support_drop = original_support - occluded_support

            output_rows.append(
                {
                    "patch_index": patch_record["patch_index"],
                    "row": patch_record["row"],
                    "col": patch_record["col"],
                    "x1": patch_record["x1"],
                    "y1": patch_record["y1"],
                    "x2": patch_record["x2"],
                    "y2": patch_record["y2"],
                    "patch_image_path": str(patch_image_path),
                    "question": args.question,
                    "target_answer": target_answer,
                    "original_raw_response": original_raw_response,
                    "original_parsed_answer": original_parsed_answer,
                    "original_score_predicted_answer": original_score.predicted_answer,
                    "original_yes_score": original_score.yes_score,
                    "original_no_score": original_score.no_score,
                    "original_yes_probability": original_score.yes_probability,
                    "original_no_probability": original_score.no_probability,
                    "original_yes_no_margin": original_score.margin,
                    "original_support": original_support,
                    "occluded_raw_response": occluded_raw_response,
                    "occluded_parsed_answer": occluded_parsed_answer,
                    "occluded_score_predicted_answer": occluded_score.predicted_answer,
                    "occluded_yes_score": occluded_score.yes_score,
                    "occluded_no_score": occluded_score.no_score,
                    "occluded_yes_probability": occluded_score.yes_probability,
                    "occluded_no_probability": occluded_score.no_probability,
                    "occluded_yes_no_margin": occluded_score.margin,
                    "occluded_support": occluded_support,
                    "support_drop": support_drop,
                }
            )
            print(
                f"[{index}/{len(patch_records)}] patch={patch_record['patch_index']} "
                f"r={patch_record['row']} c={patch_record['col']} "
                f"parsed={occluded_parsed_answer} "
                f"original_margin={original_score.margin:.6f} "
                f"occluded_margin={occluded_score.margin:.6f} "
                f"drop={support_drop:.6f}"
            )

        write_rows(output_rows, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    print(f"Wrote patch scores: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
