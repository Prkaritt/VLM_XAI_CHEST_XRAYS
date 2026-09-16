"""Minimal closed-answer CheXagent inference wrapper.

This module is the first project workflow:

    chest X-ray image + yes/no clinical question -> CheXagent -> parsed answer

It intentionally does not install or download anything. The official CheXagent
repository and its model dependencies must be prepared separately before real
inference is run.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ParsedAnswer = Literal["yes", "no", "uncertain"]
LoaderName = Literal["local", "official"]
DtypeName = Literal["auto", "float16", "bfloat16", "float32"]


DEFAULT_MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
DEFAULT_OFFLOAD_FOLDER = Path(".cache/huggingface/offload")
DEFAULT_SYSTEM_CONTEXT = (
    "You are answering a closed clinical visual question about a chest X-ray. "
    "This is for research only and not for clinical diagnosis."
)


@dataclass(frozen=True)
class InferenceResult:
    image: Path
    question: str
    prompt: str
    raw_response: str
    parsed_answer: ParsedAnswer
    model_id: str = DEFAULT_MODEL_ID


def build_yes_no_prompt(question: str) -> str:
    """Build a conservative closed-answer prompt for CheXagent."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty.")

    return (
        f"{DEFAULT_SYSTEM_CONTEXT}\n\n"
        f"Question: {normalized_question}\n\n"
        "Answer with exactly one word: Yes or No."
    )


def parse_yes_no_response(response: str) -> ParsedAnswer:
    """Parse a raw model response into yes, no, or uncertain."""
    normalized = response.strip().lower()
    if not normalized:
        return "uncertain"

    first_token_match = re.search(r"[a-z]+", normalized)
    if first_token_match is None:
        return "uncertain"

    first_token = first_token_match.group(0)
    if first_token == "yes":
        return "yes"
    if first_token == "no":
        return "no"

    yes_match = re.search(r"\byes\b", normalized)
    no_match = re.search(r"\bno\b", normalized)

    if yes_match and not no_match:
        return "yes"
    if no_match and not yes_match:
        return "no"

    return "uncertain"


def resolve_device(requested_device: str) -> str:
    """Resolve an execution device without requiring torch at import time."""
    requested_device = requested_device.lower()
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype: DtypeName, device: str):
    """Resolve model dtype while importing torch only when needed."""
    import torch

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def infer_input_device(model, requested_device: str) -> str:
    """Choose a reasonable input tensor device for regular or offloaded models."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return requested_device

    non_disk_devices = [device for device in device_map.values() if device != "disk"]
    if requested_device in non_disk_devices:
        return requested_device
    if "mps" in non_disk_devices:
        return "mps"
    if "cuda" in non_disk_devices:
        return "cuda"
    if 0 in non_disk_devices:
        return "cuda:0"
    return "cpu"


class LocalCheXagent:
    """Small project-side CheXagent runner that avoids editing the official repo."""

    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: DtypeName,
        offload_folder: Path,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils.logging import set_verbosity_error

        set_verbosity_error()

        self.model_name = model_name
        self.device = device
        self.dtype = resolve_torch_dtype(dtype, device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True,
                offload_folder=str(offload_folder),
                low_cpu_mem_usage=True,
            )

        self.model.eval()
        self.input_device = infer_input_device(self.model, device)

    def generate(self, paths: list[str], prompt: str) -> str:
        import torch

        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )

        with torch.no_grad():
            output = self.model.generate(
                input_ids.to(self.input_device),
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=32,
            )[0]
        return self.tokenizer.decode(output[input_ids.size(1) : -1])


def add_optional_repo_to_path(chexagent_repo: Path | None) -> None:
    """Allow importing CheXagent from a local clone without packaging it."""
    if chexagent_repo is None:
        return

    repo_path = chexagent_repo.expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"CheXagent repo path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"CheXagent repo path is not a directory: {repo_path}")

    sys.path.insert(0, str(repo_path))


def load_official_chexagent(chexagent_repo: Path | None, device: str):
    """Load the official CheXagent class and set its generation device."""
    add_optional_repo_to_path(chexagent_repo)

    try:
        module = importlib.import_module("model_chexagent.chexagent")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the official CheXagent module. Clone the CheXagent "
            "repository and pass its path with --chexagent-repo, or make "
            "'model_chexagent.chexagent' importable in your Python environment. "
            "Do not run real inference until the CheXagent dependencies have "
            "been installed intentionally."
        ) from exc

    agent = module.CheXagent()
    if hasattr(agent, "device"):
        agent.device = device
    return agent


def load_chexagent(
    chexagent_repo: Path | None,
    device: str,
    loader: LoaderName,
    dtype: DtypeName,
    offload_folder: Path,
):
    """Load CheXagent using either the official class or local-safe runner."""
    if loader == "official":
        return load_official_chexagent(chexagent_repo=chexagent_repo, device=device)

    offload_folder.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return LocalCheXagent(
        model_name=DEFAULT_MODEL_ID,
        device=device,
        dtype=dtype,
        offload_folder=offload_folder,
    )


def run_closed_answer_inference(
    image: Path,
    question: str,
    device: str = "auto",
    chexagent_repo: Path | None = None,
    mock_response: str | None = None,
    loader: LoaderName = "local",
    dtype: DtypeName = "auto",
    offload_folder: Path = DEFAULT_OFFLOAD_FOLDER,
) -> InferenceResult:
    """Run or simulate one closed-answer CheXagent inference."""
    image_path = image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    prompt = build_yes_no_prompt(question)
    resolved_device = resolve_device(device)

    if mock_response is None:
        agent = load_chexagent(
            chexagent_repo=chexagent_repo,
            device=resolved_device,
            loader=loader,
            dtype=dtype,
            offload_folder=offload_folder,
        )
        raw_response = agent.generate([str(image_path)], prompt)
    else:
        raw_response = mock_response

    return InferenceResult(
        image=image_path,
        question=question,
        prompt=prompt,
        raw_response=raw_response,
        parsed_answer=parse_yes_no_response(raw_response),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one closed-answer CheXagent inference for one chest X-ray image."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to one chest X-ray image.")
    parser.add_argument("--question", required=True, help="Closed yes/no question for the image.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device. Use 'auto' for cuda -> mps -> cpu selection.",
    )
    parser.add_argument(
        "--chexagent-repo",
        type=Path,
        default=None,
        help="Optional path to a local clone of https://github.com/Stanford-AIMI/CheXagent. Required only for --loader official.",
    )
    parser.add_argument(
        "--loader",
        default="local",
        choices=["local", "official"],
        help="Use the project-side local-safe loader or the official CheXagent class.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for the local loader. Auto uses bfloat16 on CUDA, float16 on MPS, and float32 on CPU.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=DEFAULT_OFFLOAD_FOLDER,
        help="Disk offload folder used by the local loader when device_map='auto'.",
    )
    parser.add_argument(
        "--mock-response",
        default=None,
        help="Skip model loading and parse this response instead. Useful for local script testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the constructed prompt and device selection without loading CheXagent.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a full traceback when inference fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    prompt = build_yes_no_prompt(args.question)
    resolved_device = resolve_device(args.device)

    if args.dry_run:
        print(f"Image: {image_path}")
        print(f"Device: {resolved_device}")
        print("Prompt:")
        print(prompt)
        return 0

    try:
        result = run_closed_answer_inference(
            image=args.image,
            question=args.question,
            device=args.device,
            chexagent_repo=args.chexagent_repo,
            mock_response=args.mock_response,
            loader=args.loader,
            dtype=args.dtype,
            offload_folder=args.offload_folder,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    print(f"Model: {result.model_id}")
    print(f"Image: {result.image}")
    print(f"Question: {result.question}")
    print(f"Parsed answer: {result.parsed_answer}")
    print(f"Raw response: {result.raw_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
