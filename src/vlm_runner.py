"""Reusable VLM runner interface for experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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


ModelName = Literal["chexagent"]


@dataclass(frozen=True)
class VLMYesNoResult:
    model_name: str
    model_id: str
    image_path: Path
    question: str
    raw_response: str
    parsed_answer: ParsedAnswer
    score_predicted_answer: ParsedAnswer | None
    yes_score: float | None
    no_score: float | None
    yes_probability: float | None
    no_probability: float | None
    yes_no_margin: float | None


class VLMYesNoRunner(Protocol):
    model_name: str
    model_id: str

    def answer_yes_no(self, image_path: Path, question: str) -> VLMYesNoResult:
        """Run one yes/no image-question inference."""


class CheXagentRunner:
    """CheXagent implementation of the reusable yes/no VLM interface."""

    model_name = "chexagent"
    model_id = DEFAULT_MODEL_ID

    def __init__(
        self,
        device: str = "auto",
        dtype: DtypeName = "auto",
        offload_folder: Path = DEFAULT_OFFLOAD_FOLDER,
    ) -> None:
        resolved_device = resolve_device(device)
        self.agent = LocalCheXagent(
            model_name=DEFAULT_MODEL_ID,
            device=resolved_device,
            dtype=dtype,
            offload_folder=offload_folder,
        )

    def answer_yes_no(self, image_path: Path, question: str) -> VLMYesNoResult:
        resolved_image_path = image_path.expanduser().resolve()
        if not resolved_image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {resolved_image_path}")

        prompt = build_yes_no_prompt(question)
        raw_response = self.agent.generate([str(resolved_image_path)], prompt)
        parsed_answer = parse_yes_no_response(raw_response)
        score = self.agent.score_yes_no([str(resolved_image_path)], prompt)

        return VLMYesNoResult(
            model_name=self.model_name,
            model_id=self.model_id,
            image_path=resolved_image_path,
            question=question,
            raw_response=raw_response,
            parsed_answer=parsed_answer,
            score_predicted_answer=score.predicted_answer,
            yes_score=score.yes_score,
            no_score=score.no_score,
            yes_probability=score.yes_probability,
            no_probability=score.no_probability,
            yes_no_margin=score.margin,
        )


def create_vlm_runner(
    model: ModelName,
    device: str = "auto",
    dtype: DtypeName = "auto",
    offload_folder: Path = DEFAULT_OFFLOAD_FOLDER,
) -> VLMYesNoRunner:
    if model == "chexagent":
        return CheXagentRunner(
            device=device,
            dtype=dtype,
            offload_folder=offload_folder,
        )
    raise ValueError(f"Unsupported model: {model}")
