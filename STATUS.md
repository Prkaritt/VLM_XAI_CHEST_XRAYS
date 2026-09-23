# Project Status

## Last Updated

2026-09-21

## Current Phase

Stage 1: Causal patch-occlusion scoring prototype

## Overall Status

CheXagent single-image inference and Yes/No score extraction work on one downloaded MIMIC-CXR pilot image

## Completed

* Created the initial repository structure.
* Added `AGENTS.md`.
* Added `PROJECT_BRIEF.md`.
* Added `README.md`.
* Added `STATUS.md`.
* Added `checkpoints.md`.
* Added `pyproject.toml`.
* Added `.gitignore`.
* Defined the provisional research scope.
* Created the initial Git commit and pushed the repository to GitHub.
* Selected CheXagent as the initial medical VLM for the pilot.
* Added a minimal closed-answer CheXagent inference wrapper for one image and one yes/no question.
* Added README notes for local dry-run/mock testing and future GPU-based inference.
* Defined project-local ignored folders for disposable external code and model cache artifacts.
* Added `data/raw/samples/` as the ignored local folder for one-off sample images.
* Added a local-safe CheXagent loader to avoid the official class's full-model move after disk/CPU offload.
* Added a debug flag to print full tracebacks for model-loading failures.
* Completed the first real local CheXagent inference on one sample image using CPU float32.
* Narrowed the research plan to answer-conditioned causal evidence invariance.
* Marked the earlier generic heatmap-stability plan as superseded.
* Added a model-independent patch occlusion utility for CheXagent-aligned `512x512` debug patch generation.
* Generated and visually inspected a 4-patch debug occlusion set under ignored `runs/`.
* Added optional next-token Yes/No score extraction to the local CheXagent runner.
* Added `--score-yes-no` for reporting Yes/No scores, probabilities among the two labels, and the Yes-No margin.
* Validated `--score-yes-no` on one downloaded MIMIC-CXR pilot image; generated answer and score-predicted answer both returned `yes`.
* Downloaded the 10-image Pleural Effusion pilot cohort selected from the MIMIC-CXR-JPG test split.
* Added a baseline manifest runner to process selected pilot images and save CheXagent responses plus Yes/No margins.

## Current Research Direction

Evaluate whether clinically equivalent chest X-ray prompts that produce the same model answer rely on the same causal visual evidence.

Current objective:

* Build causal patch-occlusion maps for answer-stable paraphrases.
* Measure causal evidence similarity across paraphrases.
* Test cross-paraphrase causal transfer.
* Compare causal evidence regions with radiologist-annotated pathology regions.

Central research question:

```text
When clinically equivalent prompts produce the same answer, does a medical VLM rely on the same causal visual evidence?
```

## Not Yet Completed

* Run baseline CheXagent inference for all 10 pilot images.
* Inspect the QBA metadata structure.
* Inspect the Attention Without Grounding implementation for patch occlusion, answer-margin extraction, and causal map construction.
* Connect patch occlusion outputs to CheXagent answer-margin scoring.
* Implement the causal map similarity matrix.
* Implement cross-paraphrase causal transfer.
* Decide whether subsequent model runs should stay local CPU-only or move to Colab/GPU.
* Run the first experiment.

## Current Next Action

Run `src/run_chexagent_baseline_manifest.py` with `--limit 1`, then run the full 10-image baseline if the smoke test succeeds.

## Current Decisions

* Continue with CheXagent-2.3B as the current prototype VLM because the one-image closed-answer pipeline works locally.
* Replicate the core experiment on a second VLM family only after the CheXagent causal-map workflow is complete.
* Keep the official CheXagent repository under ignored `external/CheXagent/` when testing locally.
* Keep Hugging Face model downloads under ignored `.cache/huggingface/` when testing locally.
* Keep one-off local image samples under ignored `data/raw/samples/`.
* Keep downloaded MIMIC-CXR-JPG images under ignored `mimic-cxr-jpg/`.
* Prefer the project-side local-safe CheXagent loader for low-memory local testing; keep the official loader available for comparison.
* Use `python3` in user-facing run commands.
* Use `--device cpu --dtype float32` for the reliable local smoke-test path on the current machine.
* Do not fine-tune during the initial pilot.
* Use frontal chest X-rays.
* Begin with pleural effusion and pneumothorax.
* Start with one image and two clinically equivalent answer-stable paraphrases.
* Scale to 3-4 paraphrases only after the first 2x2 causal-evidence similarity matrix works.
* Use causal patch-occlusion maps as the main explanation object.
* Use CheXagent-aligned `512x512` images for the first patching prototype.
* Use a `16x16` grid with soft gray-fill masking for the first patching prototype.
* Use the local CheXagent runner for Yes/No score extraction because it exposes the model and tokenizer logits directly.
* Treat the Yes-No margin as a model-internal support score, not a calibrated clinical probability.
* Treat saliency/attention heatmaps as superseded for the main method.
* Analyze answer instability separately from causal-evidence instability.
* Use clinical grounding as supporting validation, not the main novelty claim.

## Blockers

* Local dataset file paths and metadata structure must be confirmed.
* Underlying image access must be confirmed.
* CheXagent answer-margin extraction has been validated on one real MIMIC-CXR pilot image, but not yet across the full 10-image cohort.
* The Attention Without Grounding patch-occlusion implementation has been inspected for reusable design choices.
* CheXagent dependencies were installed locally by the user.
* CheXagent model files have downloaded into the project-local Hugging Face cache.
* The official CheXagent class failed locally because offloaded model modules cannot be moved afterward.
* The local-safe loader failed with MPS/disk offload, but succeeded with CPU float32.
* The initial patch grid, occlusion fill, and Yes-No margin definition have been selected for the prototype.

## Handoff Instructions

The next agent should read:

1. `AGENTS.md`
2. `PROJECT_BRIEF.md`
3. This file

The next agent should not begin large-scale processing or model experimentation until the dataset structure and pilot cohort definition have been confirmed.
