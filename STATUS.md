# Project Status

## Last Updated

2026-09-16

## Current Phase

Stage 0: Documentation and project scaffold

## Overall Status

Repository initialized; first local CheXagent single-image inference completed

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

## Current Research Direction

Evaluate whether semantically equivalent chest X-ray questions produce:

* Consistent answers
* Stable visual explanations
* Clinically grounded explanations
* Faithful explanations under controlled perturbations

## Not Yet Completed

* Verify local access to the required dataset files.
* Confirm access to the required underlying image files.
* Inspect the QBA metadata structure.
* Define the first pilot cohort.
* Select the initial explanation method.
* Decide whether subsequent model runs should stay local CPU-only or move to Colab/GPU.
* Run the first experiment.

## Current Next Action

Decide the next incremental workflow step after the successful single-image CheXagent answer: structured result logging or Colab/GPU setup.

## Current Decisions

* Use CheXagent as the initial pretrained medical VLM.
* Keep the official CheXagent repository under ignored `external/CheXagent/` when testing locally.
* Keep Hugging Face model downloads under ignored `.cache/huggingface/` when testing locally.
* Keep one-off local image samples under ignored `data/raw/samples/`.
* Prefer the project-side local-safe CheXagent loader for low-memory local testing; keep the official loader available for comparison.
* Use `python3` in user-facing run commands.
* Use `--device cpu --dtype float32` for the reliable local smoke-test path on the current machine.
* Do not fine-tune during the initial pilot.
* Use frontal chest X-rays.
* Begin with pleural effusion and pneumothorax.
* Start with approximately 50 images.
* Start with one original closed-answer yes/no question per image.
* Add paraphrases and question variants only after the single-image workflow works.
* Analyze answer instability separately from explanation instability.
* Evaluate stability, clinical grounding, and faithfulness.

## Blockers

* Local dataset file paths and metadata structure must be confirmed.
* Underlying image access must be confirmed.
* CheXagent dependencies were installed locally by the user.
* CheXagent model files have downloaded into the project-local Hugging Face cache.
* The official CheXagent class failed locally because offloaded model modules cannot be moved afterward.
* The local-safe loader failed with MPS/disk offload, but succeeded with CPU float32.
* The initial explanation method has not yet been selected.

## Handoff Instructions

The next agent should read:

1. `AGENTS.md`
2. `PROJECT_BRIEF.md`
3. This file

The next agent should not begin large-scale processing or model experimentation until the dataset structure and pilot cohort definition have been confirmed.
