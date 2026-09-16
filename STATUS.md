# Project Status

## Last Updated

2026-09-16

## Current Phase

Stage 0: Documentation and project scaffold

## Overall Status

Not started

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

## Current Research Direction

Evaluate whether semantically equivalent chest X-ray questions produce:

* Consistent answers
* Stable visual explanations
* Clinically grounded explanations
* Faithful explanations under controlled perturbations

## Not Yet Completed

* Verify PhysioNet dataset access.
* Confirm access to the required underlying image files.
* Inspect the QBA metadata structure.
* Define the first pilot cohort.
* Select the initial medical VLM.
* Select the initial explanation method.
* Implement the first smoke test.
* Run the first experiment.

## Current Next Action

Verify dataset access and inspect the QBA metadata structure before downloading or processing the full dataset.

## Current Decisions

* Use a pretrained, off-the-shelf medical VLM.
* Do not fine-tune during the initial pilot.
* Use frontal chest X-rays.
* Begin with pleural effusion and pneumothorax.
* Start with approximately 50 images.
* Use one original question and three paraphrases per case.
* Analyze answer instability separately from explanation instability.
* Evaluate stability, clinical grounding, and faithfulness.

## Blockers

* Dataset and underlying image access must be confirmed.
* The initial model and explanation method have not yet been selected.

## Handoff Instructions

The next agent should read:

1. `AGENTS.md`
2. `PROJECT_BRIEF.md`
3. This file

The next agent should not begin large-scale processing or model experimentation until the dataset structure and pilot cohort definition have been confirmed.
