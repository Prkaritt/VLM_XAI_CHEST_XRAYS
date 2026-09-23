# Checkpoints

## 2026-09-16

* Initialized the project repository for the VLM chest X-ray explainability pilot.
* Added the initial documentation scaffold: `README.md`, `PROJECT_BRIEF.md`, `STATUS.md`, `AGENTS.md`, `checkpoints.md`, `.gitignore`, and `pyproject.toml`.
* Defined the initial pilot scope: MIMIC-Ext-CXR-QBA, frontal chest X-rays, pleural effusion and pneumothorax, one pretrained medical VLM, one explanation method, and paraphrased question variants.
* Created the initial Git commit and pushed the repository to GitHub.
* Set the next incremental action: inspect the QBA metadata structure locally before processing images or running model experiments.

## 2026-09-16

* Selected CheXagent as the first VLM for the pilot.
* Narrowed the first implementation milestone to one image, one closed yes/no question, and one parsed model answer.
* Added `src/chexagent_inference.py`, a minimal CheXagent wrapper with closed-answer prompt construction, yes/no/uncertain response parsing, device selection, optional local CheXagent repo import, dry-run mode, and mock-response mode.
* Documented the first command-line workflow in `README.md`.
* Did not install packages, download model weights, process data, or run real model inference.

## 2026-09-16

* Chose a project-local but Git-ignored setup for heavy/disposable artifacts.
* Added `external/` to `.gitignore` for local clones such as `external/CheXagent/`.
* Documented `.cache/huggingface/` as the local Hugging Face cache location for CheXagent weights and tokenizer files.
* Documented cleanup commands for removing local external code and model cache artifacts after experimentation.

## 2026-09-16

* Added `data/raw/samples/` as the project-local folder for one-off sample images used in pipeline checks.
* Kept actual sample images ignored by Git while allowing `data/raw/samples/README.md` to document the folder.

## 2026-09-16

* Attempted first real local CheXagent load after tokenizer/config validation.
* Hugging Face downloaded tokenizer/config files and checkpoint shards into the project-local cache.
* The official CheXagent class failed before inference because it tried to move a model after `device_map="auto"` had offloaded modules to CPU or disk.
* Updated the project wrapper with a default local-safe loader that avoids editing the official CheXagent repo and avoids the problematic post-offload full-model move.

## 2026-09-16

* Confirmed the project-local Hugging Face cache grew to approximately 14 GB after model download.
* Retried local loading with the local-safe loader and encountered a meta-tensor/offload error before completing inference.
* Added a `--debug` option to `src/chexagent_inference.py` so the next run can print a full traceback for diagnosis.

## 2026-09-16

* Diagnosed local CheXagent execution in the `vlm-xai` conda environment.
* Confirmed MPS/disk-offloaded generation failed locally with a meta-tensor error.
* Confirmed CPU float16 reached the visual encoder but failed with a float/half dtype mismatch.
* Completed the first real local CheXagent inference using `python3`, CPU, and float32 on `data/raw/samples/sample_chest_xray.jpg`.
* The first local model response to "Is there pleural effusion?" was parsed as `no` with raw response `No`.

## 2026-09-21

* Added optional next-token Yes/No score extraction to `src/chexagent_inference.py`.
* Kept score extraction local-loader only because the project-side runner exposes the CheXagent model and tokenizer logits directly.
* Added `--score-yes-no` to report Yes score, No score, probabilities among the two labels, and the Yes-No margin.
* Defined the first support score for patch occlusion as the Yes-No margin from CheXagent's next-token logits.
* Did not run a real model scoring pass; the next step is for the user to validate `--score-yes-no` on the sample image using the existing local environment.

## 2026-09-21

* Built a 10-image Pleural Effusion pilot cohort locally from the MIMIC-CXR-JPG manually labeled test-set file, split metadata, image metadata, and `IMAGE_FILENAMES`.
* Downloaded only the selected pilot JPG images into ignored `mimic-cxr-jpg/`.
* Validated CheXagent Yes/No score extraction on one positive MIMIC-CXR pilot image; both generated answer and score-predicted answer were `yes`.
* Added `src/run_chexagent_baseline_manifest.py` to run CheXagent baseline inference over a manifest and save raw responses, parsed answers, Yes/No scores, probabilities, and margins.
* Kept the temporary pilot-manifest builder and generated manifest outputs local/ignored.
