# VLM Explainability for Chest X-rays

## Overview

This repository contains a pilot study of causal visual evidence used by medical vision-language models when answering chest X-ray questions.

The project examines whether clinically equivalent paraphrased questions that produce the same answer also rely on the same causally important image regions, and whether those regions remain clinically grounded.

## Current Research Objective

The current objective is to determine whether medical VLMs that produce the same answer to clinically equivalent paraphrased questions also rely on the same causal visual evidence.

Formal statement:

```text
This study investigates causal-evidence invariance in medical VLMs under
meaning-preserving query reformulation. Specifically, we test whether
answer-stable paraphrases depend on similar causally important image regions,
whether causal evidence identified under one paraphrase transfers to another
equivalent paraphrase, and whether that evidence aligns with
radiologist-annotated pathology regions.
```

Central research question:

```text
When clinically equivalent prompts produce the same answer, does a medical VLM
rely on the same causal visual evidence?
```

## Revised Scope

The current pilot uses:

- MIMIC-Ext-CXR-QBA v1.0.1
- Frontal chest X-rays
- Pleural effusion and pneumothorax
- Current prototype model: CheXagent-2.3B
- Later comparison model: one second VLM family after the CheXagent causal-map workflow is complete
- One image and two paraphrases first, then 3-4 paraphrases
- Causal patch-occlusion maps as the main explanation object
- Cross-paraphrase causal transfer
- Clinical grounding against radiologist-annotated pathology regions

See `PROJECT_BRIEF.md` for the complete scope.

## Superseded Research Assumptions

The earlier plan emphasized generic heatmap stability, attention/saliency comparison, broad perturbation faithfulness, and answer-flip analysis. That plan is now superseded.

The narrowed contribution is:

```text
Answer-conditioned causal evidence invariance.
```

The headline failure mode is:

```text
Same answer, different evidence.
```

Ordinary attention or saliency heatmaps should not be the main explanation mechanism. They may be used for context later, but the main explanation object should be a causal patch-occlusion map.

## Core Methodology

For each image-question pair:

1. Run baseline model inference.
2. Extract the answer and answer margin.
3. Divide the image into a patch grid.
4. Occlude one patch at a time.
5. Re-run the VLM.
6. Compute the answer-margin drop for each patch.
7. Assemble a causal importance map.

For patch `r` and question `q`:

```text
delta_r(q) = margin_original(q) - margin_occluded_patch_r(q)
```

A large positive `delta_r(q)` means removing patch `r` substantially reduced support for the original answer.

For paraphrase groups:

```text
q1 -> C1
q2 -> C2
q3 -> C3
q4 -> C4
```

The main analysis compares:

- causal map similarity across answer-stable paraphrases
- top-k causal patch overlap / IoU
- cross-paraphrase causal transfer
- clinical grounding against radiologist annotations

The primary result should be a paraphrase causal-evidence similarity matrix. The strongest methodological addition is cross-paraphrase causal transfer: identify causal evidence under `q_i`, intervene on that evidence, and test the effect under `q_j`.

## Patch Occlusion Utility

The first causal-map building block is a model-independent patch occlusion utility:

```text
src/patch_occlusion.py
```

For CheXagent, the visual input is resized to `512x512`, so the initial debug configuration uses a `16x16` grid with `32x32` pixel patches and soft gray-fill masking.

Generate a small debug set first:

```bash
python3 -m src.patch_occlusion \
  --image data/raw/samples/sample_chest_xray.jpg \
  --output-dir runs/patch_debug/sample_chest_xray \
  --image-size 512 \
  --grid-size 16 \
  --fill gray \
  --blur-radius 3 \
  --limit 4
```

The utility writes:

```text
runs/patch_debug/sample_chest_xray/original_resized.png
runs/patch_debug/sample_chest_xray/metadata.json
runs/patch_debug/sample_chest_xray/manifest.jsonl
runs/patch_debug/sample_chest_xray/occluded/
```

`runs/` is ignored by Git. These generated images are local debugging artifacts and should not be committed.

## Repository Files

```text
AGENTS.md          Instructions for AI agents
PROJECT_BRIEF.md   Stable project context and research scope
README.md          Repository overview and usage information
STATUS.md          Current progress and next action
checkpoints.md     Historical milestone notes
pyproject.toml     Python project configuration
```

Additional folders such as `configs/`, `data/`, `src/`, and `runs/` will be added as implementation begins.

## Initial CheXagent Workflow

The first completed implementation milestone is intentionally small:

```text
one chest X-ray image + one closed yes/no question -> CheXagent -> parsed answer
```

This is the current working prototype model path. The causal patch-occlusion workflow should be built on top of this working CheXagent pipeline first, then replicated on another model family later if time and compute allow.

The wrapper lives at:

```text
src/chexagent_inference.py
```

Example command shape:

```bash
python3 -m src.chexagent_inference \
  --image /path/to/chest_xray.jpg \
  --question "Is there pleural effusion?"
```

For local script testing without loading the model:

```bash
python3 -m src.chexagent_inference \
  --image /path/to/chest_xray.jpg \
  --question "Is there pleural effusion?" \
  --mock-response "Yes"
```

For prompt/device inspection only:

```bash
python3 -m src.chexagent_inference \
  --image /path/to/chest_xray.jpg \
  --question "Is there pleural effusion?" \
  --dry-run
```

Real inference requires the official CheXagent code and dependencies to be prepared separately. Do not install model dependencies or download model weights casually; this should be done intentionally, ideally first in a GPU environment such as Colab.

Recommended local-only layout:

```text
data/raw/samples/        Local one-off sample images for pipeline checks
external/CheXagent/       Local clone of the official CheXagent repository
.cache/huggingface/       Local Hugging Face cache for model weights and tokenizer files
```

Image files in `data/raw/samples/`, `external/`, and `.cache/` are ignored by Git. They are disposable local artifacts and can be deleted when model experimentation is finished:

```bash
rm -f data/raw/samples/*
rm -rf external/
rm -rf .cache/
```

To keep model downloads inside the project folder during a local test:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
```

The official CheXagent repository can be cloned locally with:

```bash
mkdir -p external
git clone https://github.com/Stanford-AIMI/CheXagent.git external/CheXagent
```

For the current local Apple Silicon test environment, the successful real
inference path was CPU float32:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m src.chexagent_inference \
  --image data/raw/samples/sample_chest_xray.jpg \
  --question "Is there pleural effusion?" \
  --device cpu \
  --dtype float32
```

By default, the wrapper uses a project-side local loader instead of directly
instantiating the official CheXagent class. This avoids moving the whole model
after `device_map="auto"` has offloaded some modules to CPU or disk, which can
happen on low-memory local machines.

To compare with the official class directly:

```bash
python3 -m src.chexagent_inference \
  --image data/raw/samples/sample_chest_xray.jpg \
  --question "Is there pleural effusion?" \
  --device mps \
  --loader official \
  --chexagent-repo external/CheXagent
```

Likely model-runtime dependencies include:

- `torch`
- `transformers`
- `accelerate`
- `pillow`
- the official CheXagent repository from `https://github.com/Stanford-AIMI/CheXagent`

## Data Handling

MIMIC-Ext-CXR-QBA and its underlying MIMIC-CXR data are restricted-access resources.

Raw data must:

- Remain outside Git version control
- Be stored in an approved protected location
- Never be uploaded to public repositories
- Never be included in credentials, logs, or unnecessary AI-agent context

## Agent Workflow

Before making substantial changes, an AI agent should read:

1. `AGENTS.md`
2. `PROJECT_BRIEF.md`
3. `STATUS.md`

Agents should update `STATUS.md` after meaningful progress and record important milestones in `checkpoints.md`.

## Project Status

The research direction is defined and narrowed around causal evidence invariance under answer-preserving paraphrases. A single-image CheXagent closed-answer inference smoke test has completed locally. The immediate development milestone is to generate and compare causal maps for one image and two answer-stable paraphrases.
