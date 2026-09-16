# VLM Explainability for Chest X-rays

## Overview

This repository contains a pilot study of visual explanations produced by a pretrained medical vision-language model for chest X-ray questions.

The project examines whether semantically equivalent questions that produce the same answer also produce similar visual explanations.

## Current Research Scope

The initial pilot uses:

- MIMIC-Ext-CXR-QBA v1.0.1
- Frontal chest X-rays
- Pleural effusion and pneumothorax
- One pretrained medical VLM
- One explanation method
- Original questions and three paraphrases
- Explanation stability, clinical grounding, and perturbation-based faithfulness

See `PROJECT_BRIEF.md` for the complete scope.

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

The first implementation milestone is intentionally small:

```text
one chest X-ray image + one closed yes/no question -> CheXagent -> parsed answer
```

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

This project is currently in the documentation and planning stage. No data processing or model experiments have started.
