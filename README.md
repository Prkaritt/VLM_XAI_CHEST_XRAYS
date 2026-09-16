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
