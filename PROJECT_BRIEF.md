# VLM Explainability for Chest X-rays

## Purpose

This project investigates whether medical vision-language models use the same causal visual evidence when clinically equivalent paraphrased questions produce the same answer for a chest X-ray.

## Working Research Question

When clinically equivalent prompts produce the same answer, does a medical VLM rely on the same causal visual evidence?

## Formal Objective

This study investigates causal-evidence invariance in medical VLMs under meaning-preserving query reformulation. Specifically, it tests whether answer-stable paraphrases depend on similar causally important image regions, whether causal evidence identified under one paraphrase transfers to another equivalent paraphrase, and whether that evidence aligns with radiologist-annotated pathology regions.

## Initial Pilot Scope

* Dataset: MIMIC-Ext-CXR-QBA v1.0.1
* Image source: MIMIC-CXR-JPG, when required for the image files
* Image views: Frontal chest X-rays only
* Initial findings: Pleural effusion and pneumothorax
* Initial prototype: One image and two clinically equivalent paraphrases
* Initial cohort after prototype: A small frontal chest X-ray cohort
* Questions: One finding and one clear anatomical region per question
* Question variants: Start with two answer-stable paraphrases, then scale to 3-4 paraphrases
* Answers: Short or yes/no answers where possible
* Current prototype model: CheXagent-2.3B
* Later comparison model: One second VLM family after the CheXagent causal-map workflow is complete
* Fine-tuning: Not used in the initial pilot
* Main explanation object: Causal patch-occlusion map, not ordinary saliency or attention

## Evaluation Dimensions

1. Answer stability across paraphrased questions as a filter for analysis
2. Causal evidence similarity across answer-stable paraphrases
3. Cross-paraphrase causal transfer
4. Clinical grounding using available localization annotations
5. Supporting sanity checks for the perturbation pipeline

Answer instability and causal-evidence instability must be analyzed separately. The main claim is not answer consistency alone.

## Core Methodology

For each image-question pair, run baseline inference, extract the answer and answer margin, divide the image into patches, occlude one patch at a time, re-run the VLM, and compute the answer-margin change.

For patch `r` and question `q`:

```text
delta_r(q) = margin_original(q) - margin_occluded_patch_r(q)
```

Repeating this over all patches produces the causal map:

```text
C(I, q)
```

For answer-stable paraphrases, compare causal maps using metrics such as Spearman correlation over patch scores, top-k patch overlap / IoU, and possibly rank overlap.

The strongest methodological addition is cross-paraphrase causal transfer: identify the top causal region under `q_i`, mask that region, then test whether the answer margin also drops under equivalent paraphrases `q_j`.

## Superseded Plan

The earlier broader plan focused on generic heatmap similarity, attention/saliency comparison, broad perturbation faithfulness, and answer-flip analysis. That plan is superseded by the narrower focus on answer-conditioned causal evidence invariance.

Ordinary saliency or attention heatmaps should not be treated as the main explanation mechanism. Causal maps may still be visualized as heatmaps, but their values must come from direct image intervention.

## Non-Goals

* Clinical deployment or medical diagnosis
* Fine-tuning during the initial pilot
* Scaling to many models or datasets before the causal-map prototype is validated
* Treating visually appealing heatmaps as proof of faithfulness
* Generic answer-flip analysis as the main contribution
* Large model leaderboard construction
* SAE mechanistic analysis
* LoRA fine-tuning

## Reproducibility Principles

* Record the dataset version, cohort definition, model, patch grid, occlusion strategy, margin definition, random seed, and Git commit for every experiment.
* Keep restricted data outside version control.
* Do not place patient-level data, images, reports, credentials, or secrets in public files or agent prompts.
* Use the repository as the persistent project context for future AI agents.
