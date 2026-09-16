# VLM Explainability for Chest X-rays

## Purpose

This project investigates whether a medical vision-language model uses the same visual evidence when semantically equivalent questions produce the same answer for a chest X-ray.

## Working Research Question

When semantically equivalent clinical questions receive the same answer, do medical VLMs rely on the same visual evidence?

## Initial Pilot Scope

* Dataset: MIMIC-Ext-CXR-QBA v1.0.1
* Image source: MIMIC-CXR-JPG, when required for the image files
* Image views: Frontal chest X-rays only
* Initial findings: Pleural effusion and pneumothorax
* Initial cohort: Approximately 50 frontal chest X-rays
* Questions: One finding and one clear anatomical region per question
* Question variants: One original question plus three semantically equivalent paraphrases
* Answers: Short or yes/no answers where possible
* Model: One pretrained, off-the-shelf medical VLM
* Fine-tuning: Not used in the initial pilot
* Explanations: One heatmap or saliency-based explanation method

## Evaluation Dimensions

1. Answer consistency across paraphrased questions
2. Explanation stability across paraphrased questions
3. Clinical grounding using available localization annotations
4. Perturbation-based explanation faithfulness
5. Lightweight explanation sanity checks

Answer instability and explanation instability must be analyzed separately.

## Non-Goals
smok
* Clinical deployment or medical diagnosis
* Fine-tuning during the initial pilot
* Scaling to many models or datasets before the pilot is validated
* Treating visually appealing heatmaps as proof of faithfulness

## Reproducibility Principles

* Record the dataset version, cohort definition, model, explainer, configuration, random seed, and Git commit for every experiment.
* Keep restricted data outside version control.
* Do not place patient-level data, images, reports, credentials, or secrets in public files or agent prompts.
* Use the repository as the persistent project context for future AI agents.
