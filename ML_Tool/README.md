# Polymer Property Data Extraction and Machine Learning

This repository contains the main data-processing and machine-learning components used to construct polymer property datasets and develop property-prediction models. The workflow covers literature-assisted data extraction, molecular fingerprint generation, model training and evaluation, and prediction of polymer candidates.

## Repository structure

| Folder | Description |
|---|---|
| [`1.LLM-Extractor`](1.LLM-Extractor/README.md) | Extracts polymer compositions and physical properties from PDF articles with keyword filtering and a large language model, then generates standardized and classified CSV tables. |
| [`2.ML`](2.ML/README.md) | Generates molecular fingerprints and trains, tunes, evaluates, and applies regression models for five polymer properties. It also contains plotting scripts and saved model results. |
| `3.ML_Tool` | Contains `Prediction-AB.csv`, the compiled prediction table for polymer A–B combinations, including thermal and mechanical properties. |

## Typical workflow

1. Place literature PDFs in `1.LLM-Extractor/Data/PDFs/` and run the extraction pipeline.
2. Review and curate the extracted records, including polymer names, SMILES representations, units, and target values.
3. Use the scripts in `2.ML/` to generate molecular fingerprints, perform hyperparameter searches, train models with cross-validation, and evaluate predictions.
4. Apply the trained models to candidate polymers and collect the resulting property predictions.

The handoff between literature extraction and machine learning includes data review and SMILES preparation and is therefore not intended to be a completely automatic end-to-end process.

## Usage

Each main folder has its own README and `requirements.txt`. Follow the instructions in the relevant subfolder and run its commands from that subfolder. Separate Python environments are recommended because the extraction and machine-learning components may use different Python and package versions.

LLM credentials must be supplied through environment variables. Do not store API keys or access tokens in source files or committed configuration files.
