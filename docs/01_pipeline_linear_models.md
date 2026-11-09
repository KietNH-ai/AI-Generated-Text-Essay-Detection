# 01 — Linear and Classical ML Experiments

## Purpose

This document describes the implemented classical-model workflow. The executable experiment notebooks are `notebooks/04_important_feature.ipynb` and `notebooks/05_LG_RF_LGBM_XGB_SVM.ipynb`. The reusable pipeline API in `src/pipelinebuilder.py` is a separate utility and is not the runner used by those notebooks.

## Data input

The shared path definitions are in `src/path.py`:

- Raw input: `data/raw/raw_data.csv`
- Cleaned data: `data/processed/processed_data.csv`
- Feature-enriched data: `data/processed/data_with_features.csv`

Notebook 05 reads the cleaned dataset and expects the engineered columns used in its experiment. The processed files are generated locally and are not present in every checkout; run notebook 02 first, and follow notebook 04/05 setup cells for the expected feature table.

The task is binary classification, with `label=0` for human-written text and `label=1` for AI-generated text. The essay text and engineered numeric features are used as predictors.

## Implemented experiments

Notebook 04 investigates linguistic/stylometric features, including distribution comparisons and Random Forest feature analysis. Its feature definitions and displayed result tables are experiment-specific; do not treat its exploratory statistics as a causal explanation.

Notebook 05 compares Logistic Regression, Random Forest, LightGBM, XGBoost, and SVM. Its main representation combines word and bigram TF-IDF (up to 5,000 features) with scaled numeric features. The notebook uses a stratified 80/20 train/test split with a fixed seed and stratified five-fold cross-validation on the training portion. The held-out test portion is used for final metrics.

## Leakage and evaluation

Fit the vectorizer and feature scaler using training data only, then apply the fitted transforms to validation/test data. Keep the held-out test set out of model selection. Notebook 05 reports accuracy, precision, recall, F1, and ROC-AUC where the estimator provides a score. SVM uses its decision scores for ranking metrics.

`src/pipelinebuilder.py` exposes a configurable API (`fit`, `predict`, `predict_proba`, `evaluate`) and uses weighted precision/recall/F1 plus ROC-AUC. Its defaults and paths are not identical to notebook 05; cite the runner actually used when reporting results.

## Outputs

Notebook outputs and saved artifacts depend on the executed cells and local setup. Generated datasets are under `data/processed/`; plots and model artifacts may be written under the paths selected in the notebook. These generated artifacts are not guaranteed to be included in the repository.
