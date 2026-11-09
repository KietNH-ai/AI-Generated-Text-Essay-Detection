# 04 — Project Dataflow and Experiment Map

## Scope and source of truth

The project workflow is implemented primarily as numbered Jupyter notebooks. `src/path.py` defines shared project paths; `src/pipelinebuilder.py` provides a reusable pipeline API but is not the orchestration layer for every notebook. For exact parameters and outputs, use the executed notebook cells and their saved summaries.

Current notebook sequence:

1. `01_(1)_data_understanding.ipynb` — inspect the raw dataset.
2. `01_(2)_visual_raw_data.ipynb` — raw-data visualizations.
3. `02_data_cleanning.ipynb` — clean and save the processed table.
4. `03_visual_cleaned_data.ipynb` — EDA and linguistic-feature summaries.
5. `04_important_feature.ipynb` — RQ1 feature analysis and Random Forest experiments.
6. `05_LG_RF_LGBM_XGB_SVM.ipynb` — classical model comparison.
7. `06_modernBERT_deBERTa.ipynb` and `07_DistilBERT_RoBERTa.ipynb` — transformer experiments.
8. `08_RQ3_01_prepare_human_holdout.ipynb` — prepare RQ3 holdout and leakage-free training data.
9. `09_RQ3_02_retrain_traditional_ML.ipynb` — retrain/evaluate classical models internally.
10. `10_RQ3_03_retrain_ModernBERT_DeBERTa.ipynb` and `11_RQ3_04_retrain_DistilBERT_RoBERTa.ipynb` — retrain transformer models.
11. `12_RQ3_06_merge_and_evaluate_external.ipynb` — assemble the external benchmark and evaluate frozen models.

RQ3 is a completed research workflow according to the project context, but its generated data, checkpoints, and result directory are not present in the current workspace snapshot. The notebook remains as a record/reproduction path and requires its input data and trained artifacts to run end to end.

## Paths and generated artifacts

Shared constants in `src/path.py` resolve to:

- Raw data: `data/raw/raw_data.csv`
- Processed table: `data/processed/processed_data.csv`
- Feature-enriched table: `data/processed/data_with_features.csv`
- Notebook figures: `result/fig/`
- Notebook models: `result/model/`

These files are generated locally and ignored/not guaranteed to be committed. In the current workspace, only the raw CSV is present under `data/`; processed tables, RQ3 benchmark files, and `result/` outputs must be restored or regenerated before running dependent notebooks. Some notebooks also support Colab/Drive paths and define their output folders internally.

`src/pipelinebuilder.py` has its own legacy output defaults (`results/` and `models/`), which differ from `src/path.py`'s `result/` paths. Do not assume its generated artifacts land in the notebook output directories.

## Cleaning and features

Notebook 02 reads the raw CSV, removes rows with missing text/label, cleans whitespace, drops very short essays, applies a label-wise IQR filter to word count (`k=3` in the cleaning cell), normalizes text into `text_clean`, removes duplicates on the cleaned text, and saves `processed_data.csv`. Confirm the actual executed cells before relying on this summary if the cleaning notebook changes.

Notebooks 03–05 perform EDA, feature analysis, and classical-model preparation. Feature definitions vary across the exploratory feature analysis and combined model experiment; use the relevant notebook's feature columns rather than assuming a single universal feature set.

## Evaluation design

The classical comparison in notebook 05 uses a stratified 80/20 train/test split with a fixed seed and five-fold stratified cross-validation on the training data. The transformer notebooks use an outer stratified 80/20 split, an inner validation split for Stage A model/epoch selection, and Stage B retraining before final outer-test evaluation.

The reusable API in `src/pipelinebuilder.py` has its own `evaluate` method and defaults. Its weighted precision/recall/F1 metrics should not be conflated with the positive-class metrics reported in the experiment notebooks.

## RQ3 design

RQ3 preparation isolates 1,000 human essays for external evaluation and builds training inputs that exclude this holdout. The retraining notebooks create fresh model artifacts; notebook 12 combines the holdout with 1,000 AI essays and evaluates frozen models. The benchmark must remain external to training, checkpoint selection, and threshold tuning. Interpret subgroup comparisons only when the required generator/source metadata is available.

## Reproducibility checklist

For a reproducible run, record the dataset version, notebook and cell configuration, random seed, feature set, split indices or manifests, model/tokenizer identifiers, epoch selection, threshold, and output artifact locations. Keep generated artifacts distinct from source inputs and report which metric averaging/positive class was used.
