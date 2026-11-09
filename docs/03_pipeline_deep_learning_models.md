# 03 — Transformer Experiments

## Purpose and implemented models

Transformer training is implemented in notebooks 06, 07, 10, and 11:

- `06_modernBERT_deBERTa.ipynb`: ModernBERT and DeBERTa-v3 experiments.
- `07_DistilBERT_RoBERTa.ipynb`: DistilBERT and RoBERTa experiments.
- `10_RQ3_03_retrain_ModernBERT_DeBERTa.ipynb` and `11_RQ3_04_retrain_DistilBERT_RoBERTa.ipynb`: retraining on data prepared for RQ3.

The main notebooks consume essay text directly through each model's tokenizer. They also load a feature-enriched dataset so the label and split metadata remain aligned with the classical experiments; numeric features are not the transformer's text representation.

## Data and split protocol

The notebooks use a stratified 80% training pool and 20% held-out test split with a fixed seed. Stage A creates an inner validation split from the training pool for epoch/model selection and early stopping. Stage B retrains using the selected epoch count before evaluating on the outer test split. Consult each notebook's configuration cell for exact seed, validation fraction, model revision, sequence length, and training parameters.

Keep the outer test set out of epoch selection, threshold tuning, and checkpoint selection. Long essays are handled according to the tokenizer/truncation configuration in the notebook.

## Metrics and checkpoints

The transformer notebooks report precision, recall, and F1, and may also report accuracy or ROC-AUC. They save training summaries, predictions, and model/tokenizer checkpoints to notebook-defined output paths. Checkpoint folders can be large; generated results are local artifacts and are not necessarily tracked by Git.

The shared utility in `src/pipelinebuilder.py` provides a simpler PyTorch training loop for configured transformer models. It is not the same two-stage Hugging Face `Trainer` procedure used by notebooks 06/07/10/11.

## RQ3 external evaluation

Notebook `12_RQ3_06_merge_and_evaluate_external.ipynb` is the current external evaluation notebook. It combines a human holdout with AI essays, loads frozen trained models, and evaluates without fine-tuning on the external benchmark. It reports metrics such as precision, recall, F1, AUROC, human false-positive rate, and subgroup results where inputs are available. Ensure the benchmark is not used for training or threshold selection.
