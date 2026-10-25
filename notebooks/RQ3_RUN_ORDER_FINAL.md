# RQ3 FINAL — OpenRouter workflow

## Current group status
All RQ3 detectors have already been retrained and their checkpoints have been saved.

Therefore, the group should **not retrain again** unless a checkpoint is missing or invalid.

## Final execution order

### A. Provenance notebooks — already completed
Keep these notebooks as evidence of the retraining method:

1. `08_RQ3_01_prepare_human_holdout.ipynb`
2. `09_RQ3_02_retrain_traditional_ML.ipynb`
3. `10_RQ3_03_retrain_ModernBERT_DeBERTa.ipynb`
4. `11_RQ3_04_retrain_DistilBERT_RoBERTa.ipynb`

Do not use the 2,000-row RQ3 external benchmark to train any of these models.

### B. Start here now — AI generation
Run:

`12_RQ3_05_generate_1000AI_OpenRouter_VERIFIED.ipynb`

Only one key is required:

`OPENROUTER_API_KEY`

Fixed model slugs:
- `openai/gpt-5.6-terra`
- `anthropic/claude-sonnet-5.5`
- `google/gemini-3.8-flash`
- `meta-llama/llama-3.3-70b-instruct`
- `deepseek/deepseek-v4-pro-0813`

Recommended sequence:
1. Fill and verify `data/rq3/rq3_prompts.csv`.
2. `RUN_MODE = "dry_run"`
3. `RUN_MODE = "smoke"` — exactly 5 API calls.
4. Check outputs and OpenRouter Activity/Billing.
5. `RUN_MODE = "production"` — resume until exactly 1,000 AI rows.

Output:
`data/rq3/rq3_ai_1000.csv`

### C. External evaluation
After AI generation is complete, run:

`13_RQ3_06_merge_and_evaluate_external.ipynb`

Inputs:
- `data/rq3/rq3_human_holdout_1000.csv`
- `data/rq3/rq3_ai_1000.csv`
- all RQ3 retrained checkpoints

It produces the 2,000-row external benchmark and evaluates:
- Accuracy
- Precision / Recall / F1
- AUROC
- Human False Positive Rate
- performance by generator family
- performance by generation condition
- performance drop relative to internal evaluation

## Critical rule
After AI generation begins, checkpoints are FROZEN.

Never:
- train on `rq3_human_holdout_1000.csv`
- train on `rq3_ai_1000.csv`
- tune thresholds on the 2,000-row external benchmark
- load an old RQ2 supervised checkpoint in place of an RQ3 retrained checkpoint
