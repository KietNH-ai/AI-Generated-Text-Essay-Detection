# Dataflow --- AI-Generated Essay Detection

## 1. Overview

The project uses a shared dataflow followed by three independent model
branches.

``` text
                    ┌──────────────────────┐
                    │   Raw Dataset        │
                    │   DAIGT V2           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Understanding   │
                    │ + EDA                │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning        │
                    │ - missing values     │
                    │ - short text         │
                    │ - outliers           │
                    │ - duplicates          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Normalization   │
                    │ → text_clean         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Validation           │
                    │ Clean dataset        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train / Val / Test   │
                    │ Split                │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
      │ Linear ML    │ │ Tree-based   │ │ Deep Learning   │
      │              │ │ ML           │ │ / Transformers  │
      │ TF-IDF       │ │ RF / XGB /   │ │ DistilBERT /    │
      │ + Linear     │ │ LightGBM     │ │ RoBERTa         │
      └──────┬───────┘ └──────┬───────┘ └────────┬────────┘
             │                │                  │
             └────────────────┼──────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Predictions          │
                    │ Human / AI           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Evaluation           │
                    │ Precision / Recall   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Comparison +         │
                    │ Research Analysis    │
                    └──────────────────────┘
```

## 2. Stage 1 --- Raw data

The development notebook uses the DAIGT V2 training dataset.

The core classification fields are the essay `text` and binary `label`.

## 3. Stage 2 --- Data understanding and EDA

The initial analysis checks:

-   dataset shape;
-   column names;
-   data types;
-   memory usage;
-   label distribution;
-   descriptive statistics;
-   source/model distribution when the `source` column exists.

This stage describes the dataset before modifying it.

## 4. Stage 3 --- Data cleaning

The current cleaning procedure contains:

### Missing values

Rows missing either `text` or `label` are removed.

Text values are not imputed because missing essay content cannot be
reliably reconstructed.

### Empty and short texts

Whitespace is removed and text with fewer than 20 words is discarded.

`word_count` is used as a measurement for this filtering stage and is
also retained as a potential model feature.

### Outliers

Outliers are filtered using a group-wise IQR procedure on `word_count`,
grouped by `label`.

The current implementation uses:

`k = 3`

This is performed once; the pipeline should not apply an additional
global outlier filter afterward.

### Text normalization

`text_clean` is created by:

-   converting text to lowercase;
-   removing URLs;
-   removing email addresses;
-   removing HTML tags;
-   retaining letters, numbers, whitespace and basic punctuation;
-   collapsing repeated whitespace.

### Duplicates

Duplicate essays are removed using:

`text_clean`

This catches duplicates that differ only through casing, spacing or
other normalized formatting.

## 5. Stage 4 --- Validation

The cleaned dataset is validated to ensure:

-   no missing `text_clean`;
-   no duplicate `text_clean`;
-   both target classes remain;
-   label distribution is inspected again;
-   `word_count` statistics are inspected again.

The cleaned data is then saved as:

`train_cleaned.csv`

## 6. Stage 5 --- Split

The cleaned dataset is separated into training, validation and test data
before model-specific learning.

The exact split ratio should be kept consistent across experiments.

No test-set information should be used for fitting transformations or
selecting the final model.

## 7. Stage 6 --- Three model branches

### Branch A --- Linear ML

``` text
text_clean
   ↓
TF-IDF
   ↓
optional linguistic features
   ↓
linear classifier
```

### Branch B --- Tree-based ML

``` text
text_clean / engineered features
   ↓
numerical feature matrix
   ↓
Random Forest / XGBoost / LightGBM
```

### Branch C --- Deep Learning

``` text
text_clean
   ↓
Transformer tokenizer
   ↓
DistilBERT / RoBERTa
   ↓
fine-tuning
```

## 8. Stage 7 --- Evaluation

All branches produce predictions on the same evaluation protocol.

Primary metrics:

-   Precision;
-   Recall.

This shared evaluation layer is necessary for a fair comparison.

## 9. Stage 8 --- Research analysis

The final results are used to address three research questions:

1.  Which linguistic features most clearly distinguish human-written and
    LLM-generated essays?
2.  Which model family detects AI-generated text most accurately?
3.  Does a model trained on one LLM retain its performance when
    evaluated on text generated by other LLMs?

The final analysis should distinguish between:

-   model performance;
-   feature-level evidence;
-   cross-LLM generalization;
-   possible false positives on polished human writing.

## 10. Leakage control

The following rule applies throughout the dataflow:

> Anything that learns parameters from data must be fitted only on the
> training set.

This includes TF-IDF vocabulary/weights, scalers, feature-selection
procedures, model parameters and any threshold-selection process.

Validation data is used for model selection and tuning.

The test set is reserved for final reporting.
