# Pipeline 01 --- Linear Machine Learning Models

## 1. Purpose

This branch evaluates linear machine-learning models for binary
detection of human-written versus AI-generated essays.

The branch is designed as a strong, interpretable baseline and is
aligned with the project's first two research questions:

1.  Which linguistic features (sentence length, lexical diversity,
    perplexity) most clearly distinguish human-written vs LLM-generated
    essays?
2.  Which model (TF-IDF + LightGBM/XGBoost vs fine-tuned
    DistilBERT/RoBERTa) detects AI-generated text most accurately?

## 2. Input

The branch receives the cleaned dataset produced by the common
data-preparation pipeline.

Primary fields:

-   `text`: original essay text.
-   `text_clean`: normalized text used for text representation.
-   `label`: binary target indicating Human / AI.
-   `word_count`: text-length feature.

The exact source dataset used during development is the DAIGT V2
training dataset.

## 3. Common preprocessing

Before entering a model branch, the data goes through:

1.  Data understanding.
2.  Exploratory data analysis.
3.  Missing-value handling.
4.  Removal of empty or extremely short texts.
5.  Group-wise IQR outlier filtering on `word_count`.
6.  Text normalization:
    -   lowercase;
    -   remove URLs;
    -   remove email addresses;
    -   remove HTML tags;
    -   remove unsupported special characters;
    -   normalize whitespace.
7.  Duplicate removal using `text_clean`.
8.  Final validation.

The cleaned data is then saved as `train_cleaned.csv`.

## 4. Representation

The main classical representation is TF-IDF.

Conceptually:

`text_clean → TF-IDF vectorization → linear classifier`

TF-IDF converts each essay into a numerical feature vector based on the
importance of terms within the document and across the corpus.

Additional linguistic features can be concatenated with the TF-IDF
representation when required by the experiment, such as:

-   sentence length;
-   lexical diversity;
-   word count;
-   perplexity.

## 5. Model

The linear branch is intended for linear classifiers such as Logistic
Regression.

Typical flow:

``` text
Cleaned text
    ↓
Train / validation / test split
    ↓
TF-IDF
    ↓
Optional linguistic features
    ↓
Linear classifier
    ↓
Prediction
    ↓
Precision / Recall
```

The vectorizer and any learned preprocessing parameters must be fitted
only on the training split to avoid data leakage.

## 6. Evaluation

The primary metrics selected for the project are:

-   **Precision** --- how many texts predicted as AI are actually AI.
-   **Recall** --- how many AI-generated texts are successfully
    detected.

Recall is especially important when the objective is to reduce missed
AI-generated essays.

The same evaluation protocol should be used across model branches so
that the comparison is fair.

## 7. Role in the overall research

This branch provides the classical linear baseline.

Its results are compared with tree-based models and transformer-based
deep-learning models to determine whether more complex models provide a
meaningful improvement.

## 8. Expected output

Each experiment should produce:

-   trained model;
-   validation/test predictions;
-   Precision;
-   Recall;
-   experiment configuration;
-   saved vectorizer/feature transformation when applicable;
-   results table for cross-model comparison.
