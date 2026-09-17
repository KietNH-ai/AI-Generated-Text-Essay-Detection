# Pipeline 02 --- Tree-Based Machine Learning Models

## 1. Purpose

This branch evaluates non-linear tree-based machine-learning models for
human-versus-AI essay classification.

It is the second classical branch of the project and is intended to
determine whether non-linear decision boundaries can capture patterns
that a linear classifier cannot.

## 2. Models

The project's tree-based experiments can include:

-   Random Forest;
-   XGBoost;
-   LightGBM, when included in the experiment configuration.

Random Forest is particularly useful as a bagging-based baseline, while
XGBoost and LightGBM provide boosted-tree alternatives.

## 3. Input

The branch receives the same validated dataset as the linear branch.

Potential inputs include:

-   TF-IDF features;
-   word count;
-   sentence-level statistics;
-   lexical diversity;
-   perplexity;
-   other engineered linguistic features defined by the experiment.

The feature set must be documented for every experiment because changing
the representation changes the meaning of the comparison.

## 4. Pipeline

``` text
Cleaned dataset
    ↓
Train / validation / test split
    ↓
Feature engineering
    ↓
Numerical feature matrix
    ↓
Tree-based model
    ↓
Prediction probabilities
    ↓
Classification threshold
    ↓
Precision / Recall
```

For experiments using TF-IDF, the TF-IDF transformer is fitted only on
the training data.

For engineered numerical features, transformations that learn parameters
from data must also be fitted only on the training split.

## 5. Random Forest

Random Forest builds multiple decision trees and aggregates their
predictions.

Its main role in this project is as a robust non-linear baseline and as
a model from which feature importance can be examined.

Typical analysis includes:

-   model performance;
-   feature importance;
-   comparison with Logistic Regression;
-   error analysis.

## 6. XGBoost / LightGBM

Boosted-tree models learn an ensemble of trees sequentially, allowing
later trees to focus on errors made by earlier trees.

For this project they are useful for testing whether non-linear feature
interactions improve AI-text detection.

The experiment should record:

-   model configuration;
-   feature set;
-   threshold;
-   Precision;
-   Recall.

If a threshold other than 0.5 is selected, the threshold-selection
procedure must be performed using training/validation data rather than
the test set.

## 7. Evaluation

The primary project metrics are:

  Metric      Purpose
  ----------- ---------------------------------------------------
  Precision   Measures the reliability of AI predictions
  Recall      Measures the ability to detect AI-generated texts

The test set is used only for final evaluation.

## 8. Feature analysis

Tree-based models can support feature-importance analysis.

This is useful for Research Question 1 because the project wants to
investigate which linguistic signals distinguish human-written and
AI-generated essays.

Feature importance should be interpreted as model-dependent evidence
rather than proof of causal importance.

## 9. Role in the overall research

This branch answers whether a non-linear classical model can outperform
the linear baseline and provides an interpretable bridge between simple
machine learning and transformer-based models.

## 10. Expected output

Each experiment should produce:

-   trained tree model;
-   feature configuration;
-   predictions/probabilities;
-   selected classification threshold;
-   Precision;
-   Recall;
-   feature-importance results when supported;
-   comparison-ready result record.
