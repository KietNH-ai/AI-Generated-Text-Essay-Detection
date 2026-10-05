# AI-Generated Text Essay Detection

## 1. Project Overview

This project focuses on detecting AI-generated essays by comparing traditional machine learning and transformer-based approaches.

The project investigates two main approaches:

* **Traditional Machine Learning**
* **Transformer-Based Models**

The models are trained and evaluated to distinguish between human-written and AI-generated text.

---

## 2. Dataset

The dataset contains human-written and AI-generated essays.

The dataset is divided into:

* **80% Training Set**
* **20% Testing Set**

The split is performed before the model-specific text representation and encoding steps.

---

## 3. Research Questions

### RQ1

Which linguistic features (sentence length, vocabulary diversity, and perplexity) most clearly distinguish human-written essays from LLM-generated essays?

### RQ2

Which model (TF-IDF + LightGBM/XGBoost versus fine-tuned DistilBERT/RoBERTa) achieves the highest accuracy in detecting AI-generated text?

### RQ3

How well does a model trained on text from a specific source or dataset generalize when evaluated on different datasets or writing topics?

---

## 4. Methodology

### 4.1 Data Processing

The dataset is first cleaned and divided into training and testing sets using an **80/20 split**.

After splitting the data, each modeling approach applies its own text processing method.

### 4.2 Model Comparison

The project evaluates two groups of models.

#### Traditional Machine Learning

Text is converted into numerical representations using **TF-IDF and linguistic/stylometric features**.

Five traditional machine learning models are evaluated:

* Logistic Regression
* Random Forest
* SVM
* XGBoost
* LightGBM

#### Transformer-Based Models

Text is tokenized and processed using pre-trained transformer models.

Four transformer models are evaluated:

* DistilBERT
* RoBERTa
* ModernBERT
* DeBERTa-V3

### 4.3 Model Evaluation

The models are evaluated and compared using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## 5. Project Pipeline

```text
                                                 Input Dataset
                              │
                              ▼
                       Data Cleaning
                              │
                              ▼
                    Train / Test Split
                          80% / 20%
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Traditional ML             Transformer-Based
                 │                         │
                 ▼                         ▼
          Vectorized Data            Tokenized Data
                 │                         │
                 ├──→ Logistic             ├──→ DistilBERT  
                 ├──→ Random Forest        ├──→ RoBERTa     
                 ├──→ SVM                  ├──→ ModernBERT  
                 ├──→ XGBoost              ├──→ DeBERTa-V3  
                 ├──→ LightGBM             |
                 │                         │
                 └────────────┬────────────┘
                              ▼
                       Model Evaluation
```

---

## 6. Models

| Approach       | Models              |
| -------------- | ------------------- |
| Traditional ML | Logistic Regression |
| Traditional ML | Random Forest       |
| Traditional ML | SVM                 |
| Traditional ML | XGBoost             |
| Traditional ML | LightGBM            |
| Transformer    | DistilBERT          |
| Transformer    | RoBERTa             |
| Transformer    | ModernBERT          |
| Transformer    | DeBERTa-V3          |

---

## 7. RQ3 Data Generation

The project includes an API-based text generation pipeline to support RQ3.

The pipeline is used to generate AI-written essays from multiple LLM providers and includes:

* API-based text generation
* Retry and error handling
* Rate limiting
* Generation metadata
* Usage tracking
* Resumable generation

The generated texts can be used to evaluate model generalization across different sources and writing topics.

---

## 8. Repository Structure

```text
AI-Generated-Text-Essay-Detection/
│
├── data/
├── docs/
├── notebooks/
├── src/
│
├── generate_rq3_essays.py
├── rq3_generation_config.json
│
├── requirements.txt
├── requirements_rq3_api.txt
├── README.md
└── .gitignore
```

---

## 9. Technologies

* Python
* Scikit-learn
* XGBoost
* LightGBM
* Hugging Face Transformers
* TF-IDF
* LLM APIs

---

## 10. Objective

The objective of this project is to compare traditional machine learning and transformer-based approaches for detecting AI-generated essays and to investigate their ability to distinguish AI-generated text from human-written text.
