"""
pipelinebuilder.py
==================

Module xây dựng các pipeline huấn luyện và đánh giá mô hình phân loại văn bản
(AI-generated vs Human-written) cho dự án AI-Generated Text (Essay) Detection.

========================================================================
ĐỒNG BỘ VỚI DỰ ÁN:
- Đường dẫn: Sử dụng các đường dẫn từ src.path
- Tên cột chuẩn: label, prompt_name, source, text_clean, word_count_clean
- Định dạng nhãn: Binary 0 = Human, 1 = AI-generated (tự động chuyển từ string/bool)
========================================================================

Các loại pipeline được hỗ trợ:
1. Pipeline hồi quy tuyến tính (Logistic Regression)
2. Pipeline mô hình cây quyết định (Random Forest, XGBoost, LightGBM)
3. Pipeline Deep Learning (DistilBERT, RoBERTa, NeoBERTa, DeBERTa)

Tính năng chính:
- Tự động xác thực và chuyển đổi nhãn về định dạng int 0/1
- Trích xuất 9 đặc trưng ngôn ngữ từ text_clean
- Tích hợp TF-IDF + đặc trưng ngôn ngữ cho mô hình truyền thống
- API nhất quán: fit, predict, predict_proba, evaluate
- Đánh giá đầy đủ: Accuracy, Precision, Recall, F1, ROC-AUC
- Hỗ trợ tùy chỉnh classification threshold
- Kiểm soát data leakage: transformer chỉ fit trên tập train
- ExperimentTracker: ghi lại kết quả thí nghiệm
- Phân tích feature importance (RQ1)
- Đánh giá cross-LLM generalization (RQ3)
"""

import re
import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# XGBoost
try:
    import xgboost as xgb 
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

# Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    from tqdm.auto import tqdm
    TRANSFORMERS_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# TÍCH HỢP VỚI DỰ ÁN: Đường dẫn và tên cột chuẩn
# =============================================================================

# Tên cột chuẩn của dự án
STANDARD_COLUMNS = {
    "text": "text_clean",
    "label": "label",
    "source": "source",
    "prompt": "prompt_name",
    "word_count": "word_count_clean"
}

# Giá trị nhãn chuẩn
LABEL_HUMAN = 0
LABEL_AI = 1
VALID_LABELS = {LABEL_HUMAN, LABEL_AI}

# Đường dẫn từ src.path (hoặc fallback nếu import không được)
try:
    from src.path import (
        PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        RAW_DATA_FILE, PROCESSED_DATA_FILE
    )
    PATH_IMPORTED = True
except ImportError:
    _MODULE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = _MODULE_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RAW_DATA_FILE = RAW_DATA_DIR / "raw_data.csv"
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
    PATH_IMPORTED = False

# Thư mục mặc định
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RESULTS_DIR, MODELS_DIR, EXPERIMENTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# HÀM TIỆN ÍCH: XỬ LÝ LABEL VÀ DỮ LIỆU
# =============================================================================

def validate_and_convert_label(y: Union[pd.Series, np.ndarray, List]) -> pd.Series:
    """
    Xác thực và chuyển đổi nhãn về định dạng binary int 0/1.

    Hỗ trợ đầu vào:
    - String: "0", "1", "0.0", "1.0", "True", "False", "human", "ai"
    - Boolean: True/False
    - Số: 0, 1, 0.0, 1.0

    Returns:
        pd.Series với dtype int, chỉ chứa 0 và 1

    Raises:
        ValueError: Nếu có giá trị không thể chuyển thành 0/1
    """
    y_series = pd.Series(y).copy()

    # Nếu đã là int và chỉ chứa 0/1, trả về ngay
    if pd.api.types.is_integer_dtype(y_series):
        unique_vals = set(y_series.unique())
        if unique_vals.issubset(VALID_LABELS):
            return y_series.astype(int)

    # Map các giá trị chuỗi phổ biến
    str_map = {
        "0": LABEL_HUMAN, "1": LABEL_AI,
        "0.0": LABEL_HUMAN, "1.0": LABEL_AI,
        "false": LABEL_HUMAN, "true": LABEL_AI,
        "human": LABEL_HUMAN, "ai": LABEL_AI,
        "ai-generated": LABEL_AI, "gpt": LABEL_AI,
        "llm": LABEL_AI, "machine": LABEL_AI,
    }

    def convert_value(val):
        if pd.isna(val):
            return np.nan
        # Nếu là boolean
        if isinstance(val, (bool, np.bool_)):
            return LABEL_AI if val else LABEL_HUMAN
        # Nếu là số
        if isinstance(val, (int, float, np.integer, np.floating)):
            num = float(val)
            if num == 0.0:
                return LABEL_HUMAN
            elif num == 1.0:
                return LABEL_AI
            return np.nan
        # Nếu là chuỗi
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower in str_map:
                return str_map[val_lower]
            # Thử chuyển thành số
            try:
                num = float(val_lower)
                if num == 0.0:
                    return LABEL_HUMAN
                elif num == 1.0:
                    return LABEL_AI
            except ValueError:
                pass
        return np.nan

    y_converted = y_series.map(convert_value)

    # Kiểm tra giá trị không hợp lệ
    invalid_mask = y_converted.isna()
    if invalid_mask.any():
        invalid_samples = y_series[invalid_mask].unique().tolist()[:10]
        raise ValueError(
            f"Có {invalid_mask.sum()} giá trị label không hợp lệ. "
            f"Các giá trị không hợp lệ: {invalid_samples}\n"
            f"Label phải là binary 0/1 (hoặc chuỗi/số tương đương)."
        )

    return y_converted.astype(int)


def validate_input_data(X: pd.DataFrame, config: "PipelineConfig") -> None:
    """
    Kiểm tra dữ liệu đầu vào có đủ cột văn bản không.

    Raises:
        ValueError: Nếu thiếu cột văn bản
    """
    if config.text_column not in X.columns:
        raise ValueError(
            f"Thiếu cột văn bản '{config.text_column}' trong dữ liệu đầu vào.\n"
            f"Các cột hiện có: {list(X.columns)}\n"
            f"Bạn có thể cấu hình text_column khi tạo pipeline: "
            f"PipelineFactory.create('...', text_column='tên_cột_text')"
        )


def load_cleaned_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu đã được làm sạch và tự động chuyển label về int 0/1.
    """
    if filepath is None:
        filepath = PROCESSED_DATA_FILE

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu clean: {filepath.resolve()}\n"
            f"Hãy chạy notebook data cleaning trước."
        )

    df = pd.read_csv(filepath, encoding="utf-8-sig")

    required = [STANDARD_COLUMNS["text"], STANDARD_COLUMNS["label"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")

    # Tự động chuyển label về int 0/1
    df[STANDARD_COLUMNS["label"]] = validate_and_convert_label(df[STANDARD_COLUMNS["label"]])

    print(f"✓ Đã load dữ liệu từ: {filepath.resolve()}")
    print(f"  - {len(df):,} dòng × {len(df.columns)} cột")
    print(f"  - Label đã chuyển thành int 0/1")

    return df


def filter_by_source(df: pd.DataFrame, sources: Union[str, List[str]]) -> pd.DataFrame:
    """Lọc dữ liệu theo nguồn (source). Phục vụ RQ3."""
    if STANDARD_COLUMNS["source"] not in df.columns:
        raise ValueError(f"DataFrame không có cột '{STANDARD_COLUMNS['source']}'")
    if isinstance(sources, str):
        sources = [sources]
    filtered = df[df[STANDARD_COLUMNS["source"]].isin(sources)].copy()
    print(f"✓ Lọc theo nguồn {sources}: {len(filtered):,} dòng")
    return filtered


def filter_by_prompt(df: pd.DataFrame, prompts: Union[str, List[str]]) -> pd.DataFrame:
    """Lọc dữ liệu theo chủ đề (prompt_name)."""
    if STANDARD_COLUMNS["prompt"] not in df.columns:
        raise ValueError(f"DataFrame không có cột '{STANDARD_COLUMNS['prompt']}'")
    if isinstance(prompts, str):
        prompts = [prompts]
    filtered = df[df[STANDARD_COLUMNS["prompt"]].isin(prompts)].copy()
    print(f"✓ Lọc theo chủ đề {prompts}: {len(filtered):,} dòng")
    return filtered


def split_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Tách thành features (chứa text_clean) và label (đã chuyển int 0/1)."""
    X = df[[STANDARD_COLUMNS["text"]]]
    y = validate_and_convert_label(df[STANDARD_COLUMNS["label"]])
    return X, y


# =============================================================================
# CẤU HÌNH CÁC PIPELINE
# =============================================================================

@dataclass
class PipelineConfig:
    """Cấu hình chung. Mặc định đồng bộ với tên cột chuẩn của dự án."""
    text_column: str = STANDARD_COLUMNS["text"]
    label_column: str = STANDARD_COLUMNS["label"]
    use_linguistic_features: bool = True
    use_tfidf: bool = True
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True
    classification_threshold: float = 0.5
    random_state: int = 42


@dataclass
class LinearConfig(PipelineConfig):
    """Cấu hình Logistic Regression."""
    C: float = 1.0
    penalty: str = "l2"
    max_iter: int = 2000
    solver: str = "liblinear"
    class_weight: Optional[str] = None


@dataclass
class TreeConfig(PipelineConfig):
    """Cấu hình mô hình cây quyết định."""
    model_type: str = "xgboost"

    # Random Forest
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_class_weight: Optional[str] = None

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0

    # LightGBM
    lgb_n_estimators: int = 300
    lgb_max_depth: int = -1
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.0
    lgb_reg_lambda: float = 0.0
    lgb_verbose: int = -1


@dataclass
class DLConfig(PipelineConfig):
    """Cấu hình Deep Learning."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    freeze_base: bool = False
    freeze_layers: int = 0


# =============================================================================
# CUSTOM TRANSFORMER: TRÍCH XUẤT ĐẶC TRƯNG NGÔN NGỮ
# =============================================================================

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Trích xuất 9 đặc trưng ngôn ngữ từ text_clean để phục vụ RQ1:
    1. avg_sentence_length   - Độ dài trung bình câu
    2. sentence_length_std   - Độ biến thiên độ dài câu
    3. avg_word_length       - Độ dài trung bình từ
    4. ttr                   - Đa dạng từ vựng (Type-Token Ratio)
    5. avg_complexity        - Độ phức tạp câu (dấu phẩy, chấm phẩy...)
    6. function_word_ratio   - Tỷ lệ từ chức năng
    7. content_word_ratio    - Tỷ lệ từ nội dung
    8. punctuation_density   - Mật độ dấu câu
    9. doc_length_log        - Độ dài văn bản (log scale)
    """

    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 'my',
            'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if',
            'about', 'up', 'out', 'into', 'over', 'after', 'before', 'between',
            'through', 'during', 'without', 'within', 'along', 'across', 'behind',
            'beyond', 'among', 'around', 'toward', 'upon'
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[List[str], pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        features = []
        for text in X:
            if not isinstance(text, str) or len(text.strip()) == 0:
                features.append([0.0] * 9)
                continue
            features.append(self._extract_single(text))

        return np.array(features, dtype=np.float32)

    def _extract_single(self, text: str) -> List[float]:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)

        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        num_words = max(len(words), 1)
        unique_words = set(words)

        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0] or [1]

        avg_sentence_length = np.mean(sentence_lengths)
        sentence_length_std = np.std(sentence_lengths)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
        ttr = len(unique_words) / num_words

        complexity_markers = len(re.findall(r'[,;:—\-]', text))
        avg_complexity = complexity_markers / num_sentences

        function_word_count = sum(1 for w in words if w in self.stop_words)
        function_word_ratio = function_word_count / num_words
        content_word_ratio = 1.0 - function_word_ratio

        punctuation_count = len(re.findall(r'[.!?,;:—\-]', text))
        punctuation_density = punctuation_count / num_words

        doc_length_log = np.log1p(num_words)

        return [
            avg_sentence_length, sentence_length_std, avg_word_length,
            ttr, avg_complexity, function_word_ratio, content_word_ratio,
            punctuation_density, doc_length_log
        ]

    def get_feature_names(self) -> List[str]:
        return [
            "avg_sentence_length", "sentence_length_std", "avg_word_length",
            "ttr", "avg_complexity", "function_word_ratio", "content_word_ratio",
            "punctuation_density", "doc_length_log"
        ]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.get_feature_names())


# =============================================================================
# LỚP CƠ SỐ
# =============================================================================

class BasePipeline(ABC):
    """Lớp cơ sở trừu tượng định nghĩa API nhất quán."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.is_fitted = False
        self._pipeline_type = self.__class__.__name__

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> "BasePipeline":
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.config.classification_threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Đánh giá đầy đủ các chỉ số. Tự động chuyển đổi label nếu cần."""
        y_test_clean = validate_and_convert_label(y_test)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        y_proba_ai = y_proba[:, 1]

        results = {
            "accuracy": float(accuracy_score(y_test_clean, y_pred)),
            "precision": float(precision_score(y_test_clean, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test_clean, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test_clean, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test_clean, y_proba_ai)),
            "classification_report": classification_report(y_test_clean, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_clean, y_pred).tolist(),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "pipeline_type": self._pipeline_type,
            "threshold": self.config.classification_threshold,
            "timestamp": datetime.now().isoformat()
        }
        return results

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BasePipeline":
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None


# =============================================================================
# PIPELINE MÔ HÌNH TRUYỀN THỐNG
# =============================================================================

class TraditionalPipeline(BasePipeline):
    """Pipeline cho Logistic Regression, Random Forest, XGBoost, LightGBM."""

    def __init__(self, config: PipelineConfig, classifier: Any, model_name: str):
        super().__init__(config)
        self.classifier = classifier
        self.model_name = model_name
        self._pipeline_type = f"TraditionalPipeline({model_name})"
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        transformers = []

        if self.config.use_tfidf:
            tfidf = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                sublinear_tf=self.config.tfidf_sublinear_tf
            )
            transformers.append(("tfidf", tfidf, self.config.text_column))

        if self.config.use_linguistic_features:
            linguistic = SkPipeline([
                ("extract", LinguisticFeatureExtractor()),
                ("scaler", StandardScaler())
            ])
            transformers.append(("linguistic", linguistic, self.config.text_column))

        if len(transformers) == 0:
            raise ValueError("Ít nhất một trong use_tfidf hoặc use_linguistic_features phải là True")

        preprocessor = ColumnTransformer(transformers=transformers)
        self.pipeline = SkPipeline([
            ("preprocessor", preprocessor),
            ("classifier", self.classifier)
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> "TraditionalPipeline":
        """Huấn luyện pipeline. Tự động chuyển label và kiểm tra dữ liệu."""
        validate_input_data(X_train, self.config)
        y_train_clean = validate_and_convert_label(y_train)

        fit_params = {}
        if X_val is not None and y_val is not None:
            validate_input_data(X_val, self.config)
            y_val_clean = validate_and_convert_label(y_val)

            # Fit preprocessor trên train trước, sau đó transform val cho early stopping
            self.pipeline.named_steps["preprocessor"].fit(X_train, y_train_clean)
            X_val_processed = self.pipeline.named_steps["preprocessor"].transform(X_val)

            if self.model_name == "xgboost" and hasattr(self.classifier, "eval_set"):
                fit_params = {
                    "classifier__eval_set": [(X_val_processed, y_val_clean)],
                    "classifier__verbose": False
                }
            elif self.model_name == "lightgbm" and hasattr(self.classifier, "eval_set"):
                fit_params = {
                    "classifier__eval_set": [(X_val_processed, y_val_clean)],
                    "classifier__callbacks": [lgb.log_evaluation(period=0)]
                }

        if fit_params:
            self.pipeline.named_steps["classifier"].fit(
                self.pipeline.named_steps["preprocessor"].transform(X_train), y_train_clean, **fit_params
            )
        else:
            self.pipeline.fit(X_train, y_train_clean)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được huấn luyện. Gọi fit() trước.")
        validate_input_data(X, self.config)
        return self.pipeline.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "pipeline": self.pipeline,
                "model_name": self.model_name,
                "is_fitted": self.is_fitted,
                "_pipeline_type": self._pipeline_type
            }, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TraditionalPipeline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(data["config"], classifier=None, model_name=data["model_name"])
        instance.pipeline = data["pipeline"]
        instance.is_fitted = data["is_fitted"]
        instance._pipeline_type = data.get("_pipeline_type", instance._pipeline_type)
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None

        classifier = self.pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            importance = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "feature_importances_"):
            importance = classifier.feature_importances_
        else:
            return None

        preprocessor = self.pipeline.named_steps["preprocessor"]
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        if len(feature_names) != len(importance):
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return {name: float(imp) for name, imp in zip(feature_names, importance)}


# =============================================================================
# PIPELINE DEEP LEARNING
# =============================================================================

class TextDataset(Dataset):
    """Dataset tùy chỉnh cho PyTorch DataLoader."""

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 tokenizer=None, max_length: int = 256):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DeepLearningPipeline(BasePipeline):
    """Pipeline cho DistilBERT, RoBERTa, NeoBERTa, DeBERTa..."""

    def __init__(self, config: DLConfig):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Cần cài đặt transformers và torch.")
        if not TORCH_AVAILABLE:
            raise ImportError("Cần cài đặt torch.")

        super().__init__(config)
        self.config = config  # type: DLConfig
        self.tokenizer = None
        self.model = None
        self.device = torch.device(config.device)
        self.training_history = []
        self._pipeline_type = f"DeepLearningPipeline({config.model_name.split('/')[-1]})"

    def _init_model_and_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=2
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.config.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        elif self.config.freeze_layers > 0:
            self._freeze_encoder_layers(self.config.freeze_layers)

        self.model.to(self.device)

    def _freeze_encoder_layers(self, num_layers: int) -> None:
        base_model = self.model.base_model
        if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
            layers = base_model.encoder.layer
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "layer"):
            layers = base_model.transformer.layer
        else:
            warnings.warn(f"Không tìm thấy encoder để freeze cho {self.config.model_name}")
            return

        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> "DeepLearningPipeline":
        """Huấn luyện mô hình. Tự động chuyển label và kiểm tra dữ liệu."""
        validate_input_data(X_train, self.config)
        y_train_clean = validate_and_convert_label(y_train)

        if self.tokenizer is None:
            self._init_model_and_tokenizer()

        train_texts = X_train[self.config.text_column].tolist()
        train_labels = y_train_clean.tolist()
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            validate_input_data(X_val, self.config)
            y_val_clean = validate_and_convert_label(y_val)
            val_texts = X_val[self.config.text_column].tolist()
            val_labels = y_val_clean.tolist()
            val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        self.training_history = []
        accumulation_steps = self.config.gradient_accumulation_steps

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            optimizer.zero_grad()
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}", leave=False
            )

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / accumulation_steps
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

            if (step + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            epoch_log = {"epoch": epoch + 1, "train_loss": avg_loss}

            if val_loader is not None:
                val_loss, val_acc, val_auc = self._evaluate_val(val_loader)
                epoch_log.update({"val_loss": val_loss, "val_accuracy": val_acc, "val_roc_auc": val_auc})
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{self.config.epochs} | train_loss={avg_loss:.4f}")

            self.training_history.append(epoch_log)

        self.is_fitted = True
        return self

    def _evaluate_val(self, val_loader: "DataLoader") -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        self.model.train()
        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            roc_auc = 0.0

        return avg_loss, accuracy, roc_auc

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được huấn luyện. Gọi fit() trước.")
        validate_input_data(X, self.config)

        self.model.eval()
        texts = X[self.config.text_column].tolist()
        dataset = TextDataset(texts, None, self.tokenizer, self.config.max_length)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, path: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise ValueError("Chưa thể lưu pipeline chưa được huấn luyện.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        extra = {
            "config": self.config, "is_fitted": self.is_fitted,
            "training_history": self.training_history, "_pipeline_type": self._pipeline_type
        }
        with open(path / "pipeline_extra.pkl", "wb") as f:
            pickle.dump(extra, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeepLearningPipeline":
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError("Cần cài đặt transformers và torch.")
        path = Path(path)
        with open(path / "pipeline_extra.pkl", "rb") as f:
            extra = pickle.load(f)
        instance = cls(extra["config"])
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
        instance.model.to(instance.device)
        instance.is_fitted = extra["is_fitted"]
        instance.training_history = extra.get("training_history", [])
        instance._pipeline_type = extra.get("_pipeline_type", instance._pipeline_type)
        return instance


# =============================================================================
# FACTORY
# =============================================================================

class PipelineFactory:
    """
    Factory tạo pipeline nhất quán.
    Hỗ trợ: logistic_regression, random_forest, xgboost, lightgbm,
            distilbert, roberta, neobert, deberta, custom:<huggingface_model>
    """

    DL_MODEL_MAP = {
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "neobert": "vinai/neobert-base",
        "deberta": "microsoft/deberta-v3-base",
    }

    @staticmethod
    def create(pipeline_type: str, **kwargs) -> BasePipeline:
        pipeline_type = pipeline_type.lower().strip()

        # --- Logistic Regression ---
        if pipeline_type == "logistic_regression":
            config = LinearConfig(**kwargs)
            classifier = LogisticRegression(
                C=config.C, penalty=config.penalty, max_iter=config.max_iter,
                solver=config.solver, class_weight=config.class_weight,
                random_state=config.random_state
            )
            return TraditionalPipeline(config, classifier, model_name="logistic_regression")

        # --- Random Forest ---
        elif pipeline_type == "random_forest":
            config = TreeConfig(model_type="random_forest", **kwargs)
            classifier = RandomForestClassifier(
                n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth,
                min_samples_split=config.rf_min_samples_split,
                min_samples_leaf=config.rf_min_samples_leaf,
                class_weight=config.rf_class_weight,
                random_state=config.random_state, n_jobs=-1
            )
            return TraditionalPipeline(config, classifier, model_name="random_forest")

        # --- XGBoost ---
        elif pipeline_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("Cần cài đặt xgboost: pip install xgboost")
            config = TreeConfig(model_type="xgboost", **kwargs)
            classifier = xgb.XGBClassifier(
                n_estimators=config.xgb_n_estimators, max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate, subsample=config.xgb_subsample,
                colsample_bytree=config.xgb_colsample_bytree,
                reg_alpha=config.xgb_reg_alpha, reg_lambda=config.xgb_reg_lambda,
                random_state=config.random_state, use_label_encoder=False,
                eval_metric="logloss", n_jobs=-1
            )
            return TraditionalPipeline(config, classifier, model_name="xgboost")

        # --- LightGBM ---
        elif pipeline_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("Cần cài đặt lightgbm: pip install lightgbm")
            config = TreeConfig(model_type="lightgbm", **kwargs)
            classifier = lgb.LGBMClassifier(
                n_estimators=config.lgb_n_estimators, max_depth=config.lgb_max_depth,
                learning_rate=config.lgb_learning_rate, num_leaves=config.lgb_num_leaves,
                subsample=config.lgb_subsample, colsample_bytree=config.lgb_colsample_bytree,
                reg_alpha=config.lgb_reg_alpha, reg_lambda=config.lgb_reg_lambda,
                random_state=config.random_state, verbose=config.lgb_verbose, n_jobs=-1
            )
            return TraditionalPipeline(config, classifier, model_name="lightgbm")

        # --- Deep Learning ---
        elif pipeline_type in PipelineFactory.DL_MODEL_MAP:
            model_name = PipelineFactory.DL_MODEL_MAP[pipeline_type]
            kwargs.setdefault("model_name", model_name)
            config = DLConfig(**kwargs)
            return DeepLearningPipeline(config)

        elif pipeline_type.startswith("custom:"):
            custom_model_name = pipeline_type.split(":", 1)[1].strip()
            kwargs.setdefault("model_name", custom_model_name)
            config = DLConfig(**kwargs)
            return DeepLearningPipeline(config)

        else:
            supported = (
                ["logistic_regression", "random_forest", "xgboost", "lightgbm"] +
                list(PipelineFactory.DL_MODEL_MAP.keys()) + ["custom:<huggingface_model>"]
            )
            raise ValueError(
                f"Loại pipeline '{pipeline_type}' không được hỗ trợ.\n"
                f"Các loại được hỗ trợ: {', '.join(supported)}"
            )

    @staticmethod
    def list_supported() -> List[str]:
        return (
            ["logistic_regression", "random_forest", "xgboost", "lightgbm"] +
            list(PipelineFactory.DL_MODEL_MAP.keys()) + ["custom:<huggingface_model>"]
        )


# =============================================================================
# EXPERIMENT TRACKER
# =============================================================================

class ExperimentTracker:
    """Theo dõi và ghi lại các thí nghiệm để so sánh (phục vụ RQ2)."""

    def __init__(self, results_dir: Union[str, Path] = EXPERIMENTS_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[Dict[str, Any]] = []

    def log_experiment(self, experiment_name: str, pipeline: BasePipeline,
                       metrics: Dict[str, Any], extra_info: Optional[Dict[str, Any]] = None) -> None:
        metrics_clean = {k: v for k, v in metrics.items()
                         if k not in ("y_pred", "y_proba", "classification_report", "confusion_matrix")}
        try:
            config_dict = self._make_serializable(asdict(pipeline.config))
        except Exception:
            config_dict = str(pipeline.config)

        experiment = {
            "experiment_name": experiment_name,
            "pipeline_type": pipeline._pipeline_type,
            "config": config_dict,
            "metrics": metrics_clean,
            "extra_info": extra_info or {},
            "timestamp": datetime.now().isoformat()
        }
        self.experiments.append(experiment)
        self._save_to_file(experiment, experiment_name)

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def _save_to_file(self, experiment: Dict[str, Any], experiment_name: str) -> None:
        safe_name = re.sub(r'[^\w\-]', '_', experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{safe_name}_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(experiment), f, indent=2, ensure_ascii=False)

    def save_summary(self, filename: str = "all_experiments.json") -> None:
        filepath = self.results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(self.experiments), f, indent=2, ensure_ascii=False)

    def get_summary_df(self) -> pd.DataFrame:
        rows = []
        for exp in self.experiments:
            row = {
                "experiment_name": exp["experiment_name"],
                "pipeline_type": exp["pipeline_type"],
                **exp["metrics"],
                **{f"extra_{k}": v for k, v in exp.get("extra_info", {}).items()},
                "timestamp": exp["timestamp"]
            }
            rows.append(row)
        return pd.DataFrame(rows)


# =============================================================================
# HÀM TIỆN ÍCH CHO NGHIÊN CỨU
# =============================================================================

def analyze_feature_importance(pipeline: BasePipeline,
                               top_n: int = 20,
                               group: bool = True) -> pd.DataFrame:
    """Phân tích đặc trưng nào phân biệt rõ nhất (RQ1)."""
    importance = pipeline.get_feature_importance()
    if importance is None:
        print("Pipeline này không hỗ trợ phân tích feature importance.")
        print("(Chỉ TraditionalPipeline mới hỗ trợ)")
        return pd.DataFrame()

    df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    if group:
        def classify(name: str) -> str:
            if name.startswith("tfidf__"):
                return "tfidf_word"
            elif name.startswith("linguistic__"):
                return "linguistic"
            return "other"
        df["feature_type"] = df["feature"].apply(classify)

    return df


def cross_llm_evaluation(pipeline: BasePipeline,
                         datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    """Đánh giá khả năng tổng quát hóa trên các LLM khác nhau (RQ3)."""
    results = []
    for source_name, (X_test, y_test) in datasets.items():
        print(f"Đánh giá trên: {source_name}...")
        metrics = pipeline.evaluate(X_test, y_test)
        results.append({
            "source": source_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"]
        })
    return pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)


def prepare_cross_llm_datasets(df: pd.DataFrame,
                                X: pd.DataFrame,
                                y: pd.Series,
                                source_column: Optional[str] = None) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Chuẩn bị datasets cho đánh giá cross-LLM (RQ3). Tự động nhóm theo source."""
    if source_column is None:
        source_column = STANDARD_COLUMNS["source"]

    if source_column not in df.columns:
        raise ValueError(f"DataFrame không có cột '{source_column}'")

    datasets = {}
    for source in df[source_column].unique():
        mask = df[source_column] == source
        if mask.sum() > 0:
            datasets[str(source)] = (X.loc[mask], y.loc[mask])

    print(f"✓ Đã tạo {len(datasets)} datasets cross-LLM từ các nguồn: {list(datasets.keys())}")
    return datasets


# =============================================================================
# VÍ DỤ SỬ DỤNG NHANH
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PipelineBuilder - Đồng bộ với dự án AI-Generated-Text-Essay-Detection")
    print("=" * 70)

    print(f"\n📁 Đường dẫn:")
    print(f"  - src.path imported: {PATH_IMPORTED}")
    print(f"  - PROJECT_ROOT:      {PROJECT_ROOT.resolve()}")

    print(f"\n📋 Tên cột chuẩn:")
    for k, v in STANDARD_COLUMNS.items():
        print(f"  - {k:12s} → {v}")

    print(f"\n🏷️  Nhãn chuẩn: {LABEL_HUMAN} = Human, {LABEL_AI} = AI-generated")

    print(f"\n🔧 Các pipeline được hỗ trợ:")
    for name in PipelineFactory.list_supported():
        print(f"  ✓ {name}")

    # Demo với dữ liệu mẫu
    print(f"\n" + "=" * 70)
    print("Demo: Tạo pipeline và kiểm tra tự động chuyển label")
    print("=" * 70)

    sample_data = pd.DataFrame({
        STANDARD_COLUMNS["label"]: ["0", "0", "0", "1", "1", "0", "1", "0"],  # String label!
        STANDARD_COLUMNS["prompt"]: ["cars", "tech", "nature", "cars", "tech", "nature", "tech", "nature"],
        STANDARD_COLUMNS["source"]: ["human", "human", "human", "gpt-3.5", "gpt-3.5", "human", "llama", "human"],
        STANDARD_COLUMNS["text"]: [
            "cars have changed how we travel every day making distances shorter",
            "technology continues to evolve at a rapid pace in modern society",
            "the forest is full of life and beautiful sounds from many birds",
            "automobiles have revolutionized transportation by enabling efficient travel",
            "technological advancements have significantly transformed our daily lives",
            "i enjoy walking in the park and listening to the natural sounds around me",
            "the rapid development of technology brings many benefits to humanity",
            "nature provides us with everything we need to live a healthy life"
        ],
        STANDARD_COLUMNS["word_count"]: [12, 11, 13, 10, 11, 14, 12, 11]
    })

    print(f"\nDữ liệu mẫu có label dạng string: {sample_data[STANDARD_COLUMNS['label']].unique().tolist()}")

    # Tự động tách và chuyển label
    X, y = split_features_label(sample_data)
    print(f"✓ Hàm split_features_label đã tự động chuyển label thành int: {y.unique().tolist()}")
    print(f"  Kiểu dữ liệu label: {y.dtype}")

    # Tạo pipeline và huấn luyện
    lr = PipelineFactory.create("logistic_regression", C=1.0)
    print(f"\n✓ Đã tạo: {lr._pipeline_type}")

    lr.fit(X, y)
    metrics = lr.evaluate(X, y)

    print(f"\n📊 Kết quả đánh giá:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1:        {metrics['f1']:.4f}")
    print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Demo validate_and_convert_label với nhiều định dạng
    print(f"\n🔄 Demo validate_and_convert_label với các định dạng khác nhau:")
    test_labels = pd.Series(["0", "1", "True", "False", "1.0", "0.0", "ai", "human"])
    converted = validate_and_convert_label(test_labels)
    demo_df = pd.DataFrame({"input": test_labels, "output": converted})
    print(demo_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Hoàn thành demo!")
    print("=" * 70)
