import os
import joblib
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

ARTIFACTS_DIR = "artifacts"
N_BINS = 5
FEATURES = [
    'length', 'word_count', 'avg_word_len',
    'sentence_count', 'type_token_ratio', 'repetition_rate',
]

# Cấu trúc mạng cập nhật — quan hệ nhân quả giữa các đặc trưng:
#
#  length ──────────────────────────────────────────────────► label
#    │                                                         ▲
#    └──► word_count ──────────────────────────────────────────┤
#              │                                               │
#              ├──► sentence_count ──────────────────────────►│
#              │                                               │
#              └──► avg_word_len ──────────────────────────── ►│
#                                                              │
#  word_count ──► type_token_ratio ────────────────────────── ►│
#  (nhiều từ → TTR giảm do lặp lại nhiều hơn)                 │
#                                                              │
#  word_count ──► repetition_rate ─────────────────────────── ►│
#  (văn bản dài hơn có nhiều cơ hội lặp từ hơn)               │
#
# Giải thích logic bổ sung:
# - type_token_ratio (TTR): AI đôi khi dùng từ vựng đơn điệu hơn human
# - repetition_rate: AI có xu hướng lặp từ khoá nhiều hơn
# - Cả 2 phụ thuộc vào word_count (văn bản dài → TTR giảm tự nhiên)
#   nên thêm cạnh word_count → type_token_ratio và word_count → repetition_rate

EDGES = [
    ('length',     'word_count'),
    ('word_count', 'sentence_count'),
    ('word_count', 'avg_word_len'),
    ('word_count', 'type_token_ratio'),
    ('word_count', 'repetition_rate'),
    # Tất cả đặc trưng → label
    ('length',           'label'),
    ('word_count',       'label'),
    ('avg_word_len',     'label'),
    ('sentence_count',   'label'),
    ('type_token_ratio', 'label'),
    ('repetition_rate',  'label'),
]

# Cache model và metadata
_model_cache = None
_metadata_cache = None


def _load_model_and_metadata():
    global _model_cache, _metadata_cache
    if _model_cache is None:
        _model_cache = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
        _metadata_cache = joblib.load(os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
    return _model_cache, _metadata_cache


def discretize(df: pd.DataFrame, bin_edges: dict = None):
    """
    Rời rạc hóa đặc trưng liên tục thành N_BINS mức theo phân vị (qcut).
    Phù hợp với phân phối lệch của length/word_count.
    - bin_edges=None: tính mới khi train
    - bin_edges=dict: dùng lại khi predict (đảm bảo nhất quán)
    """
    df = df.copy()
    if bin_edges is None:
        bin_edges = {}
        for feat in FEATURES:
            _, edges = pd.qcut(df[feat], q=N_BINS, labels=False,
                                retbins=True, duplicates='drop')
            bin_edges[feat] = edges
    for feat in FEATURES:
        df[feat] = pd.cut(
            df[feat], bins=bin_edges[feat], labels=False, include_lowest=True
        ).astype('Int64').fillna(0)
    return df, bin_edges


def train_bayesian_network(df_struct: pd.DataFrame):
    """
    Huấn luyện Bayesian Network với 6 đặc trưng cấu trúc + ngữ nghĩa.
    Dùng BayesianEstimator với prior BDeu (equivalent_sample_size=10).
    """
    global _model_cache, _metadata_cache
    _model_cache = None
    _metadata_cache = None

    df_disc, bin_edges = discretize(df_struct[FEATURES + ['label']])
    df_disc['label'] = df_disc['label'].astype(int)

    model = BayesianNetwork(EDGES)
    model.fit(
        df_disc,
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10,
    )

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
    joblib.dump({'bin_edges': bin_edges}, os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
    print("   Bayesian Network (6 đặc trưng) đã lưu thành công.")


def predict_bayesian_network(features: dict) -> float:
    """
    Dự đoán xác suất văn bản là AI (label=1) từ đặc trưng cấu trúc.
    Dùng suy luận chính xác Variable Elimination trên toàn mạng.
    Trả về float trong [0, 1].
    """
    model, metadata = _load_model_and_metadata()
    bin_edges = metadata['bin_edges']

    evidence = {}
    for feat in FEATURES:
        val = features.get(feat, 0)
        edges = bin_edges[feat]
        bin_idx = int(np.searchsorted(edges, val, side='right')) - 1
        bin_idx = max(0, min(bin_idx, N_BINS - 1))
        evidence[feat] = bin_idx

    try:
        infer = VariableElimination(model)
        result = infer.query(variables=['label'], evidence=evidence, show_progress=False)
        return float(result.values[1])
    except Exception:
        return 0.5
