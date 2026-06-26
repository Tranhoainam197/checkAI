import os
import joblib
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

ARTIFACTS_DIR = "artifacts"
N_BINS = 5
FEATURES = ['length', 'word_count', 'avg_word_len']

# Cấu trúc mạng phản ánh quan hệ phụ thuộc thật giữa các đặc trưng:
# length ảnh hưởng word_count (văn bản dài hơn thường nhiều từ hơn),
# word_count ảnh hưởng avg_word_len (cách diễn đạt), cả ba cùng ảnh hưởng label.
EDGES = [
    ('length', 'word_count'),
    ('word_count', 'avg_word_len'),
    ('length', 'label'),
    ('word_count', 'label'),
    ('avg_word_len', 'label'),
]


def discretize(df: pd.DataFrame, bin_edges: dict = None):
    """
    Rời rạc hóa các đặc trưng liên tục thành N_BINS mức theo PHÂN VỊ (qcut),
    phù hợp với phân phối lệch của length/word_count (nhiều mẫu ngắn, ít mẫu dài).
    Nếu bin_edges=None thì tính mới (lúc train), ngược lại dùng lại (lúc predict).
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
    Huấn luyện Bayesian Network với cấu trúc phản ánh phụ thuộc giữa các đặc trưng
    cấu trúc văn bản (length, word_count, avg_word_len) và nhãn (label).
    Dùng BayesianEstimator với prior BDeu (equivalent_sample_size=10).
    """
    df_disc, bin_edges = discretize(df_struct[FEATURES + ['label']])
    df_disc['label'] = df_disc['label'].astype(int)

    model = BayesianNetwork(EDGES)
    model.fit(
        df_disc,
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10
    )

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
    joblib.dump({'bin_edges': bin_edges}, os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
    print("   Bayesian Network đã lưu thành công.")


def predict_bayesian_network(features: dict) -> float:
    """
    Dự đoán xác suất văn bản là AI (label=1) từ đặc trưng cấu trúc,
    bằng suy luận chính xác (Variable Elimination) trên toàn bộ mạng.
    Trả về float trong [0, 1].
    """
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
    metadata = joblib.load(os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
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
        return 0.5  # fallback nếu suy luận thất bại (ví dụ tổ hợp evidence chưa từng thấy)