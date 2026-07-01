import os
import joblib
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Thêm import hàm trích xuất đặc trưng để tự động xử lý nếu đầu vào là chuỗi
from data_utils import extract_structural_features 

ARTIFACTS_DIR = "artifacts"
N_BINS = 5
FEATURES = [
    'length', 'word_count', 'avg_word_len',
    'sentence_count', 'type_token_ratio', 'repetition_rate',
]

EDGES = [
    ('length', 'label'),
    ('length', 'word_count'),
    ('word_count', 'label'),
    ('word_count', 'sentence_count'),
    ('word_count', 'avg_word_len'),
    ('sentence_count', 'label'),
    ('avg_word_len', 'label'),
    ('word_count', 'type_token_ratio'),
    ('type_token_ratio', 'label'),
    ('word_count', 'repetition_rate'),
    ('repetition_rate', 'label')
]

_model_cache = None
_metadata_cache = None

def _load_model_and_metadata():
    global _model_cache, _metadata_cache
    if _model_cache is None or _metadata_cache is None:
        _model_cache = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
        _metadata_cache = joblib.load(os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
    return _model_cache, _metadata_cache

def discretize(df: pd.DataFrame, bin_edges_dict=None) -> tuple[pd.DataFrame, dict]:
    df_disc = df.copy()
    out_edges = {}
    for feat in FEATURES:
        if bin_edges_dict and feat in bin_edges_dict:
            edges = bin_edges_dict[feat]
            # Đảm bảo biên an toàn cho dữ liệu kiểm thử ngoài khoảng train
            edges = list(edges)
            edges[0] = -np.inf
            edges[-1] = np.inf
        else:
            _, edges = pd.qcut(df[feat], q=N_BINS, retbins=True, duplicates='drop')
            edges = list(edges)
            edges[0] = -np.inf
            edges[-1] = np.inf
            out_edges[feat] = edges
            
        df_disc[feat] = pd.cut(df[feat], bins=edges, labels=False, include_lowest=True)
    return df_disc, out_edges

def train_bayesian_network(df_struct: pd.DataFrame):
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

def predict_bayesian_network(input_data) -> float:
    """
    Dự đoán xác suất văn bản là AI (label=1) từ đặc trưng cấu trúc.
    Hỗ trợ đầu vào linh hoạt: vừa nhận 'dict' đặc trưng, vừa nhận 'str' văn bản thô.
    """
    # SỬA LỖI ĐỒNG BỘ: Nếu đầu vào là chuỗi văn bản thô, tự động trích xuất đặc trưng
    if isinstance(input_data, str):
        features = extract_structural_features(input_data)
    elif isinstance(input_data, dict):
        features = input_data
    else:
        return 0.5

    model, metadata = _load_model_and_metadata()
    bin_edges = metadata['bin_edges']

    evidence = {}
    for feat in FEATURES:
        val = features.get(feat, 0)
        edges = bin_edges[feat]
        
        # Tìm xem giá trị thực thuộc bin nào
        bin_idx = np.digitize([val], edges)[0] - 1
        bin_idx = max(0, min(bin_idx, len(edges) - 2))
        evidence[feat] = bin_idx

    try:
        inference = VariableElimination(model)
        query_result = inference.query(variables=['label'], evidence=evidence, show_progress=False)
        # Lấy xác suất của lớp 1 (AI Generated)
        prob_ai = query_result.values[1]
        return float(prob_ai)
    except Exception:
        # Fallback an toàn nếu suy luận đồ thị gặp lỗi cấu trúc
        return 0.5