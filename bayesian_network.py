import os
import joblib
import numpy as np
import pandas as pd
# SỬA ĐỔI: Dùng DiscreteBayesianNetwork thay cho class cũ đã bị deprecated
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

ARTIFACTS_DIR = "artifacts"
N_BINS = 5
FEATURES = ['length', 'word_count', 'avg_word_len']

_bn_model = None
_bn_metadata = None
_bn_infer = None

def _load_models_to_ram():
    global _bn_model, _bn_metadata, _bn_infer
    if _bn_model is None or _bn_metadata is None:
        model_path = os.path.join(ARTIFACTS_DIR, 'model_bn.pkl')
        meta_path = os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl')
        if os.path.exists(model_path) and os.path.exists(meta_path):
            _bn_model = joblib.load(model_path)
            _bn_metadata = joblib.load(meta_path)
            _bn_infer = VariableElimination(_bn_model)

def discretize(df: pd.DataFrame, bin_edges: dict = None):
    df = df.copy()
    if bin_edges is None:
        bin_edges = {}
        for feat in FEATURES:
            _, edges = pd.cut(df[feat], bins=N_BINS, retbins=True, labels=False, duplicates='drop')
            bin_edges[feat] = edges
            
    for feat in FEATURES:
        df[feat] = pd.cut(
            df[feat], bins=bin_edges[feat], labels=False, include_lowest=True
        ).astype('Int64').fillna(0)
    return df, bin_edges

def train_bayesian_network(df_struct: pd.DataFrame):
    global _bn_model, _bn_metadata, _bn_infer
    
    df_disc, bin_edges = discretize(df_struct[FEATURES + ['label']])
    df_disc['label'] = df_disc['label'].astype(int)

    edges = [(feat, 'label') for feat in FEATURES]
    # SỬA ĐỔI: Khởi tạo mô hình bằng lớp mới của pgmpy
    model = DiscreteBayesianNetwork(edges)

    model.fit(df_disc, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'model_bn.pkl'))
    joblib.dump({'bin_edges': bin_edges}, os.path.join(ARTIFACTS_DIR, 'bn_metadata.pkl'))
    
    _bn_model, _bn_metadata, _bn_infer = None, None, None

def predict_bayesian_network(features: dict) -> float:
    try:
        _load_models_to_ram()
        bin_edges = _bn_metadata['bin_edges']

        evidence = {}
        for feat in FEATURES:
            val = features.get(feat, 0)
            edges = bin_edges[feat]
            bin_idx = int(np.digitize(val, edges[1:-1])) 
            evidence[feat] = max(0, min(bin_idx, N_BINS - 1))

        result = _bn_infer.query(variables=['label'], evidence=evidence, show_progress=False)
        return float(result.values[1])
    
    except Exception:
        return 0.5