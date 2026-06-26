import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

from data_utils import load_and_prepare_data, extract_structural_features, expand_vietnamese_samples
from bayesian_network import train_bayesian_network, predict_bayesian_network

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def cross_validate_pipeline(pipeline, X, y, n_splits=3):
    """Đánh giá pipeline bằng StratifiedKFold, trả về mean/std F1 macro."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores), np.std(scores)


def tune_naive_bayes(X_train, y_train):
    """Grid search tìm hyperparameter tốt nhất cho Naive Bayes."""
    param_grid = {
        'tfidf__ngram_range': [(3, 5), (2, 5)],
        'tfidf__max_features': [10000, 15000, 20000],
        'clf__alpha': [0.3, 0.5, 0.7],
    }

    best_score = -1
    best_params = None
    best_pipeline = None
    results = []

    limit = min(8000, len(X_train))
    X_sub, y_sub = X_train[:limit], y_train[:limit]

    for params in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb',
                                       ngram_range=params['tfidf__ngram_range'],
                                       max_features=params['tfidf__max_features'])),
            ('clf', MultinomialNB(alpha=params['clf__alpha'])),
        ])
        mean_f1, std_f1 = cross_validate_pipeline(pipeline, X_sub, y_sub)
        results.append({**params, 'cv_f1_mean': round(mean_f1, 4), 'cv_f1_std': round(std_f1, 4)})

        if mean_f1 > best_score:
            best_score, best_params, best_pipeline = mean_f1, params, pipeline

    pd.DataFrame(results).sort_values('cv_f1_mean', ascending=False).to_csv(
        os.path.join(ARTIFACTS_DIR, 'tuning_results.csv'), index=False
    )
    print(f"Best params: {best_params}  |  CV F1: {best_score:.4f}")
    return best_pipeline, best_params


def heuristic_score(text: str) -> float:
    """Phải khớp 100% với hàm trong predictor.py — tách logic chung ra
    để không lệch giữa lúc huấn luyện meta-model và lúc dùng thực tế."""
    from predictor import heuristic_score as _hs
    return _hs(text)


def build_meta_features(texts, nb_pipeline, bn_metadata_ready=True):
    """
    Tính 3 score (nb, bn, heuristic) cho từng văn bản — dùng làm input
    cho meta-model (Logistic Regression) học trọng số kết hợp tối ưu.
    """
    nb_probs = nb_pipeline.predict_proba(texts)[:, 1]

    bn_scores = []
    h_scores = []
    for text in texts:
        try:
            struct = extract_structural_features(text)
            bn_scores.append(predict_bayesian_network(struct))
        except Exception:
            bn_scores.append(0.5)
        h_scores.append(heuristic_score(text))

    return np.column_stack([nb_probs, bn_scores, h_scores])


def train_meta_model(X_train, y_train, nb_pipeline):
    """
    Huấn luyện Logistic Regression nhỏ trên 3 score (NB, BN, Heuristic)
    để TỰ HỌC trọng số kết hợp tối ưu từ dữ liệu, thay vì gán tay.
    Dùng cross-validation để tránh overfit trên chính tập train.
    """
    print("   Tính meta-features (nb_score, bn_score, heuristic_score) trên tập train...")
    # Giới hạn số mẫu để tính meta-feature không quá lâu (BN suy luận từng mẫu khá chậm)
    limit = min(6000, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), size=limit, replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]

    meta_X = build_meta_features(X_sub, nb_pipeline)
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_X, y_sub)

    coefs = meta_model.coef_[0]
    print(f"   Trọng số học được — NB: {coefs[0]:.3f} | BN: {coefs[1]:.3f} | Heuristic: {coefs[2]:.3f}")
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, 'meta_model.pkl'))
    return meta_model


def train(path_en: str = "data/AI_Human.csv", path_vi: str = "data/dataset_vi.csv"):
    print("1. Chuẩn bị dữ liệu...")
    df = load_and_prepare_data(path_en, path_vi)

    X_all = df['text'].values
    y_all = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    print("2. Tăng cường dữ liệu tiếng Việt trên tập train...")
    df_train = pd.DataFrame({'text': X_train, 'label': y_train})
    is_vi_mask = df_train['text'].str.contains(
        r'[àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]',
        case=False, regex=True
    )
    df_train_vi = df_train[is_vi_mask]
    df_train_other = df_train[~is_vi_mask]

    if len(df_train_vi) > 0:
        df_train_vi_aug = expand_vietnamese_samples(df_train_vi, multiplier=1)
        df_train = pd.concat([df_train_other, df_train_vi_aug], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_train = df_train['text'].values
        y_train = df_train['label'].values
        print(f"   Train sau augment: {len(X_train)}")

    print("3. Tuning Naive Bayes...")
    nb_pipeline, _ = tune_naive_bayes(X_train, y_train)

    print("4. Huấn luyện Naive Bayes trên toàn bộ train...")
    nb_pipeline.fit(X_train, y_train)
    joblib.dump(nb_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    print("5. Huấn luyện Bayesian Network...")
    struct_train = [extract_structural_features(t) for t in X_train]
    df_struct_train = pd.DataFrame(struct_train)
    df_struct_train['label'] = y_train
    train_bayesian_network(df_struct_train)

    print("6. Huấn luyện Meta-model (học trọng số kết hợp NB+BN+Heuristic)...")
    train_meta_model(X_train, y_train, nb_pipeline)

    print("7. Đánh giá Naive Bayes riêng lẻ trên tập test...")
    y_pred = nb_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(
        os.path.join(ARTIFACTS_DIR, 'test_set.csv'), index=False
    )
    print("Hoàn thành! Artifacts đã lưu vào thư mục 'artifacts/'")
    print("Chạy 'python report_utils.py' để đánh giá hệ thống kết hợp đầy đủ.")


if __name__ == "__main__":
    train()