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

from data_utils import (
    load_and_prepare_data,
    extract_structural_features,
    expand_vietnamese_samples,
)
from bayesian_network import train_bayesian_network, predict_bayesian_network

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_pipeline(pipeline, X, y, n_splits=3):
    """Đánh giá pipeline bằng StratifiedKFold, trả về mean/std F1 macro."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average='macro'))
    return np.mean(scores), np.std(scores)


# ── Tuning Naive Bayes ────────────────────────────────────────────────────────

def tune_naive_bayes(X_train, y_train, label: str = ""):
    """
    Grid search tìm hyperparameter tốt nhất cho Naive Bayes.
    label: chuỗi mô tả ('EN' hoặc 'VI') để in log rõ hơn.
    """
    param_grid = {
        'tfidf__ngram_range': [(3, 5), (2, 5)],
        'tfidf__max_features': [10000, 15000, 20000],
        'clf__alpha': [0.3, 0.5, 0.7],
    }

    best_score = -1
    best_params = None
    results = []

    limit = min(8000, len(X_train))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), size=limit, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    for params in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb',
                                       ngram_range=params['tfidf__ngram_range'],
                                       max_features=params['tfidf__max_features'])),
            ('clf', MultinomialNB(alpha=params['clf__alpha'])),
        ])
        mean_f1, std_f1 = cross_validate_pipeline(pipeline, X_sub, y_sub)
        results.append({**params, 'cv_f1_mean': round(mean_f1, 4),
                        'cv_f1_std': round(std_f1, 4)})
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    # Lưu kết quả tuning riêng theo ngôn ngữ
    fname = f'tuning_results_{label.lower()}.csv' if label else 'tuning_results.csv'
    (pd.DataFrame(results)
       .sort_values('cv_f1_mean', ascending=False)
       .to_csv(os.path.join(ARTIFACTS_DIR, fname), index=False))
    print(f"   [{label}] Best params: {best_params}  |  CV F1: {best_score:.4f}")

    best_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb',
                                   ngram_range=best_params['tfidf__ngram_range'],
                                   max_features=best_params['tfidf__max_features'])),
        ('clf', MultinomialNB(alpha=best_params['clf__alpha'])),
    ])
    return best_pipeline, best_params


# ── Heuristic ─────────────────────────────────────────────────────────────────

def heuristic_score(text: str) -> float:
    from predictor import heuristic_score as _hs
    return _hs(text)


# ── Meta-features & Meta-model ────────────────────────────────────────────────

def build_meta_features(texts, nb_pipeline):
    """Tính 3 score (nb, bn, heuristic) cho từng văn bản."""
    nb_probs = nb_pipeline.predict_proba(texts)[:, 1]
    bn_scores, h_scores = [], []
    for text in texts:
        try:
            struct = extract_structural_features(text)
            bn_scores.append(predict_bayesian_network(struct))
        except Exception:
            bn_scores.append(0.5)
        h_scores.append(heuristic_score(text))
    return np.column_stack([nb_probs, bn_scores, h_scores])


def train_meta_model(X_train, y_train, nb_pipeline, artifact_name: str, label: str = ""):
    """
    Huấn luyện Logistic Regression trên 3 score (NB, BN, Heuristic)
    để tự học trọng số kết hợp tối ưu từ dữ liệu.

    Cải tiến: train riêng meta-model cho EN và VI,
    thay vì gán tay trọng số cho VI.
    """
    print(f"   [{label}] Tính meta-features trên tập train...")
    limit = min(6000, len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), size=limit, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    meta_X = build_meta_features(X_sub, nb_pipeline)
    meta_model = LogisticRegression(max_iter=1000, C=1.0)
    meta_model.fit(meta_X, y_sub)

    coefs = meta_model.coef_[0]
    print(f"   [{label}] Trọng số — NB: {coefs[0]:.3f} | BN: {coefs[1]:.3f} | Heuristic: {coefs[2]:.3f}")
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, artifact_name))
    return meta_model


# ── Pipeline chính ────────────────────────────────────────────────────────────

def train(path_en: str = "data/AI_Human.csv", path_vi: str = "data/dataset_vi.csv"):

    # 1. Load dữ liệu — trả về EN và VI riêng biệt
    print("1. Chuẩn bị dữ liệu (tách EN / VI riêng biệt)...")
    df_en, df_vi = load_and_prepare_data(path_en, path_vi)
    print(f"   EN: {len(df_en)} mẫu | VI: {len(df_vi)} mẫu")

    # 2. Train/test split riêng từng ngôn ngữ, giữ stratify
    X_en, X_en_test, y_en, y_en_test = train_test_split(
        df_en['text'].values, df_en['label'].values,
        test_size=0.2, stratify=df_en['label'].values, random_state=42,
    )
    X_vi, X_vi_test, y_vi, y_vi_test = train_test_split(
        df_vi['text'].values, df_vi['label'].values,
        test_size=0.2, stratify=df_vi['label'].values, random_state=42,
    )

    # 3. Augment tiếng Việt trên tập train VI (multiplier=3, từ điển mở rộng)
    print("2. Tăng cường dữ liệu tiếng Việt (multiplier=3)...")
    df_vi_train = pd.DataFrame({'text': X_vi, 'label': y_vi})
    df_vi_aug = expand_vietnamese_samples(df_vi_train, multiplier=3)
    X_vi = df_vi_aug['text'].values
    y_vi = df_vi_aug['label'].values
    print(f"   VI train sau augment: {len(X_vi)} mẫu")

    # 4. Bayesian Network — train trên cả EN+VI gộp lại
    #    (BN dùng đặc trưng cấu trúc, không phụ thuộc ngôn ngữ)
    print("3. Huấn luyện Bayesian Network (EN + VI gộp)...")
    X_all_train = np.concatenate([X_en, X_vi])
    y_all_train = np.concatenate([y_en, y_vi])
    struct_all = [extract_structural_features(t) for t in X_all_train]
    df_struct = pd.DataFrame(struct_all)
    df_struct['label'] = y_all_train
    train_bayesian_network(df_struct)

    # 5a. Tuning + train NB cho tiếng Anh
    print("4. Tuning Naive Bayes tiếng Anh...")
    nb_en_pipeline, _ = tune_naive_bayes(X_en, y_en, label="EN")
    print("   Huấn luyện NB tiếng Anh trên toàn bộ train EN...")
    nb_en_pipeline.fit(X_en, y_en)
    joblib.dump(nb_en_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb_en.pkl'))

    # 5b. Tuning + train NB cho tiếng Việt
    print("5. Tuning Naive Bayes tiếng Việt...")
    nb_vi_pipeline, _ = tune_naive_bayes(X_vi, y_vi, label="VI")
    print("   Huấn luyện NB tiếng Việt trên toàn bộ train VI...")
    nb_vi_pipeline.fit(X_vi, y_vi)
    joblib.dump(nb_vi_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb_vi.pkl'))

    # Lưu thêm model_nb.pkl (EN) để tương thích ngược với code cũ
    joblib.dump(nb_en_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    # 6a. Meta-model cho tiếng Anh (học từ dữ liệu EN)
    print("6. Huấn luyện Meta-model tiếng Anh...")
    train_meta_model(X_en, y_en, nb_en_pipeline,
                     artifact_name='meta_model_en.pkl', label="EN")
    # Lưu thêm meta_model.pkl để tương thích ngược
    import shutil
    shutil.copy(os.path.join(ARTIFACTS_DIR, 'meta_model_en.pkl'),
                os.path.join(ARTIFACTS_DIR, 'meta_model.pkl'))

    # 6b. Meta-model cho tiếng Việt (học từ dữ liệu VI — thay vì gán tay)
    print("7. Huấn luyện Meta-model tiếng Việt (từ dữ liệu, không gán tay)...")
    train_meta_model(X_vi, y_vi, nb_vi_pipeline,
                     artifact_name='meta_model_vi.pkl', label="VI")

    # 7. Đánh giá riêng lẻ từng mô hình NB trên tập test
    print("\n8. Đánh giá Naive Bayes EN trên test set:")
    y_pred_en = nb_en_pipeline.predict(X_en_test)
    print(classification_report(y_en_test, y_pred_en, target_names=['Human', 'AI']))

    print("9. Đánh giá Naive Bayes VI trên test set:")
    y_pred_vi = nb_vi_pipeline.predict(X_vi_test)
    print(classification_report(y_vi_test, y_pred_vi, target_names=['Human', 'AI']))

    # 8. Lưu test set gộp để report_utils đánh giá hệ thống kết hợp
    X_test_all = np.concatenate([X_en_test, X_vi_test])
    y_test_all = np.concatenate([y_en_test, y_vi_test])
    pd.DataFrame({'text': X_test_all, 'label': y_test_all}).to_csv(
        os.path.join(ARTIFACTS_DIR, 'test_set.csv'), index=False
    )

    print("\nHoàn thành! Artifacts đã lưu vào 'artifacts/'")
    print("Chạy 'python report_utils.py' để đánh giá hệ thống kết hợp đầy đủ.")


if __name__ == "__main__":
    train()
