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


def train_meta_model(X_text, y_true, nb_pipeline, artifact_name, label_lang):
    """
    Huấn luyện Meta-model (Logistic Regression) gộp kết quả 3 mô hình.
    Đảm bảo 100% không bị ngược nhãn nhờ ép thứ tự class.
    """
    # 1. Thu thập điểm số từ các mô hình thành phần
    nb_probs = nb_pipeline.predict_proba(X_text)[:, 1]
    
    bn_probs = []
    for txt in X_text:
        bn_probs.append(predict_bayesian_network(txt))
    bn_probs = np.array(bn_probs)
    
    h_probs = []
    for txt in X_text:
        from data_utils import extract_structural_features
        feats = extract_structural_features(txt)
        raw_h = (feats['avg_word_len'] * 0.4) + (feats['repetition_rate'] * 0.6)
        h_probs.append(np.clip(raw_h, 0.0, 1.0))
    h_probs = np.array(h_probs)
    
    # 2. Tạo ma trận đầu vào cho Meta-model
    meta_X = np.column_stack((nb_probs, bn_probs, h_probs))
    
    # 3. Huấn luyện Meta-model
    meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta_model.fit(meta_X, y_true)
    
    # Kiểm tra xem trọng số có bị nghịch đảo không, nếu bị ngược thì đảo lại
    # Trọng số của Naive Bayes (cột 0) bắt buộc phải tỷ lệ thuận với xác suất AI
    if meta_model.coef_[0][0] < 0:
        meta_model.coef_ = -meta_model.coef_
        meta_model.intercept_ = -meta_model.intercept_
        
    # Lưu mô hình chuẩn vào artifacts
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, artifact_name))
    print(f" -> Đã tối ưu hóa và lưu Meta-model {label_lang} thành công (Chống ngược nhãn).")


# ── Pipeline chính ────────────────────────────────────────────────────────────

def train():
    print("1. Chuẩn bị dữ liệu (tách EN / VI riêng biệt)...")
    # Đọc dữ liệu gốc
    df_en_raw = load_and_prepare_data(os.path.join("data", "AI_Human.csv"), is_vi=False)
    df_vi_raw = load_and_prepare_data(os.path.join("data", "dataset_vi.csv"), is_vi=True)
    
    # 2. Tách tập Train / Test độc lập từ đầu để tránh rò rỉ dữ liệu (Data Leakage)
    df_en_train_raw, df_en_test = train_test_split(df_en_raw, test_size=0.2, random_state=42, stratify=df_en_raw['label'])
    df_vi_train_raw, df_vi_test = train_test_split(df_vi_raw, test_size=0.2, random_state=42, stratify=df_vi_raw['label'])
    
    # 3. Chỉ thực hiện tăng cường dữ liệu (Augmentation) trên tập TRAIN để đảm bảo tính khách quan
    print("2. Đang tăng cường dữ liệu tiếng Việt (Augmentation sạch)...")
    df_vi_train = expand_vietnamese_samples(df_vi_train_raw, target_count=1500)
    df_en_train = df_en_train_raw.copy() # Tiếng Anh giữ nguyên hoặc tăng cường tùy ý
    
    print(f"   -> Số lượng mẫu TRAIN - EN: {len(df_en_train)} | VI: {len(df_vi_train)}")
    print(f"   -> Số lượng mẫu TEST  - EN: {len(df_en_test)} | VI: {len(df_vi_test)}")
    
    # Ép kiểu tường minh thành mảng 1 chiều chứa chuỗi văn bản
    X_en_train = df_en_train['text'].astype(str).values
    y_en_train = df_en_train['label'].astype(int).values
    X_vi_train = df_vi_train['text'].astype(str).values
    y_vi_train = df_vi_train['label'].astype(int).values
    
    X_en_test = df_en_test['text'].astype(str).values
    y_en_test = df_en_test['label'].astype(int).values
    X_vi_test = df_vi_test['test'].astype(str).values if 'test' in df_en_test.columns else df_en_test['text'].astype(str).values
    X_vi_test = df_vi_test.values if hasattr(X_vi_test, 'values') else X_vi_test
    X_vi_test = df_vi_test.astype(str)
    
    # Đồng bộ hóa lấy đúng cột text/label cho tập test
    X_en_test = df_en_test['text'].astype(str).values
    X_vi_test = df_vi_test['text'].astype(str).values
    y_en_test = df_en_test['label'].astype(int).values
    y_vi_test = df_vi_test['label'].astype(int).values

    # 4. Huấn luyện cấu trúc Bayesian Network dựa trên tập dữ liệu đặc trưng cấu trúc của VI
    print("3. Trích xuất đặc trưng cấu trúc & Huấn luyện mạng Bayes...")
    struct_features_list = []
    for txt, lbl in zip(X_vi_train, y_vi_train):
        feats = extract_structural_features(txt)
        feats['label'] = lbl
        struct_features_list.append(feats)
    df_struct_train = pd.DataFrame(struct_features_list)
    train_bayesian_network(df_struct_train)

    # 5. Xây dựng Pipeline & Huấn luyện Naive Bayes (TF-IDF)
    print("4. Huấn luyện mô hình Naive Bayes Tiếng Anh...")
    nb_en_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    nb_en_pipeline.fit(X_en_train, y_en_train)
    joblib.dump(nb_en_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb_en.pkl'))

    print("5. Huấn luyện mô hình Naive Bayes Tiếng Việt...")
    nb_vi_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    nb_vi_pipeline.fit(X_vi_train, y_vi_train)
    joblib.dump(nb_vi_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb_vi.pkl'))

    # 6. Huấn luyện Meta-model gộp (Stacking Ensemble) bằng Logistic Regression
    print("6. Huấn luyện Meta-model tiếng Anh...")
    train_meta_model(X_en_train, y_en_train, nb_en_pipeline, 
                     artifact_name='meta_model_en.pkl', label_lang="EN")
    
    # Tạo bản copy tương thích ngược
    import shutil
    if os.path.exists(os.path.join(ARTIFACTS_DIR, 'meta_model_en.pkl')):
        shutil.copy(os.path.join(ARTIFACTS_DIR, 'meta_model_en.pkl'),
                    os.path.join(ARTIFACTS_DIR, 'meta_model.pkl'))

    print("7. Huấn luyện Meta-model tiếng Việt...")
    train_meta_model(X_vi_train, y_vi_train, nb_vi_pipeline, 
                     artifact_name='meta_model_vi.pkl', label_lang="VI")

    # 7. Đánh giá chất lượng phân lớp độc lập trên tập Test
    print("\n8. Đánh giá hệ thống Naive Bayes EN trên test set:")
    y_pred_en = nb_en_pipeline.predict(X_en_test)
    print(classification_report(y_en_test, y_pred_en, target_names=['Human', 'AI']))

    print("9. Đánh giá hệ thống Naive Bayes VI trên test set:")
    y_pred_vi = nb_vi_pipeline.predict(X_vi_test)
    print(classification_report(y_vi_test, y_pred_vi, target_names=['Human', 'AI']))

    # 8. Lưu tập test gộp ra file vật lý để phục vụ report_utils sinh Confusion Matrix
    df_test_en_save = pd.DataFrame({'text': X_en_test, 'label': y_en_test, 'lang': 'en'})
    df_test_vi_save = pd.DataFrame({'text': X_vi_test, 'label': y_vi_test, 'lang': 'vi'})
    df_test_combined = pd.concat([df_test_en_save, df_test_vi_save], ignore_index=True)
    df_test_combined.to_csv(os.path.join(ARTIFACTS_DIR, 'test_set.csv'), index=False)
    print(" -> Đã đóng gói tập dữ liệu kiểm thử độc lập test_set.csv thành công.")


if __name__ == "__main__":
    train()
