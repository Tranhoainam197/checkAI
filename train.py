import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score

from data_utils import load_and_prepare_data, extract_structural_features
from bayesian_network import train_bayesian_network

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

    # Giới hạn 8000 mẫu để tuning nhanh hơn
    limit = min(8000, len(X_train))
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]

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
            best_score = mean_f1
            best_params = params
            best_pipeline = pipeline

    pd.DataFrame(results).sort_values('cv_f1_mean', ascending=False).to_csv(
        os.path.join(ARTIFACTS_DIR, 'tuning_results.csv'), index=False
    )
    print(f"Best params: {best_params}  |  CV F1: {best_score:.4f}")
    return best_pipeline, best_params


def train(path_en: str = "data/AI_Human.csv", path_vi: str = "data/dataset_vi.csv"):
    print("1. Chuẩn bị dữ liệu...")
    df = load_and_prepare_data(path_en, path_vi)

    # Chia train/test 80/20 theo stratify
    from sklearn.model_selection import train_test_split
    X_all = df['text'].values
    y_all = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    print("2. Tuning Naive Bayes...")
    nb_pipeline, _ = tune_naive_bayes(X_train, y_train)

    print("3. Huấn luyện Naive Bayes trên toàn bộ train...")
    nb_pipeline.fit(X_train, y_train)
    joblib.dump(nb_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    print("4. Huấn luyện Bayesian Network...")
    struct_train = [extract_structural_features(t) for t in X_train]
    df_struct_train = pd.DataFrame(struct_train)
    df_struct_train['label'] = y_train
    train_bayesian_network(df_struct_train)

    print("5. Đánh giá trên tập test...")
    from sklearn.metrics import classification_report
    y_pred = nb_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

    # Lưu test set để report_utils dùng
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(
        os.path.join(ARTIFACTS_DIR, 'test_set.csv'), index=False
    )
    print("Hoàn thành! Artifacts đã lưu vào thư mục 'artifacts/'")


if __name__ == "__main__":
    train()