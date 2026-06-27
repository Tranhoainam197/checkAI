import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import f1_score

from data_utils import clean_text, extract_structural_features
from bayesian_network import train_bayesian_network

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def cross_validate_pipeline(pipeline, X, y, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average='macro'))
    return np.mean(scores)


def tune_naive_bayes(X_train, y_train):
    param_grid = {
        'tfidf__ngram_range': [(3, 5), (2, 5)],
        'tfidf__max_features': [10000, 15000],
        'clf__alpha': [0.3, 0.5, 0.7],
    }
    best_score, best_params = -1, None
    limit = min(8000, len(X_train))
    X_sub, y_sub = X_train[:limit], y_train[:limit]

    print(f" --> Tối ưu siêu tham số trên {limit} mẫu...")
    for params in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', **{k.split('__')[1]: v for k, v in params.items() if 'tfidf' in k})),
            ('clf', MultinomialNB(alpha=params['clf__alpha'])),
        ])
        mean_f1 = cross_validate_pipeline(pipeline, X_sub, y_sub)
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    return best_params


def train(path_en: str = "data/AI_vs_Human_Text_Dataset_v2.csv"):
    if not os.path.exists(path_en):
        print(f"LỖI: Không tìm thấy file {path_en}")
        return
        
    df = pd.read_csv(path_en)
    df = df[['text_content', 'label_binary']].rename(columns={'text_content': 'text', 'label_binary': 'label'}).dropna()
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # SỬA ĐỔI 1: Ép kiểu rõ ràng về mảng Numpy để tránh lỗi bẻ chỉ mục của Arrow Extension Array
    X_all = df['cleaned_text'].astype(str).to_numpy()
    y_all = df['label'].values.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, 
        y_all, 
        test_size=0.2, 
        stratify=y_all, 
        random_state=42
    )

    # Lưu test set cho report_utils
    test_indices = df.index[df['cleaned_text'].isin(X_test)]
    df.loc[test_indices, ['text', 'label']].to_csv(os.path.join(ARTIFACTS_DIR, 'test_set.csv'), index=False)

    best_params = tune_naive_bayes(X_train, y_train)

    print(" --> Huấn luyện Naive Bayes...")
    
    final_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=best_params['tfidf__ngram_range'], max_features=best_params['tfidf__max_features'])),
        ('clf', MultinomialNB(alpha=best_params['clf__alpha'])),
    ])
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    print(" --> Huấn luyện Bayesian Network...")
    # SỬA ĐỔI 2: Dùng .to_numpy() thay cho .values để đồng bộ kiểu dữ liệu gốc cho tập raw train
    X_train_raw = df.loc[df.index.isin(df.loc[~df.index.isin(test_indices)].index), 'text'].to_numpy()[:len(X_train)]
    
    df_struct_train = pd.DataFrame([extract_structural_features(t) for t in X_train_raw])
    df_struct_train['label'] = y_train
    train_bayesian_network(df_struct_train)
    
    print(" 🎉 Huấn luyện hoàn tất!")


if __name__ == "__main__":
    train()