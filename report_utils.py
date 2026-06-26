# report_utils.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from predictor import predict_text

ARTIFACTS_DIR = "artifacts"


def evaluate_naive_bayes_only():
    """Đánh giá riêng mô hình Naive Bayes (không có BN/Heuristic) trên tập test."""
    test_path = os.path.join(ARTIFACTS_DIR, 'test_set.csv')
    if not os.path.exists(test_path):
        print("Chưa có test_set.csv. Hãy chạy train.py trước.")
        return None

    df_test = pd.read_csv(test_path)
    nb_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    y_true = df_test['label'].values
    y_pred = nb_model.predict(df_test['text'].values)

    print("=== Naive Bayes (riêng lẻ) ===")
    print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))
    return confusion_matrix(y_true, y_pred)


def evaluate_combined_system():
    """
    Đánh giá HỆ THỐNG KẾT HỢP đầy đủ (NB + Bayesian Network + Heuristic,
    kết hợp bằng meta-model đã học trọng số) qua predict_text() —
    đây là kết quả người dùng thực sự thấy trong app.
    """
    test_path = os.path.join(ARTIFACTS_DIR, 'test_set.csv')
    if not os.path.exists(test_path):
        print("Chưa có test_set.csv. Hãy chạy train.py trước.")
        return None

    df_test = pd.read_csv(test_path)
    y_true = df_test['label'].values
    y_pred = []
    for text in df_test['text']:
        result = predict_text(text)
        y_pred.append(1 if result['label'] == "AI Generated" else 0)
    y_pred = np.array(y_pred)

    print("=== Hệ thống kết hợp (NB + BN + Heuristic, qua meta-model) ===")
    print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'AI'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix — Hệ thống kết hợp")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("Đã lưu confusion_matrix.png vào artifacts/")
    return cm


def evaluate_on_test_set():
    """Hàm gọi từ app.py: đánh giá hệ thống kết hợp (số liệu hiển thị cho người dùng)."""
    return evaluate_combined_system()


def get_text_stats(text: str) -> dict:
    """Thống kê cơ bản của văn bản để hiển thị trên giao diện."""
    words = text.split()
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    punct_count = sum(1 for c in text if c in '.,;:!?')
    return {
        'Số ký tự': len(text),
        'Số từ': len(words),
        'Số câu': len(sentences),
        'Độ dài từ TB': round(np.mean([len(w) for w in words]), 2) if words else 0,
        'Tỉ lệ dấu câu (%)': round(punct_count / max(len(text), 1) * 100, 2),
    }


if __name__ == "__main__":
    evaluate_naive_bayes_only()
    print()
    evaluate_combined_system()