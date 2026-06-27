import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from data_utils import clean_text

ARTIFACTS_DIR = "artifacts"

def evaluate_on_test_set():
    test_path = os.path.join(ARTIFACTS_DIR, 'test_set.csv')
    if not os.path.exists(test_path):
        return None

    df_test = pd.read_csv(test_path)
    nb_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_nb.pkl'))

    y_true = df_test['label'].values
    cleaned_test_texts = [clean_text(str(t)) for t in df_test['text'].values]
    y_pred = nb_model.predict(cleaned_test_texts)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'AI'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    return cm

def get_text_stats(text: str) -> dict:
    words = text.split()
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    punct_count = sum(1 for c in text if c in '.,;:!?')
    return {
        'Số ký tự': len(text),
        'Số từ': len(words),
        'Số câu': len(sentences),
        'Độ dài từ TB (ký tự)': round(np.mean([len(w) for w in words]), 2) if words else 0,
        'Tỉ lệ dấu câu (%)': round(punct_count / max(len(text), 1) * 100, 2),
    }

if __name__ == "__main__":
    evaluate_on_test_set()