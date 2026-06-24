import joblib
import numpy as np
from data_utils import clean_text, repair_mojibake, extract_structural_features
from bayesian_network import predict_bayesian_network

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

# Load model một lần duy nhất khi import module — tránh load lại mỗi lần predict
_nb_model = None

def _get_nb_model():
    global _nb_model
    if _nb_model is None:
        _nb_model = joblib.load("artifacts/model_nb.pkl")
    return _nb_model


def chunk_text(text: str) -> list[str]:
    """Chia văn bản dài thành các đoạn nhỏ có overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks if chunks else [text]


def heuristic_score(text: str) -> float:
    """
    Tính điểm heuristic dựa trên đặc trưng bề mặt của văn bản.
    Trả về float trong [0, 1], càng cao càng giống AI.
    """
    score = 0.0
    words = text.split()
    if not words:
        return 0.5

    # Câu rất đều nhau về độ dài → AI
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if len(sentences) > 2:
        sent_lens = [len(s) for s in sentences]
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        if cv < 0.3:
            score += 0.3

    # Tỉ lệ dấu câu thấp → AI
    punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
    if punct_ratio < 0.02:
        score += 0.2

    # Tỉ lệ chữ hoa thấp → AI (ít cảm xúc)
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio < 0.01:
        score += 0.1

    return min(score, 1.0)


def predict_text(text: str, bn_available: bool = True) -> dict:
    """
    Hàm dự đoán chính. Nhận văn bản thô, trả về dict kết quả.
    """
    nb_model = _get_nb_model()

    text = repair_mojibake(text)
    text = clean_text(text)

    if len(text) > CHUNK_SIZE:
        chunks = chunk_text(text)
    else:
        chunks = [text]

    # Naive Bayes score (trung bình các chunk)
    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    # Bayesian Network score
    bn_score = 0.5
    if bn_available:
        try:
            struct = extract_structural_features(text)
            bn_score = predict_bayesian_network(struct)
        except Exception:
            bn_available = False

    # Heuristic score
    h_score = heuristic_score(text)

    # Kết hợp theo trọng số
    if bn_available:
        combined = (nb_score * 0.80 + h_score) * 0.85 + bn_score * 0.15
    else:
        combined = nb_score * 0.85 + h_score * 0.15

    label = "AI Generated" if combined >= 0.5 else "Human Written"
    confidence = combined if combined >= 0.5 else 1 - combined

    return {
        'label': label,
        'confidence': round(confidence * 100, 2),
        'nb_score': round(nb_score * 100, 2),
        'bn_score': round(bn_score * 100, 2),
        'heuristic_score': round(h_score * 100, 2),
        'combined_score': round(combined * 100, 2),
    }


def predict_line_by_line(text: str) -> list[dict]:
    """Phân tích từng dòng văn bản riêng lẻ."""
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
    results = []
    for line in lines:
        res = predict_text(line, bn_available=False)
        results.append({'line': line, **res})
    return results