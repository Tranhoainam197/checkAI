import joblib
import numpy as np
from data_utils import clean_text, repair_mojibake, extract_structural_features
from bayesian_network import predict_bayesian_network

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

_nb_model = None
_meta_model = None


def _get_nb_model():
    global _nb_model
    if _nb_model is None:
        _nb_model = joblib.load("artifacts/model_nb.pkl")
    return _nb_model


def _get_meta_model():
    """Meta-model (Logistic Regression) học trọng số kết hợp NB+BN+Heuristic
    từ dữ liệu thật, thay cho công thức cộng tay. Nếu chưa huấn luyện (file
    chưa tồn tại), trả về None — predict_text sẽ tự fallback dùng NB làm chính."""
    global _meta_model
    if _meta_model is None:
        try:
            _meta_model = joblib.load("artifacts/meta_model.pkl")
        except FileNotFoundError:
            _meta_model = False  # đánh dấu đã thử và không có, tránh load lại mỗi lần
    return _meta_model or None


def chunk_text(text: str) -> list[str]:
    """Chia văn bản dài thành các đoạn nhỏ có overlap, cắt theo ranh giới từ
    gần nhất để không làm hỏng n-gram ký tự ở điểm cắt."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            boundary = text.rfind(' ', start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
    return [c for c in chunks if c] or [text]


def heuristic_score(text: str) -> float:
    """
    Tính điểm heuristic dựa trên đặc trưng bề mặt của văn bản.
    Trả về float trong [0, 1], càng cao càng giống AI.
    """
    score = 0.0
    words = text.split()
    if not words:
        return 0.5

    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if len(sentences) > 2:
        sent_lens = [len(s) for s in sentences]
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        if cv < 0.3:
            score += 0.3

    punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
    if punct_ratio < 0.02:
        score += 0.2

    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio < 0.01:
        score += 0.1

    return min(score, 1.0)


def predict_text(text: str, bn_available: bool = True) -> dict:
    """
    Hàm dự đoán chính. Kết hợp NB + BN + Heuristic bằng meta-model
    (Logistic Regression) đã học trọng số tối ưu từ dữ liệu train.
    Nếu chưa có meta-model, fallback: dùng NB làm kết quả chính.
    """
    nb_model = _get_nb_model()
    meta_model = _get_meta_model()

    text = repair_mojibake(text)
    text = clean_text(text)

    chunks = chunk_text(text)

    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    bn_score = 0.5
    if bn_available:
        try:
            struct = extract_structural_features(text)
            bn_score = predict_bayesian_network(struct)
        except Exception:
            bn_available = False

    h_score = heuristic_score(text)

    if meta_model is not None:
        meta_input = np.array([[nb_score, bn_score if bn_available else 0.5, h_score]])
        combined = float(meta_model.predict_proba(meta_input)[0, 1])
    else:
        # Fallback an toàn: chưa có meta-model -> tin tưởng NB là chính
        combined = nb_score

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
    """Phân tích từng dòng văn bản riêng lẻ (không dùng BN vì 1 dòng quá ngắn
    để ước lượng đặc trưng cấu trúc tin cậy)."""
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
    results = []
    for line in lines:
        res = predict_text(line, bn_available=False)
        results.append({'line': line, **res})
    return results