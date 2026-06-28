import re
import joblib
import numpy as np
from data_utils import clean_text, repair_mojibake, extract_structural_features
from bayesian_network import predict_bayesian_network

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

# Ngưỡng quyết định
THRESHOLD_VI = 0.42   # tiếng Việt: hạ thấp hơn vì NB yếu với VI
THRESHOLD_EN = 0.50   # tiếng Anh: giữ nguyên

# Regex phát hiện tiếng Việt
_VI_PATTERN = re.compile(
    r'[àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]',
    re.IGNORECASE
)

_nb_model = None
_meta_model = None


def _get_nb_model():
    global _nb_model
    if _nb_model is None:
        _nb_model = joblib.load("artifacts/model_nb.pkl")
    return _nb_model


def _get_meta_model():
    """Meta-model (Logistic Regression) học trọng số kết hợp NB+BN+Heuristic
    từ dữ liệu thật. Nếu chưa có file thì trả về None — fallback dùng NB."""
    global _meta_model
    if _meta_model is None:
        try:
            _meta_model = joblib.load("artifacts/meta_model.pkl")
        except FileNotFoundError:
            _meta_model = False
    return _meta_model or None


def is_vietnamese(text: str) -> bool:
    """Kiểm tra văn bản có phải tiếng Việt không dựa trên tỉ lệ ký tự đặc trưng."""
    matches = len(_VI_PATTERN.findall(text))
    return matches / max(len(text), 1) > 0.01


def chunk_text(text: str) -> list[str]:
    """
    Chia văn bản dài thành các đoạn nhỏ có overlap, cắt theo ranh giới từ
    gần nhất để không làm hỏng n-gram ký tự ở điểm cắt.
    Đảm bảo start luôn tăng để tránh vòng lặp vô tận.
    """
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
        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end
    return [c for c in chunks if c] or [text]


def heuristic_score(text: str) -> float:
    """
    Tính điểm heuristic dựa trên đặc trưng bề mặt của văn bản.
    Trả về float trong [0, 1], càng cao càng giống AI.

    Các tín hiệu:
    - Độ đều độ dài câu (CV thấp → AI viết đều đặn hơn human)
    - Tỉ lệ dấu câu thấp (AI ít dùng dấu câu biến thể)
    - Tỉ lệ từ dài cao (AI dùng từ học thuật nhiều hơn)
    - Thiếu câu ngắn (human thường xen câu ngắn)
    """
    score = 0.0
    words = text.split()
    if not words:
        return 0.5

    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    # Tín hiệu 1: Độ đều câu (Coefficient of Variation thấp → AI)
    if len(sentences) > 2:
        sent_lens = [len(s) for s in sentences]
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        if cv < 0.3:
            score += 0.35

    # Tín hiệu 2: Tỉ lệ dấu câu thấp (AI ít dùng dấu câu đa dạng)
    punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
    if punct_ratio < 0.02:
        score += 0.25

    # Tín hiệu 3: Tỉ lệ từ dài (AI thường dùng từ học thuật dài hơn)
    long_word_ratio = sum(1 for w in words if len(w) > 8) / max(len(words), 1)
    if long_word_ratio > 0.15:
        score += 0.2

    # Tín hiệu 4: Thiếu câu ngắn (human thường có câu ngắn xen kẽ)
    if sentences:
        short_sent_ratio = sum(1 for s in sentences if len(s) < 40) / len(sentences)
        if short_sent_ratio < 0.2:
            score += 0.2

    return min(score, 1.0)


def _compute_bn_score_chunked(chunks: list[str]) -> float:
    """
    Tính BN score bằng cách trích xuất đặc trưng cấu trúc từng chunk
    rồi lấy trung bình — hoạt động tốt hơn với văn bản dài.
    """
    scores = []
    for chunk in chunks:
        try:
            struct = extract_structural_features(chunk)
            scores.append(predict_bayesian_network(struct))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores)) if scores else 0.5


def _combine_scores_vi(nb_score: float, bn_score: float, h_score: float,
                        meta_model) -> float:
    """
    Công thức kết hợp riêng cho tiếng Việt.

    Vấn đề: meta-model học trọng số chủ yếu từ dữ liệu tiếng Anh (NB mạnh hơn
    với EN), nên khi NB yếu với tiếng Việt, meta-model "nhấn chìm" tín hiệu
    heuristic và BN vốn vẫn khá chính xác.

    Giải pháp: dùng trung bình có trọng số cố định cho tiếng Việt, ưu tiên
    heuristic cao hơn (vì heuristic không phụ thuộc ngôn ngữ) thay vì tin
    hoàn toàn vào meta-model.
    """
    # Trọng số: NB=0.40, BN=0.25, Heuristic=0.35
    # Heuristic được nâng lên 0.35 (từ ~0.08 của meta-model) vì nó
    # không bị ảnh hưởng bởi sự khác biệt ngôn ngữ trong dữ liệu train.
    return 0.35 * nb_score + 0.20 * bn_score + 0.45 * h_score


def predict_text(text: str, bn_available: bool = True) -> dict:
    """
    Hàm dự đoán chính. Kết hợp NB + BN + Heuristic.

    - Tiếng Anh: dùng meta-model (Logistic Regression) học từ dữ liệu → tối ưu
    - Tiếng Việt: dùng trung bình có trọng số cố định, ưu tiên heuristic cao hơn
                  vì meta-model học chủ yếu từ pattern tiếng Anh
    - Ngưỡng quyết định: 0.40 (VI) / 0.50 (EN)
    """
    nb_model = _get_nb_model()
    meta_model = _get_meta_model()

    text = repair_mojibake(text)
    text = clean_text(text)

    vi = is_vietnamese(text)
    threshold = THRESHOLD_VI if vi else THRESHOLD_EN

    chunks = chunk_text(text)

    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    bn_score = 0.5
    if bn_available:
        bn_score = _compute_bn_score_chunked(chunks)

    h_score = heuristic_score(text)

    if vi:
        # Tiếng Việt: trọng số cố định, ưu tiên heuristic
        combined = _combine_scores_vi(nb_score, bn_score, h_score, meta_model)
    elif meta_model is not None:
        # Tiếng Anh: dùng meta-model đã học từ dữ liệu
        meta_input = np.array([[nb_score, bn_score, h_score]])
        combined = float(meta_model.predict_proba(meta_input)[0, 1])
    else:
        # Fallback: chưa có meta-model
        combined = nb_score

    label = "AI Generated" if combined >= threshold else "Human Written"
    confidence = combined if combined >= threshold else 1 - combined

    return {
        'label': label,
        'confidence': round(confidence * 100, 2),
        'nb_score': round(nb_score * 100, 2),
        'bn_score': round(bn_score * 100, 2),
        'heuristic_score': round(h_score * 100, 2),
        'combined_score': round(combined * 100, 2),
        'language': 'vi' if vi else 'en',
        'threshold_used': threshold,
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
