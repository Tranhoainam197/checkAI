import re
import joblib
import numpy as np
from data_utils import clean_text, repair_mojibake, extract_structural_features
from bayesian_network import predict_bayesian_network

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

# Ngưỡng quyết định — giữ nguyên từ phiên bản gốc
THRESHOLD_VI = 0.42
THRESHOLD_EN = 0.50

_VI_PATTERN = re.compile(
    r'[àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]',
    re.IGNORECASE,
)

# Cache các model — load lần đầu, tái dùng từ lần sau
_nb_en_model  = None
_nb_vi_model  = None
_meta_en_model = None
_meta_vi_model = None


def _load(path):
    """Load model từ disk, trả về None nếu file chưa tồn tại."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def _get_nb_en():
    global _nb_en_model
    if _nb_en_model is None:
        # Ưu tiên model EN riêng; fallback về model_nb.pkl (phiên bản cũ)
        _nb_en_model = _load("artifacts/model_nb_en.pkl") or _load("artifacts/model_nb.pkl")
    return _nb_en_model


def _get_nb_vi():
    global _nb_vi_model
    if _nb_vi_model is None:
        # Ưu tiên model VI riêng; fallback về EN nếu chưa có
        _nb_vi_model = _load("artifacts/model_nb_vi.pkl") or _get_nb_en()
    return _nb_vi_model


def _get_meta_en():
    global _meta_en_model
    if _meta_en_model is None:
        _meta_en_model = (_load("artifacts/meta_model_en.pkl")
                          or _load("artifacts/meta_model.pkl"))
    return _meta_en_model


def _get_meta_vi():
    global _meta_vi_model
    if _meta_vi_model is None:
        # Ưu tiên meta-model VI học từ dữ liệu;
        # fallback về EN nếu chưa train (ví dụ dùng model cũ)
        _meta_vi_model = (_load("artifacts/meta_model_vi.pkl")
                          or _load("artifacts/meta_model.pkl"))
    return _meta_vi_model


def is_vietnamese(text: str) -> bool:
    matches = len(_VI_PATTERN.findall(text))
    return matches / max(len(text), 1) > 0.01


def chunk_text(text: str) -> list[str]:
    """Chia văn bản dài thành các đoạn nhỏ có overlap."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start, n = [], 0, len(text)
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
    Tính điểm heuristic từ đặc trưng bề mặt. Trả về [0, 1], càng cao càng giống AI.
    Các tín hiệu:
    - Độ đều độ dài câu (CV thấp → AI viết đều đặn)
    - Tỉ lệ dấu câu thấp (AI ít dùng dấu câu biến thể)
    - Tỉ lệ từ dài cao (AI dùng từ học thuật nhiều hơn)
    - Thiếu câu ngắn (human thường xen câu ngắn)
    """
    score = 0.0
    words = text.split()
    if not words:
        return 0.5

    sentences = [s.strip() for s in
                 text.replace('!', '.').replace('?', '.').split('.')
                 if s.strip()]

    if len(sentences) > 2:
        sent_lens = [len(s) for s in sentences]
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        if cv < 0.3:
            score += 0.35

    punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
    if punct_ratio < 0.02:
        score += 0.25

    long_word_ratio = sum(1 for w in words if len(w) > 8) / max(len(words), 1)
    if long_word_ratio > 0.15:
        score += 0.2

    if sentences:
        short_sent_ratio = sum(1 for s in sentences if len(s) < 40) / len(sentences)
        if short_sent_ratio < 0.2:
            score += 0.2

    return min(score, 1.0)


def _compute_bn_score_chunked(chunks: list[str]) -> float:
    scores = []
    for chunk in chunks:
        try:
            struct = extract_structural_features(chunk)
            scores.append(predict_bayesian_network(struct))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores)) if scores else 0.5


def predict_text(text: str, bn_available: bool = True) -> dict:
    """
    Hàm dự đoán chính. Kết hợp NB + BN + Heuristic qua meta-model.

    Cải tiến so với phiên bản cũ:
    - Tiếng Anh: dùng nb_en + meta_en (học từ dữ liệu EN)
    - Tiếng Việt: dùng nb_vi + meta_vi (học từ dữ liệu VI, không gán tay)
    - Không còn hàm _combine_scores_vi() với trọng số gán tay
    """
    text = repair_mojibake(text)
    text = clean_text(text)

    vi = is_vietnamese(text)
    threshold = THRESHOLD_VI if vi else THRESHOLD_EN

    # Chọn model phù hợp ngôn ngữ
    nb_model  = _get_nb_vi()  if vi else _get_nb_en()
    meta_model = _get_meta_vi() if vi else _get_meta_en()

    chunks = chunk_text(text)

    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    bn_score = _compute_bn_score_chunked(chunks) if bn_available else 0.5

    h_score = heuristic_score(text)

    if meta_model is not None:
        meta_input = np.array([[nb_score, bn_score, h_score]])
        combined = float(meta_model.predict_proba(meta_input)[0, 1])
    else:
        # Fallback: chưa có meta-model → dùng NB score
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
    """Phân tích từng dòng văn bản riêng lẻ (không dùng BN vì 1 dòng quá ngắn)."""
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
    return [{'line': line, **predict_text(line, bn_available=False)} for line in lines]
