import joblib
import numpy as np
from deep_translator import GoogleTranslator
from data_utils import clean_text, repair_mojibake, extract_structural_features
from bayesian_network import predict_bayesian_network

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

# Bộ dịch tự động phát hiện ngôn ngữ sang tiếng Anh
translator_to_en = GoogleTranslator(source='auto', target='en')
_nb_model = None

def _get_nb_model():
    global _nb_model
    if _nb_model is None:
        _nb_model = joblib.load("artifacts/model_nb.pkl")
    return _nb_model

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks if chunks else [text]

def is_vietnamese(text: str) -> bool:
    """Nhận biết nhanh tiếng Việt dựa trên dấu."""
    vietnamese_chars = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    return any(char in vietnamese_chars for char in text.lower())

def heuristic_score(text: str) -> float:
    """Tính điểm heuristic (Dùng Tiếng Anh)."""
    score = 0.0
    words = text.split()
    if not words: return 0.5

    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if len(sentences) > 2:
        sent_lens = [len(s) for s in sentences]
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        if cv < 0.3: score += 0.3

    if (sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)) < 0.02: score += 0.2
    if (sum(1 for c in text if c.isupper()) / max(len(text), 1)) < 0.01: score += 0.1
    return min(score, 1.0)

def predict_text(raw_text: str, bn_available: bool = True) -> dict:
    nb_model = _get_nb_model()
    text = repair_mojibake(raw_text)

    # 1. CỔNG DỊCH THUẬT: Phát hiện và dịch sang Tiếng Anh
    if is_vietnamese(text):
        try:
            processing_text = translator_to_en.translate(text)
        except Exception:
            processing_text = text # Fallback nếu rớt mạng
    else:
        processing_text = text

    # 2. Tính điểm Mạng Bayes & Heuristic trên văn bản Tiếng Anh gốc
    h_score = heuristic_score(processing_text)
    
    bn_score = 0.5
    if bn_available:
        try:
            struct = extract_structural_features(processing_text)
            bn_score = predict_bayesian_network(struct)
        except Exception:
            bn_available = False

    # 3. Tính điểm Naive Bayes trên văn bản Tiếng Anh đã làm sạch sâu
    cleaned_text = clean_text(processing_text)
    chunks = chunk_text(cleaned_text) if len(cleaned_text) > CHUNK_SIZE else [cleaned_text]
    
    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    # 4. Tính toán kết quả cuối cùng
    combined = (nb_score * 0.80 + h_score) * 0.85 + bn_score * 0.15 if bn_available else nb_score * 0.85 + h_score * 0.15
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
    """
    Phân tích từng dòng bằng cơ chế Batch Translation (Gom cụm dịch 1 lần).
    Tăng tốc độ xử lý lên gấp nhiều lần, giảm thiểu nghẽn băng thông Internet.
    """
    # 1. Lọc và lấy ra các dòng hợp lệ (> 20 ký tự)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
    if not lines:
        return []

    # 2. Kiểm tra nếu là Tiếng Việt thì gom cụm dịch 1 lần duy nhất
    translated_lines = []
    if any(is_vietnamese(line) for line in lines):
        try:
            # Nối các dòng bằng ký tự đặc biệt để dịch hàng loạt
            delimiter = " ||| "
            combined_text = delimiter.join(lines)
            
            # Gửi yêu cầu dịch 1 lần duy nhất qua Internet
            translated_combined = translator_to_en.translate(combined_text)
            
            # Tách ngược lại thành danh sách các dòng tiếng Anh
            translated_lines = [ch.strip() for ch in translated_combined.split("|||")]
            
            # Phòng hờ trường hợp bộ dịch gộp nhầm hoặc làm mất dấu, đảm bảo độ dài mảng khớp nhau
            if len(translated_lines) != len(lines):
                translated_lines = lines # Fallback nếu lỗi cấu trúc tách chuỗi
        except Exception:
            translated_lines = lines # Fallback nếu mất mạng hoặc lỗi API
    else:
        translated_lines = lines

    # 3. Tiến hành suy luận song song (Chỉ tính toán toán học, không gọi Internet nữa)
    results = []
    for raw_line, eng_line in zip(lines, translated_lines):
        # Gọi hàm predict_text nhưng truyền văn bản đã dịch sẵn vào để bỏ qua bước dịch trùng lặp
        res = _predict_with_pre_translated_text(raw_line, eng_line)
        results.append({'line': raw_line, **res})
        
    return results


def _predict_with_pre_translated_text(raw_text: str, eng_text: str) -> dict:
    """Hàm phụ trợ: Dự đoán văn bản khi đã có sẵn bản dịch tiếng Anh (Tối ưu RAM)"""
    nb_model = _get_nb_model()
    
    cleaned_text = clean_text(eng_text)
    chunks = chunk_text(cleaned_text) if len(cleaned_text) > CHUNK_SIZE else [cleaned_text]
    raw_chunks = chunk_text(eng_text)

    h_scores = [heuristic_score(ch) for ch in raw_chunks]
    h_score = float(np.mean(h_scores))
    
    # Ở chế độ quét dòng ngắn, tắt bớt Mạng Bayes (bn_available=False) để tối ưu độ nhạy
    nb_probs = nb_model.predict_proba(chunks)[:, 1]
    nb_score = float(np.mean(nb_probs))

    # Tính điểm kết hợp cho chế độ quét dòng
    combined = nb_score * 0.80 + h_score * 0.20

    label = "AI Generated" if combined >= 0.45 else "Human Written"
    confidence = combined if combined >= 0.5 else 1 - combined

    return {
        'label': label,
        'confidence': round(confidence * 100, 2),
        'nb_score': round(nb_score * 100, 2),
        'bn_score': 0.0,
        'heuristic_score': round(h_score * 100, 2),
        'combined_score': round(combined * 100, 2),
    }