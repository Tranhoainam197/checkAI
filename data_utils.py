import numpy as np
import re

def clean_text(text: str) -> str:
    """Làm sạch sâu văn bản để đưa vào Naive Bayes."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def repair_mojibake(text: str) -> str:
    """Sửa lỗi encoding bị vỡ (nếu có)."""
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def extract_structural_features(text: str) -> dict:
    """Trích xuất đặc trưng cấu trúc cho Bayesian Network."""
    text_str = str(text)
    words = text_str.split()
    word_count = len(words) if len(words) > 0 else 1
    
    return {
        'length': len(text_str),
        'word_count': word_count,
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0
    }
