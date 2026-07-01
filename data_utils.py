import pandas as pd
import numpy as np
import re
from collections import Counter
from pyvi import ViTokenizer

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def repair_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    if any(ord(c) > 127 for c in text):
        return text
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

# Giữ nguyên từ điển đồng nghĩa để hoán đổi từ khi tăng cường dữ liệu
_SYNONYMS_VI = {
    "tôi": ["mình", "em", "chúng tôi", "bản thân"],
    "bạn": ["các bạn", "anh", "chị", "mọi người"],
    "mô hình": ["hệ thống", "giải pháp", "phương pháp", "công cụ"],
    "trí tuệ nhân tạo": ["AI", "công nghệ thông minh", "trí thông minh nhân tạo"],
    "học tập": ["nghiên cứu", "rèn luyện", "trau dồi"]
}

def expand_vietnamese_samples(df: pd.DataFrame, target_count: int = 1500) -> pd.DataFrame:
    """Tăng cường dữ liệu nhưng KHÔNG thêm tiền tố/hậu tố làm nhiễu nhãn văn phong AI/Human"""
    current_count = len(df)
    if current_count >= target_count:
        return df

    needed = target_count - current_count
    augmented_records = []
    existing_records = df.to_dict('records')

    for _ in range(needed):
        base = np.random.choice(existing_records)
        text = str(base['text'])
        label = base['label']
        
        # Chỉ thực hiện thay thế từ đồng nghĩa để giữ nguyên cấu trúc văn phong nhãn gốc
        words = text.split()
        for i, w in enumerate(words):
            w_clean = w.lower().strip(",.?!()\"")
            if w_clean in _SYNONYMS_VI and np.random.rand() < 0.3:
                syn = np.random.choice(_SYNONYMS_VI[w_clean])
                words[i] = words[i].replace(w_clean, syn)
                
        augmented_records.append({'text': ' '.join(words), 'label': label})

    return pd.concat([df, pd.DataFrame(augmented_records)], ignore_index=True)

def load_and_prepare_data(file_path: str, is_vi: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Chuẩn hóa tên cột
    df.columns = [c.lower().strip() for c in df.columns]
    
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    
    if is_vi:
        df[text_col] = df[text_col].apply(repair_mojibake)
        # Tiến hành tách từ tiếng Việt ngay từ bước nạp dữ liệu để các thuật toán xử lý đồng bộ
        df[text_col] = df[text_col].apply(lambda x: ViTokenizer.tokenize(x))
        
    df['label'] = df[label_col].astype(int)
    return df[[text_col, 'label']].rename(columns={text_col: 'text'})

def extract_structural_features(text: str) -> dict:
    """Trích xuất đặc trưng cấu trúc cho Bayesian Network (đã chuẩn hóa phân tách từ)"""
    text = str(text)
    # Vì text đã được tách từ qua ViTokenizer (ví dụ: "trí_tuệ nhân_tạo"), split bằng khoảng trắng vẫn chuẩn
    words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences) if sentences else 1
    avg_word_len = np.mean([len(w.replace('_', '')) for w in words]) if words else 0

    if words:
        unique_words = set(words)
        ttr = len(unique_words) / word_count
        freq = Counter(words)
        repeated = sum(1 for cnt in freq.values() if cnt >= 3)
        repetition_rate = repeated / word_count
    else:
        ttr = 0.0
        repetition_rate = 0.0

    return {
        'length': len(text),
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'sentence_count': sentence_count,
        'type_token_ratio': ttr,
        'repetition_rate': repetition_rate
    }