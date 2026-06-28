import pandas as pd
import numpy as np
import re


def clean_text(text: str) -> str:
    """Làm sạch văn bản: chuẩn hóa khoảng trắng, xuống dòng."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def repair_mojibake(text: str) -> str:
    """
    Sửa lỗi encoding tiếng Việt bị vỡ (mojibake), nếu có.
    Chỉ áp dụng khi văn bản gốc KHÔNG chứa ký tự Unicode hợp lệ sẵn
    (tránh làm hỏng văn bản UTF-8 đã đúng).
    """
    if not isinstance(text, str) or not text:
        return text
    # Nếu đã có ký tự tiếng Việt hoặc Unicode multibyte hợp lệ → giữ nguyên
    if any(ord(c) > 127 for c in text):
        return text
    try:
        repaired = text.encode('latin-1').decode('utf-8')
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def expand_vietnamese_samples(df_vi: pd.DataFrame, multiplier: int = 2,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Tăng cường dữ liệu tiếng Việt bằng từ đồng nghĩa + prefix/suffix ngữ cảnh.

    CHỈ được gọi trên tập TRAIN (sau khi đã train_test_split), không bao giờ
    gọi trên toàn bộ dataset trước khi chia, để tránh rò rỉ dữ liệu (data leakage)
    giữa các bản gần-trùng-nhau của cùng một câu gốc.

    multiplier: số bản augment sinh thêm cho mỗi mẫu gốc.
    """
    rng = np.random.RandomState(random_state)
    synonyms = {
        "tôi": ["mình", "em", "chúng tôi"],
        "mô hình": ["hệ thống", "giải pháp", "phương pháp"],
        "nghiên cứu": ["tìm hiểu", "khảo sát", "phân tích"],
        "kết quả": ["kết luận", "đầu ra", "output"],
        "dữ liệu": ["thông tin", "tập mẫu", "dataset"],
    }
    prefixes = ["Theo nghiên cứu, ", "Có thể thấy rằng ", "Nhìn chung, ", ""]
    suffixes = [" Đây là một điểm quan trọng.", " Điều này rất đáng chú ý.", ""]

    augmented_rows = [{'text': t, 'label': l} for t, l in zip(df_vi['text'], df_vi['label'])]

    for text, label in zip(df_vi['text'], df_vi['label']):
        for _ in range(multiplier):
            new_text = str(text)
            for word, replacements in synonyms.items():
                if word in new_text:
                    new_text = new_text.replace(word, rng.choice(replacements), 1)
            new_text = rng.choice(prefixes) + new_text + rng.choice(suffixes)
            augmented_rows.append({'text': new_text, 'label': label})

    df_aug = pd.DataFrame(augmented_rows).drop_duplicates(subset='text').reset_index(drop=True)
    return df_aug


def load_and_prepare_data(path_en: str, path_vi: str,
                           en_per_class: int = 4000,
                           vi_per_class: int = 4000) -> pd.DataFrame:
    """
    Đọc, làm sạch và kết hợp dữ liệu tiếng Anh + tiếng Việt.
    Trả về DataFrame với cột: text, label (0=Human, 1=AI).

    Không augment dữ liệu ở đây — augment phải được gọi sau khi
    train_test_split, chỉ trên phần train, để tránh leakage.
    """
    # --- Tiếng Anh: AI_Human.csv có cột text, generated ---
    df_en = pd.read_csv(path_en)
    df_en = df_en.rename(columns={'generated': 'label'})[['text', 'label']]
    df_en['text'] = df_en['text'].apply(clean_text)
    df_en = df_en.dropna(subset=['text'])
    df_en = df_en[df_en['text'].str.len() > 20]

    en_parts = []
    for lbl in [0, 1]:
        subset = df_en[df_en['label'] == lbl]
        en_parts.append(subset.sample(n=min(en_per_class, len(subset)), random_state=42))
    df_en = pd.concat(en_parts)

    # --- Tiếng Việt: dataset_vi.csv có cột text, label sẵn ---
    df_vi = pd.read_csv(path_vi)
    df_vi['text'] = df_vi['text'].apply(clean_text)
    df_vi = df_vi.dropna(subset=['text'])
    df_vi = df_vi[df_vi['text'].str.len() > 20]

    vi_parts = []
    for lbl in [0, 1]:
        subset = df_vi[df_vi['label'] == lbl]
        vi_parts.append(subset.sample(n=min(vi_per_class, len(subset)), random_state=42))
    df_vi = pd.concat(vi_parts)

    # --- Kết hợp ---
    df = pd.concat([df_en, df_vi], ignore_index=True)
    df['label'] = df['label'].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def extract_structural_features(text: str) -> dict:
    """
    Trích xuất đặc trưng cấu trúc dùng cho Bayesian Network.
    Bổ sung sentence_count để BN có thêm tín hiệu phân biệt AI/Human
    (văn bản AI thường có số câu đều và nhiều hơn).
    """
    text = str(text)
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return {
        'length': len(text),
        'word_count': len(words),
        'avg_word_len': float(np.mean([len(w) for w in words])) if words else 0.0,
        'sentence_count': len(sentences),
    }
