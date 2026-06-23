import pandas as pd
import numpy as np
import re


def clean_text(text: str) -> str:
    """Làm sạch văn bản: chuẩn hóa khoảng trắng, xuống dòng."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def repair_mojibake(text: str) -> str:
    """Sửa lỗi encoding tiếng Việt bị vỡ (mojibake)."""
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def expand_vietnamese_samples(df_vi: pd.DataFrame, target_per_class: int = 2000) -> pd.DataFrame:
    """
    Tăng cường dữ liệu tiếng Việt bằng cách thay thế từ đồng nghĩa
    và thêm prefix/suffix ngữ cảnh.
    """
    synonyms = {
        "tôi": ["mình", "em", "chúng tôi"],
        "mô hình": ["hệ thống", "giải pháp", "phương pháp"],
        "nghiên cứu": ["tìm hiểu", "khảo sát", "phân tích"],
        "kết quả": ["kết luận", "đầu ra", "output"],
        "dữ liệu": ["thông tin", "tập mẫu", "dataset"],
    }
    prefixes = ["Theo nghiên cứu, ", "Có thể thấy rằng ", "Nhìn chung, ", ""]
    suffixes = [" Đây là một điểm quan trọng.", " Điều này rất đáng chú ý.", ""]

    augmented_rows = []
    for _, row in df_vi.iterrows():
        text = str(row['text'])
        label = row['label']
        augmented_rows.append({'text': text, 'label': label})

        for _ in range(target_per_class // max(len(df_vi[df_vi['label'] == label]), 1)):
            new_text = text
            for word, replacements in synonyms.items():
                if word in new_text:
                    new_text = new_text.replace(word, np.random.choice(replacements), 1)
            prefix = np.random.choice(prefixes)
            suffix = np.random.choice(suffixes)
            new_text = prefix + new_text + suffix
            augmented_rows.append({'text': new_text, 'label': label})

    df_aug = pd.DataFrame(augmented_rows).drop_duplicates(subset='text')
    # Giữ đúng target_per_class mẫu mỗi nhãn
    result = []
    for lbl in df_aug['label'].unique():
        subset = df_aug[df_aug['label'] == lbl]
        result.append(subset.sample(n=min(target_per_class, len(subset)), random_state=42))
    return pd.concat(result).reset_index(drop=True)


def load_and_prepare_data(path_en: str, path_vi: str,
                          en_per_class: int = 4000,
                          vi_per_class: int = 2000) -> pd.DataFrame:
    """
    Đọc, làm sạch và kết hợp dữ liệu tiếng Anh + tiếng Việt.
    Trả về DataFrame với cột: text, label (0=Human, 1=AI).
    """
    # --- Tiếng Anh ---
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

    # --- Tiếng Việt ---
    df_vi = pd.read_csv(path_vi)
    df_vi['text'] = df_vi['text'].apply(repair_mojibake).apply(clean_text)
    df_vi = df_vi.dropna(subset=['text'])
    df_vi = df_vi[df_vi['text'].str.len() > 20]
    df_vi = expand_vietnamese_samples(df_vi, target_per_class=vi_per_class)

    # --- Kết hợp ---
    df = pd.concat([df_en, df_vi], ignore_index=True)
    df['label'] = df['label'].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def extract_structural_features(text: str) -> dict:
    """Trích xuất đặc trưng cấu trúc dùng cho Bayesian Network."""
    text = str(text)
    words = text.split()
    return {
        'length': len(text),
        'word_count': len(words),
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
    }