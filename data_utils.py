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
    Chỉ áp dụng khi văn bản gốc KHÔNG chứa ký tự Unicode hợp lệ sẵn.
    """
    if not isinstance(text, str) or not text:
        return text
    if any(ord(c) > 127 for c in text):
        return text
    try:
        repaired = text.encode('latin-1').decode('utf-8')
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# ── Augment tiếng Việt ───────────────────────────────────────────────────────

# Từ điển đồng nghĩa mở rộng (từ 5 → 20 nhóm)
_SYNONYMS_VI = {
    "tôi": ["mình", "em", "chúng tôi", "bản thân"],
    "bạn": ["các bạn", "anh", "chị", "mọi người"],
    "mô hình": ["hệ thống", "giải pháp", "phương pháp", "công cụ"],
    "nghiên cứu": ["tìm hiểu", "khảo sát", "phân tích", "điều tra"],
    "kết quả": ["kết luận", "đầu ra", "kết quả thu được", "thành quả"],
    "dữ liệu": ["thông tin", "tập mẫu", "dataset", "tập dữ liệu"],
    "quan trọng": ["cần thiết", "thiết yếu", "có ý nghĩa", "đáng chú ý"],
    "thực hiện": ["tiến hành", "triển khai", "áp dụng", "làm"],
    "cho thấy": ["chứng minh", "minh chứng", "chỉ ra", "phản ánh"],
    "vấn đề": ["bài toán", "thách thức", "hạn chế", "điểm yếu"],
    "phát triển": ["cải tiến", "nâng cao", "xây dựng", "hoàn thiện"],
    "sử dụng": ["áp dụng", "dùng", "tận dụng", "khai thác"],
    "hiệu quả": ["tốt", "chính xác", "tối ưu", "vượt trội"],
    "đề xuất": ["kiến nghị", "gợi ý", "đưa ra", "trình bày"],
    "phương pháp": ["cách tiếp cận", "kỹ thuật", "chiến lược", "hướng"],
    "ứng dụng": ["áp dụng thực tế", "triển khai", "tích hợp", "sử dụng"],
    "học": ["tìm hiểu", "nghiên cứu", "khám phá", "nắm bắt"],
    "cần": ["cần thiết phải", "nên", "phải", "đòi hỏi"],
    "tuy nhiên": ["nhưng", "song", "dù vậy", "mặc dù"],
    "đặc biệt": ["nổi bật", "đáng chú ý", "điển hình", "cụ thể"],
}

_PREFIXES_VI = [
    "Theo nghiên cứu, ",
    "Có thể thấy rằng ",
    "Nhìn chung, ",
    "Thực tế cho thấy, ",
    "Theo quan sát, ",
    "Dựa trên phân tích, ",
    "Đáng lưu ý là ",
    "",  # giữ nguyên không thêm prefix
]

_SUFFIXES_VI = [
    " Đây là một điểm quan trọng.",
    " Điều này rất đáng chú ý.",
    " Cần xem xét kỹ hơn về vấn đề này.",
    " Đây là xu hướng phổ biến hiện nay.",
    " Kết quả này mở ra nhiều hướng nghiên cứu mới.",
    "",  # giữ nguyên không thêm suffix
]


def expand_vietnamese_samples(df_vi: pd.DataFrame, multiplier: int = 3,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Tăng cường dữ liệu tiếng Việt bằng từ đồng nghĩa + prefix/suffix.

    Cải tiến so với phiên bản cũ:
    - Từ điển đồng nghĩa mở rộng từ 5 → 20 nhóm từ
    - multiplier mặc định tăng từ 1 → 3 (sinh 3 bản augment / mẫu gốc)
    - Mỗi lần thay thế nhiều từ hơn (lặp ngẫu nhiên 1–3 lần thay vì 1 lần)
    - Prefix/suffix phong phú hơn (8 prefix, 6 suffix)

    CHỈ gọi trên tập TRAIN sau khi train_test_split để tránh data leakage.
    """
    rng = np.random.RandomState(random_state)
    augmented_rows = [{'text': t, 'label': l}
                      for t, l in zip(df_vi['text'], df_vi['label'])]

    synonym_items = list(_SYNONYMS_VI.items())

    for text, label in zip(df_vi['text'], df_vi['label']):
        for _ in range(multiplier):
            new_text = str(text)

            # Chọn ngẫu nhiên 1–4 nhóm từ để thay thế (tăng đa dạng)
            n_replace = rng.randint(1, 5)
            chosen = rng.choice(len(synonym_items), size=n_replace, replace=False)
            for idx in chosen:
                word, replacements = synonym_items[idx]
                if word in new_text:
                    new_text = new_text.replace(word, rng.choice(replacements), 1)

            prefix = rng.choice(_PREFIXES_VI)
            suffix = rng.choice(_SUFFIXES_VI)
            new_text = prefix + new_text + suffix
            augmented_rows.append({'text': new_text, 'label': label})

    df_aug = (pd.DataFrame(augmented_rows)
                .drop_duplicates(subset='text')
                .reset_index(drop=True))
    return df_aug


# ── Load dữ liệu ─────────────────────────────────────────────────────────────

def load_and_prepare_data(path_en: str, path_vi: str,
                           en_per_class: int = 4000,
                           vi_per_class: int = 4000):
    """
    Đọc, làm sạch dữ liệu tiếng Anh + tiếng Việt.
    Trả về (df_en, df_vi) riêng biệt thay vì gộp chung,
    để train.py có thể huấn luyện NB riêng cho từng ngôn ngữ.

    Không augment ở đây — augment phải gọi sau train_test_split,
    chỉ trên phần train, để tránh data leakage.
    """
    # --- Tiếng Anh ---
    df_en = pd.read_csv(path_en)
    df_en = df_en.rename(columns={'generated': 'label'})[['text', 'label']]
    df_en['text'] = df_en['text'].apply(clean_text)
    df_en = df_en.dropna(subset=['text'])
    df_en = df_en[df_en['text'].str.len() > 20]
    df_en['lang'] = 'en'

    en_parts = []
    for lbl in [0, 1]:
        subset = df_en[df_en['label'] == lbl]
        en_parts.append(subset.sample(n=min(en_per_class, len(subset)), random_state=42))
    df_en = pd.concat(en_parts).reset_index(drop=True)

    # --- Tiếng Việt ---
    df_vi = pd.read_csv(path_vi)
    df_vi['text'] = df_vi['text'].apply(clean_text)
    df_vi = df_vi.dropna(subset=['text'])
    df_vi = df_vi[df_vi['text'].str.len() > 20]
    df_vi['lang'] = 'vi'

    vi_parts = []
    for lbl in [0, 1]:
        subset = df_vi[df_vi['label'] == lbl]
        vi_parts.append(subset.sample(n=min(vi_per_class, len(subset)), random_state=42))
    df_vi = pd.concat(vi_parts).reset_index(drop=True)

    return df_en, df_vi


# ── Đặc trưng cấu trúc cho Bayesian Network ──────────────────────────────────

def extract_structural_features(text: str) -> dict:
    """
    Trích xuất đặc trưng cấu trúc dùng cho Bayesian Network.

    Cải tiến so với phiên bản cũ:
    - Giữ 4 đặc trưng gốc: length, word_count, avg_word_len, sentence_count
    - Thêm type_token_ratio: tỉ lệ từ độc nhất / tổng từ
      → AI thường lặp từ ít hơn human (TTR cao hơn một chút)
      → nhưng AI cũng đôi khi dùng từ vựng đơn điệu → TTR thấp
      → đây là tín hiệu bổ sung hữu ích cho BN
    - Thêm repetition_rate: tỉ lệ từ xuất hiện ≥ 3 lần / tổng từ
      → AI có xu hướng lặp một số từ chủ chốt nhiều lần hơn human
    """
    text = str(text)
    words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # type-token ratio
    if words:
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        # Đếm từ lặp ≥ 3 lần
        from collections import Counter
        freq = Counter(words)
        repeated = sum(1 for cnt in freq.values() if cnt >= 3)
        repetition_rate = repeated / len(words)
    else:
        ttr = 0.0
        repetition_rate = 0.0

    return {
        'length': len(text),
        'word_count': len(words),
        'avg_word_len': float(np.mean([len(w) for w in words])) if words else 0.0,
        'sentence_count': len(sentences),
        'type_token_ratio': round(ttr, 4),
        'repetition_rate': round(repetition_rate, 4),
    }
