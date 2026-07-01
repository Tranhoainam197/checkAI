import streamlit as st
import requests
import pandas as pd
import os
import io
from bs4 import BeautifulSoup
import docx
import pypdf
from predictor import predict_text, predict_line_by_line
from report_utils import get_text_stats, evaluate_on_test_set

st.set_page_config(page_title="AI Text Detector", page_icon="🔍", layout="wide")

ARTIFACTS_EXIST = (
    os.path.exists("artifacts/model_nb_en.pkl") and
    os.path.exists("artifacts/model_nb_vi.pkl") and
    os.path.exists("artifacts/model_bn.pkl")
)

# ── Helpers ──────────────────────────────────────────────

def extract_text_from_url(url: str) -> tuple[str, str | None]:
    """Trả về (text, error_message). error_message=None nếu thành công."""
    try:
        resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs).strip()
        if not text:
            return "", "Không tìm thấy nội dung văn bản (thẻ <p>) tại URL này."
        return text, None
    except Exception as e:
        return "", f"Lỗi khi tải URL: {e}"


def extract_text_from_file(uploaded_file) -> tuple[str, str | None]:
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.txt'):
            return uploaded_file.read().decode('utf-8', errors='ignore'), None
        elif name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif name.endswith('.pdf'):
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            return '\n'.join(page.extract_text() or '' for page in reader.pages), None
        return "", "Định dạng file không được hỗ trợ."
    except Exception as e:
        return "", f"Lỗi khi đọc file: {e}"


# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        padding: 20px 24px; border-radius: 12px; margin: 16px 0;
        border-left: 6px solid var(--accent-color);
        background: var(--accent-color)15;
    }
    .result-title { margin: 0; font-size: 1.5rem; }
    .result-sub { margin: 6px 0 0 0; font-size: 1rem; opacity: 0.85; }
    .source-label { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AI Text Detector")
    st.caption("Naive Bayes · Bayesian Network · Heuristic")
    st.divider()

    st.subheader("📥 Nguồn văn bản")
    source = st.radio(
        "Chọn cách nhập văn bản",
        options=["✏️ Nhập văn bản", "📁 Tải lên tệp tin", "🔗 Nhập URL bài viết"],
        label_visibility="collapsed",
    )

    raw_text = ""
    input_error = None

    if source == "✏️ Nhập văn bản":
        st.markdown('<p class="source-label">Dán hoặc nhập văn bản cần kiểm tra</p>', unsafe_allow_html=True)
        raw_text = st.text_area(
            "Văn bản", height=220, label_visibility="collapsed",
            placeholder="Dán văn bản vào đây..."
        )

    elif source == "📁 Tải lên tệp tin":
        st.markdown('<p class="source-label">Hỗ trợ .txt, .docx, .pdf</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Tệp tin", type=['txt', 'docx', 'pdf'], label_visibility="collapsed"
        )
        if uploaded_file:
            raw_text, input_error = extract_text_from_file(uploaded_file)

    else:  # Nhập URL
        st.markdown('<p class="source-label">Đường dẫn bài viết</p>', unsafe_allow_html=True)
        url_input = st.text_input(
            "URL", label_visibility="collapsed", placeholder="https://..."
        )
        if url_input:
            raw_text, input_error = extract_text_from_url(url_input)

    st.divider()
    line_by_line = st.toggle("🔎 Phân tích từng dòng", value=False)

    st.divider()
    st.subheader("📊 Báo cáo mô hình")
    show_report = st.button("Xem báo cáo đánh giá", disabled=not ARTIFACTS_EXIST,
                             use_container_width=True)

    if not ARTIFACTS_EXIST:
        st.warning("Chưa có model đã huấn luyện. Hãy chạy `python train.py` trước.")

# ── Main ─────────────────────────────────────────────────
st.title("🔍 Hệ Thống Kiểm Tra Văn Bản AI")
st.caption("Phát hiện văn bản do AI sinh ra hay do con người viết — hỗ trợ Tiếng Anh & Tiếng Việt")

if input_error:
    st.warning(f"⚠️ {input_error}")

if source != "✏️ Nhập văn bản" and raw_text:
    with st.expander("📄 Xem nội dung đã trích xuất", expanded=False):
        st.text_area("Nội dung", value=raw_text, height=150, disabled=True,
                      label_visibility="collapsed")

analyze_btn = st.button("🚀 Phân tích văn bản", type="primary",
                         disabled=not ARTIFACTS_EXIST)

# ── Báo cáo mô hình ──────────────────────────────────────
if show_report:
    with st.spinner("Đang đánh giá hệ thống trên tập test..."):
        # Capture stdout để hiển thị classification report lên giao diện Streamlit
        import io as _io
        import sys as _sys
        _buf = _io.StringIO()
        _old_stdout = _sys.stdout
        _sys.stdout = _buf
        cm = evaluate_on_test_set()
        _sys.stdout = _old_stdout
        report_text = _buf.getvalue()

    if report_text:
        st.subheader("📋 Classification Report")
        st.code(report_text, language="")

    if cm is not None and os.path.exists("artifacts/confusion_matrix.png"):
        st.subheader("📊 Confusion Matrix — Hệ thống kết hợp")
        st.image("artifacts/confusion_matrix.png", width=400)

# ── Kết quả phân tích ────────────────────────────────────
if analyze_btn:
    if not raw_text.strip():
        st.warning("⚠️ Vui lòng nhập hoặc chọn văn bản trước khi phân tích.")
    else:
        with st.spinner("Đang phân tích..."):
            result = predict_text(raw_text)

        is_ai = result['label'] == "AI Generated"
        lang_label = "🇻🇳 Tiếng Việt" if result.get('language') == 'vi' else "🇺🇸 Tiếng Anh"
        threshold_label = f"Ngưỡng quyết định: {result.get('threshold_used', 0.5)}"
        accent = "#ff4b4b" if is_ai else "#21c354"
        icon = "🤖" if is_ai else "🧑"

        st.markdown(f"""
        <div class="result-card" style="--accent-color:{accent}; border-left-color:{accent}; background:{accent}1a;">
            <h2 class="result-title" style="color:{accent}">{icon} {result['label']}</h2>
            <p class="result-sub">Độ tin cậy: <strong>{result['confidence']}%</strong>
            &nbsp;·&nbsp; {lang_label} &nbsp;·&nbsp; {threshold_label}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Naive Bayes", f"{result['nb_score']:.2f}%",
                    help="Xác suất AI dựa trên nội dung (TF-IDF n-gram ký tự)")
        col2.metric("Bayesian Network", f"{result['bn_score']:.2f}%",
                    help="Xác suất AI dựa trên cấu trúc văn bản (length, word_count, sentence_count, avg_word_len)")
        col3.metric("Heuristic", f"{result['heuristic_score']:.2f}%",
                    help="Tín hiệu bề mặt: độ đều câu, dấu câu, từ dài, thiếu câu ngắn")

        st.divider()
        tab1, tab2 = st.tabs(["📋 Thống kê văn bản", "📄 Phân tích từng dòng"])

        with tab1:
            stats = get_text_stats(raw_text)
            df_stats = pd.Series(stats).rename("Giá trị").to_frame()
            df_stats["Giá trị"] = df_stats["Giá trị"].astype(int)
            st.table(df_stats.style.format({"Giá trị": "{:,}"}))

        with tab2:
            if line_by_line:
                with st.spinner("Đang phân tích từng dòng..."):
                    line_results = predict_line_by_line(raw_text)
                if line_results:
                    df_lines = pd.DataFrame(line_results)[['line', 'label', 'confidence']]
                    df_lines.columns = ['Nội dung', 'Nhãn', 'Độ tin cậy (%)']
                    st.dataframe(df_lines, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có dòng nào đủ dài (>20 ký tự) để phân tích riêng.")
            else:
                st.info("Bật 'Phân tích từng dòng' ở sidebar để xem chi tiết theo từng dòng.")

elif not raw_text.strip() and ARTIFACTS_EXIST:
    st.info("👈 Chọn nguồn văn bản ở sidebar, sau đó nhấn **Phân tích văn bản**.")
