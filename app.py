import streamlit as st
import requests
import pandas as pd
import joblib
from bs4 import BeautifulSoup
import docx
import pypdf
import io
import os
import pandas as pd
from predictor import predict_text, predict_line_by_line
from report_utils import get_text_stats, evaluate_on_test_set

st.set_page_config(page_title="CheckAI", page_icon="🤖", layout="wide")

ARTIFACTS_EXIST = os.path.exists("artifacts/model_nb.pkl")


def extract_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"[Lỗi khi tải URL: {e}]"


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return '\n'.join(p.text for p in doc.paragraphs)
    elif name.endswith('.pdf'):
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    return ""


# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("⚙️ Tùy chọn")

url_input = st.sidebar.text_input("🔗 Nhập URL bài viết")
uploaded_file = st.sidebar.file_uploader("📁 Tải file (.txt, .docx, .pdf)",
                                          type=['txt', 'docx', 'pdf'])
line_by_line = st.sidebar.toggle("🔍 Phân tích từng dòng", value=False)

if st.sidebar.button("📊 Xem báo cáo mô hình") and ARTIFACTS_EXIST:
    cm = evaluate_on_test_set()
    if os.path.exists("artifacts/confusion_matrix.png"):
        st.sidebar.image("artifacts/confusion_matrix.png")

# ── Main ─────────────────────────────────────────────────
st.title("🤖 CheckAI — Phát hiện văn bản do AI tạo ra")
st.caption("Dựa trên Naive Bayes + Bayesian Network | Hỗ trợ Tiếng Anh & Tiếng Việt")

# Xác định nguồn văn bản
prefilled = ""
if url_input:
    prefilled = extract_text_from_url(url_input)
elif uploaded_file:
    prefilled = extract_text_from_file(uploaded_file)

text_input = st.text_area("✏️ Nhập hoặc dán văn bản cần kiểm tra",
                           value=prefilled, height=250,
                           placeholder="Dán văn bản vào đây...")

analyze_btn = st.button("🚀 Phân tích", type="primary", disabled=not ARTIFACTS_EXIST)

if not ARTIFACTS_EXIST:
    st.warning("⚠️ Chưa có model. Hãy chạy `python train.py` trước.")

# ── Kết quả ──────────────────────────────────────────────
if analyze_btn and text_input.strip():
    with st.spinner("Đang phân tích..."):
        result = predict_text(text_input)

    # Result card
    is_ai = result['label'] == "AI Generated"
    color = "#ff4b4b" if is_ai else "#21c354"
    icon = "🤖" if is_ai else "✍️"

    st.markdown(f"""
    <div style="background:{color}22; border-left: 5px solid {color};
                padding: 16px; border-radius: 8px; margin: 12px 0;">
        <h2 style="color:{color}; margin:0">{icon} {result['label']}</h2>
        <p style="margin:4px 0; font-size:18px;">
            Độ tin cậy: <strong>{result['confidence']}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tín hiệu bổ sung
    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Bayes", f"{result['nb_score']}%")
    col2.metric("Bayesian Network", f"{result['bn_score']}%")
    col3.metric("Heuristic", f"{result['heuristic_score']}%")

    st.divider()

    tab1, tab2 = st.tabs(["📋 Tổng quan", "📄 Từng dòng"])

    with tab1:
        stats = get_text_stats(text_input)
        st.table(pd.Series(stats).rename("Giá trị"))

    with tab2:
        if line_by_line:
            with st.spinner("Đang phân tích từng dòng..."):
                line_results = predict_line_by_line(text_input)
            import pandas as pd
            df_lines = pd.DataFrame(line_results)[['line', 'label', 'confidence']]
            df_lines.columns = ['Nội dung', 'Nhãn', 'Độ tin cậy (%)']
            st.dataframe(df_lines, use_container_width=True)
        else:
            st.info("Bật 'Phân tích từng dòng' ở sidebar để xem chi tiết.")

elif analyze_btn:
    st.warning("Vui lòng nhập văn bản trước khi phân tích.")

