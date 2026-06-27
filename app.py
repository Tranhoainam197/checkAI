import streamlit as st
import requests
import pandas as pd
import os
import io
import docx
import pypdf
from bs4 import BeautifulSoup
from predictor import predict_text, predict_line_by_line
from report_utils import get_text_stats, evaluate_on_test_set

st.set_page_config(page_title="CheckAI - Hệ Thống", page_icon="🤖", layout="wide")
ARTIFACTS_EXIST = os.path.exists("artifacts/model_nb.pkl")


def extract_text_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        return ' '.join(p.get_text() for p in BeautifulSoup(resp.text, 'html.parser').find_all('p'))
    except Exception as e: return f"[Lỗi: {e}]"

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith('.txt'): return uploaded_file.read().decode('utf-8', errors='ignore')
    elif name.endswith('.docx'): return '\n'.join(p.text for p in docx.Document(io.BytesIO(uploaded_file.read())).paragraphs)
    elif name.endswith('.pdf'): return '\n'.join(page.extract_text() or '' for page in pypdf.PdfReader(io.BytesIO(uploaded_file.read())).pages)
    return ""

# Sidebar UI
st.sidebar.title("⚙️ Cấu Hình")
url_input = st.sidebar.text_input("🔗 Nhập URL bài viết")
uploaded_file = st.sidebar.file_uploader("📁 Tải file", type=['txt', 'docx', 'pdf'])
line_by_line = st.sidebar.toggle("🔍 Chế độ phân tích từng dòng")

st.sidebar.divider()
if st.sidebar.button("Hiển thị Confusion Matrix") and ARTIFACTS_EXIST:
    with st.sidebar.spinner("Trích xuất..."):
        evaluate_on_test_set()
        if os.path.exists("artifacts/confusion_matrix.png"):
            st.sidebar.image("artifacts/confusion_matrix.png", caption="Tập Test Set")

# Main UI
st.title("🤖 CheckAI — Phân Tích Văn Bản Đa Ngôn Ngữ")
prefilled = extract_text_from_url(url_input) if url_input else extract_text_from_file(uploaded_file) if uploaded_file else ""

text_input = st.text_area("✏️ Nội dung kiểm tra", value=prefilled, height=230, placeholder="Dán văn bản vào đây...")
analyze_btn = st.button("🚀 Phân Tích", type="primary", disabled=not ARTIFACTS_EXIST)

if not ARTIFACTS_EXIST: st.warning("⚠️ Chạy file `train.py` để huấn luyện hệ thống trước.")

if analyze_btn and text_input.strip():
    with st.spinner("Đang suy luận..."):
        result = predict_text(text_input)

    is_ai = result['label'] == "AI Generated"
    color, icon = ("#ff4b4b", "🤖") if is_ai else ("#21c354", "✍️")

    st.markdown(f"""
    <div style="background:{color}15; border-left: 6px solid {color}; padding: 20px; border-radius: 8px; margin: 15px 0;">
        <h2 style="color:{color}; margin:0;">{icon} {result['label']}</h2>
        <p style="margin:8px 0 0 0; font-size:18px;">Độ tin cậy: <strong style="color:{color};">{result['confidence']}%</strong></p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("1. Naive Bayes", f"{result['nb_score']}%")
    col2.metric("2. Bayesian Network", f"{result['bn_score']}%")
    col3.metric("3. Heuristic Rules", f"{result['heuristic_score']}%")

    st.divider()
    tab1, tab2 = st.tabs(["📋 Thống Kê Đặc Trưng", "📄 Phân Tích Theo Dòng"])

    with tab1:
        st.table(pd.DataFrame(pd.Series(get_text_stats(text_input)).rename("Chỉ số")))

    with tab2:
        if line_by_line:
            with st.spinner("Quét cục bộ..."):
                line_res = predict_line_by_line(text_input)
            if line_res:
                df_lines = pd.DataFrame(line_res)[['line', 'label', 'confidence']]
                df_lines.columns = ['Đoạn văn bản', 'Dự đoán', 'Độ tin cậy (%)']
                st.dataframe(df_lines, use_container_width=True)
            else: st.info("Văn bản quá ngắn.")
        else: st.info("💡 Bật 'Chế độ phân tích từng dòng' bên thanh cấu hình để quét sâu.")