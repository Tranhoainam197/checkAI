# CheckAI — Ứng dụng phân loại văn bản do AI sinh ra

Xây dựng ứng dụng phân loại văn bản do AI sinh ra dựa trên xác suất

## 📝 Giới thiệu đề tài

Hệ thống sử dụng các thuật toán xác suất (**Naive Bayes** và **Mạng Bayes**) kết hợp với kỹ thuật **Heuristic** để phân tích và dự đoán xem một văn bản là do con người viết hay do AI (như ChatGPT, Gemini) sinh ra. Hỗ trợ song ngữ **Tiếng Anh** và **Tiếng Việt**.

## 🛠 Cấu trúc thư mục

```
CheckAI/
│
├── data/                         # Tập dữ liệu
│   ├── AI_Human.csv              # Dataset tiếng Anh
│   └── dataset_vi.csv            # Dataset tiếng Việt
│
├── artifacts/                    # Sinh ra sau khi train (không đưa lên Git)
│   ├── README.md
│   ├── model_bn.pkl
│   ├── model_nb.pkl
│   ├── meta_model.pkl
│   ├── bn_metadata.pkl
│   ├── confusion_matrix.png
│   ├── tuning_results.csv
│   └── test_set.csv
│
├── app.py                        # Giao diện Streamlit
├── train.py                      # Huấn luyện mô hình
├── predictor.py                  # Dự đoán văn bản
├── bayesian_network.py           # Mô hình Mạng Bayes
├── data_utils.py                 # Tiền xử lý dữ liệu
├── report_utils.py               # Sinh báo cáo
├── build_dataset_vi.py
├── requirements.txt              # Thư viện cần cài đặt
├── README.md                     # Tài liệu dự án
├── LICENSE
└── .gitignore
```

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu hệ thống
- Python **3.9+**
- Windows / macOS / Linux

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

```bash
# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý:** File `requirements.txt` đã ghim `numpy<2` để tương thích với pgmpy. Không cần cài lại thủ công.

### 4. Huấn luyện mô hình

```bash
python train.py
```

Sau khi chạy xong, thư mục `artifacts/` sẽ chứa các file model `.pkl` và kết quả tuning.

### 5. Khởi động ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## 🧠 Mô hình sử dụng

| Thành phần | Mô tả |
|---|---|
| **Naive Bayes** | Phân tích nội dung qua TF-IDF character n-gram (3–5) |
| **Bayesian Network** | Phân tích cấu trúc: độ dài, số từ, độ dài từ TB |
| **Heuristic** | Phát hiện dấu hiệu bề mặt: độ đều câu, dấu câu, chữ hoa |
