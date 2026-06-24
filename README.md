# CheckAI — Ứng dụng phân loại văn bản do AI sinh ra

Dự án thuộc học phần **Trí tuệ nhân tạo (HP: 121033)**, Trường Đại học Giao thông Vận tải TP.HCM.

## 📝 Giới thiệu đề tài

Hệ thống sử dụng các thuật toán xác suất (**Naive Bayes** và **Mạng Bayes**) kết hợp với kỹ thuật **Heuristic** để phân tích và dự đoán xem một văn bản là do con người viết hay do AI (như ChatGPT, Gemini) sinh ra. Hỗ trợ song ngữ **Tiếng Anh** và **Tiếng Việt**.

## 🛠 Cấu trúc thư mục

```
CheckAI/
├── data/               # Tập dữ liệu huấn luyện
│   ├── AI_Human.csv    # Dữ liệu tiếng Anh
│   └── dataset_vi.csv  # Dữ liệu tiếng Việt
├── artifacts/          # Model và kết quả đã lưu (.pkl, .png, .csv)
├── app.py              # Giao diện ứng dụng (Streamlit)
├── predictor.py        # Logic dự đoán và kết hợp mô hình
├── bayesian_network.py # Triển khai mô hình Mạng Bayes
├── data_utils.py       # Tiền xử lý và tăng cường dữ liệu
├── train.py            # Huấn luyện và tuning mô hình
├── report_utils.py     # Đánh giá và xuất báo cáo
└── requirements.txt    # Danh sách thư viện
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

Công thức kết hợp cuối cùng:
```
combined = (NB × 0.80 + Heuristic) × 0.85 + BN × 0.15
Ngưỡng quyết định: combined ≥ 0.5 → AI Generated
```

## 📊 Kết quả

- **Accuracy:** 94.58% trên tập test 2.400 mẫu
- **Precision (AI):** 98.20%
- **Recall (Human):** 98.33%