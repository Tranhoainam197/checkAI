# CheckAI - Ứng dụng phân loại văn bản do AI sinh ra

Dự án thuộc học phần **Trí tuệ nhân tạo (HP: 121033)**, Trường Đại học Giao thông Vận tải TP.HCM.

## 📝 Giới thiệu đề tài
Hệ thống sử dụng các thuật toán xác suất (**Naive Bayes** và **Mạng Bayes**) kết hợp với các kỹ thuật **Heuristic** để phân tích và dự đoán xem một văn bản là do con người viết hay do AI (như ChatGPT, Gemini) sinh ra.

## 🛠 Cấu trúc thư mục
- `data/`: Chứa các tập dữ liệu huấn luyện.
- `artifacts/`: Chứa các model đã được huấn luyện (`.pkl`).
- `app.py`: Giao diện ứng dụng (Streamlit).
- `predictor.py`: Logic dự đoán và kết hợp mô hình (Ensemble).
- `bayesian_network.py`: Triển khai mô hình Mạng Bayes.

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt thư viện
Đảm bảo bạn đã cài đặt Python 3.9+, sau đó chạy lệnh sau trong terminal:
```bash
pip install -r requirements.txt