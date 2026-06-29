import csv
import random
import re
import time
import wikipediaapi

TARGET_PER_CLASS = 1500
MAX_PER_TOPIC = 80   
MIN_LEN = 150
MAX_LEN = 600
OUTPUT_PATH = "data/dataset_vi.csv"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

WIKI_TOPICS = [
    "Trí tuệ nhân tạo", "Học máy", "Mạng nơ-ron nhân tạo", "Xử lý ngôn ngữ tự nhiên",
    "Internet", "Điện toán đám mây", "Chuỗi khối", "Robot", "Vũ trụ", "Vật lý",
    "Hóa học", "Sinh học", "Di truyền học", "Tiến hóa", "Hệ mặt trời",
    "Biến đổi khí hậu", "Năng lượng tái tạo", "Điện hạt nhân", "Trái Đất",
    "Lịch sử Việt Nam", "Chiến tranh Việt Nam", "Nhà Nguyễn", "Hồ Chí Minh",
    "Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Sông Mê Kông", "Biển Đông",
    "Lịch sử Trung Quốc", "Thế chiến thứ hai", "Cách mạng Pháp", "Đế quốc La Mã",
    "Phật giáo", "Hồi giáo", "Kitô giáo", "Triết học", "Tâm lý học",
    "Kinh tế học", "Toàn cầu hóa", "Dân số thế giới", "Giáo dục",
    "Văn học Việt Nam", "Âm nhạc", "Điện ảnh", "Thể thao", "Bóng đá",
    "COVID-19", "Ung thư", "Tiểu đường", "Tim mạch", "Vaccine",
    "Y học cổ truyền", "Dinh dưỡng", "Sức khỏe tâm thần",
    "Kinh tế Việt Nam", "Chứng khoán", "Ngân hàng", "Thương mại điện tử",
    "Nông nghiệp", "Du lịch", "Công nghiệp",
]

# Template sinh văn bản AI theo văn phong AI chuẩn
# {topic} sẽ được thay bằng chủ đề thật, {sentences} bằng câu từ Wikipedia
# Đặc trưng văn phong AI cần học:
#   - Câu hoàn chỉnh, đều đặn, ít biến động độ dài
#   - Từ nối: "tuy nhiên", "bên cạnh đó", "nhìn chung", "đáng chú ý"
#   - Không có tên riêng lạ, số liệu cụ thể, lỗi gõ
#   - Kết luận tổng quát ở cuối

AI_TEMPLATES = [
    "{topic} là một lĩnh vực quan trọng được nhiều nhà nghiên cứu quan tâm. {fact1} Tuy nhiên, vẫn còn nhiều thách thức cần được giải quyết trong lĩnh vực này. Bên cạnh đó, {fact2} Nhìn chung, đây là chủ đề có ý nghĩa lớn đối với sự phát triển của xã hội hiện đại.",
    "Trong bối cảnh hiện nay, {topic} đóng vai trò ngày càng quan trọng. {fact1} Đáng chú ý là {fact2} Bên cạnh đó, các chuyên gia nhận định rằng lĩnh vực này sẽ tiếp tục phát triển mạnh mẽ trong tương lai. Tuy nhiên, cần có sự chuẩn bị kỹ lưỡng để đối phó với những khó khăn phía trước.",
    "Có thể thấy rằng {topic} có tác động sâu rộng đến nhiều mặt của cuộc sống. {fact1} Nhìn chung, {fact2} Tuy nhiên, không phải mọi vấn đề đều đã được giải quyết thỏa đáng. Bên cạnh đó, sự hợp tác quốc tế đóng vai trò then chốt trong việc thúc đẩy tiến bộ.",
    "Theo các chuyên gia, {topic} đang trải qua giai đoạn phát triển quan trọng. {fact1} Bên cạnh đó, {fact2} Đáng chú ý là những tiến bộ gần đây đã mở ra nhiều hướng ứng dụng mới. Nhìn chung, triển vọng trong lĩnh vực này được đánh giá là khá tích cực.",
    "{topic} được xem là một trong những vấn đề trọng tâm của thời đại. {fact1} Tuy nhiên, cần nhìn nhận rằng {fact2} Bên cạnh đó, việc nâng cao nhận thức cộng đồng cũng đóng vai trò không kém phần quan trọng. Nhìn chung, đây là lĩnh vực đòi hỏi sự đầu tư nghiêm túc và lâu dài.",
    "Nghiên cứu cho thấy {topic} có nhiều điểm cần được xem xét một cách toàn diện. {fact1} Đáng chú ý là {fact2} Tuy nhiên, các giải pháp hiện tại vẫn còn nhiều hạn chế. Bên cạnh đó, sự phát triển công nghệ đang mở ra những cơ hội mới để giải quyết các khó khăn còn tồn tại.",
    # --- Template mới không dùng "khía cạnh", "tích cực", "toàn diện" ---
    "Hiện nay, {topic} là một trong những chủ đề được quan tâm rộng rãi. {fact1} Dù vậy, còn nhiều điều cần làm rõ hơn về lĩnh vực này. {fact2} Các nhà nghiên cứu cho rằng cần có thêm dữ liệu và phân tích sâu hơn để đưa ra kết luận chính xác.",
    "Sự phát triển của {topic} trong những năm gần đây đã thu hút sự chú ý của nhiều người. {fact1} Song song đó, {fact2} Điều này cho thấy tầm quan trọng ngày càng tăng của lĩnh vực này trong đời sống xã hội.",
    "Khi nói đến {topic}, điều đầu tiên cần nhắc đến là vai trò của nó trong cuộc sống hiện đại. {fact1} Mặt khác, {fact2} Vì vậy, việc tìm hiểu và nghiên cứu sâu hơn về chủ đề này là điều cần thiết.",
    "{topic} đã và đang trở thành chủ đề được thảo luận nhiều trong cộng đồng khoa học. {fact1} Hơn nữa, {fact2} Do đó, cần có những chính sách và giải pháp phù hợp để phát huy tiềm năng của lĩnh vực này.",
    "Không thể phủ nhận rằng {topic} đang thay đổi cách chúng ta nhìn nhận thế giới. {fact1} Thêm vào đó, {fact2} Chính vì vậy, nhiều tổ chức và cá nhân đang tích cực tham gia vào quá trình nghiên cứu và ứng dụng trong lĩnh vực này.",
    "Trong những năm trở lại đây, {topic} đã có những bước tiến đáng kể. {fact1} Tuy vậy, {fact2} Đây là lý do tại sao giới chuyên môn vẫn tiếp tục tìm kiếm những phương pháp tiếp cận mới và hiệu quả hơn.",
    "Một trong những xu hướng nổi bật hiện nay là sự quan tâm ngày càng tăng đối với {topic}. {fact1} Cùng với đó, {fact2} Điều này phản ánh nhu cầu ngày càng cao của xã hội đối với sự hiểu biết sâu sắc hơn về lĩnh vực này.",
    "Đối với nhiều người, {topic} vẫn còn là một lĩnh vực khá mới mẻ và cần được tìm hiểu thêm. {fact1} Tuy nhiên, {fact2} Rõ ràng là đây là một chủ đề không thể bỏ qua trong thế giới ngày nay.",
    "{topic} ngày càng nhận được nhiều sự quan tâm từ cả cộng đồng học thuật lẫn xã hội. {fact1} Đồng thời, {fact2} Những tiến bộ này mở ra nhiều cơ hội mới, nhưng cũng đặt ra không ít câu hỏi cần được giải đáp.",
]

# Biến đổi câu Wikipedia → câu AI (loại bỏ tên riêng, số liệu cụ thể, lỗi)
def wikify_to_ai_style(sentence: str) -> str:
    """
    Chuyển câu Wikipedia sang văn phong AI:
    - Xóa năm cụ thể (1945, 2023...) → thay bằng "trong giai đoạn này"
    - Xóa tên người/địa danh rõ ràng → giữ lại nếu là từ khoá chủ đề
    - Làm mượt câu bằng cách bổ sung từ nối
    """
    # Thay năm cụ thể
    sentence = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', 'giai đoạn này', sentence)
    # Xóa số liệu phần trăm/thống kê cụ thể
    sentence = re.sub(r'\d+[,.]?\d*\s*%', 'một tỷ lệ đáng kể', sentence)
    sentence = re.sub(r'\d+[,.]?\d*\s*(triệu|tỷ|nghìn)', 'một số lượng lớn', sentence)
    # Làm sạch ký tự thừa
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    return text.strip()


def split_to_chunks(text: str, min_len=MIN_LEN, max_len=MAX_LEN) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current = (current + " " + sent).strip()
        else:
            if len(current) >= min_len:
                chunks.append(current)
            current = sent
    if len(current) >= min_len:
        chunks.append(current)
    return chunks


def crawl_wikipedia_vi(topics: list, target: int, max_per_topic: int = 80):
    wiki = wikipediaapi.Wikipedia(
        language='vi',
        user_agent='CheckAI-Dataset-Builder/2.0 (educational project)'
    )
    human_chunks = []
    topic_sentences = {}
    random.shuffle(topics)

    for topic in topics:
        if len(human_chunks) >= target:
            break
        try:
            page = wiki.page(topic)
            if not page.exists():
                print(f"   [WIKI] Không tìm thấy: {topic}")
                continue
            text = clean(page.text)
            chunks = split_to_chunks(text)

            added = 0
            for chunk in chunks:
                if len(human_chunks) >= target or added >= max_per_topic:
                    break
                human_chunks.append(chunk)
                added += 1

            intro = text[:800]
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', intro)
                     if 30 < len(s.strip()) < 150]
            if sents:
                topic_sentences[topic] = sents[:8]

            print(f"   [WIKI] {topic}: +{added} đoạn (tổng {len(human_chunks)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"   [WIKI] Lỗi {topic}: {e}")
            continue

    return human_chunks, topic_sentences


def generate_ai_texts_from_topics(topic_sentences: dict, target: int) -> list:
    """
    Sinh văn bản AI từ câu Wikipedia theo ĐÚNG CHỦ ĐỀ,
    nhưng viết lại theo văn phong AI (từ nối, câu đều, tổng quát).

    Chiến lược:
    1. Lấy 2 câu thực tế từ Wikipedia của chủ đề đó
    2. Biến đổi nhẹ để bỏ số liệu/tên riêng cụ thể
    3. Bọc vào template AI (thêm từ nối chuẩn AI)
    → Mô hình phải học VĂN PHONG chứ không phải CHỦ ĐỀ
    """
    rng = random.Random(RANDOM_SEED)
    results = []
    topics = list(topic_sentences.keys())

    attempts = 0
    while len(results) < target and attempts < target * 20:
        attempts += 1
        topic = rng.choice(topics)
        sents = topic_sentences[topic]
        if len(sents) < 2:
            continue

        # Chọn 2 câu ngẫu nhiên từ chủ đề, biến đổi sang AI style
        s1, s2 = rng.sample(sents, 2)
        fact1 = wikify_to_ai_style(s1)
        fact2 = wikify_to_ai_style(s2)

        # Đảm bảo fact2 không bắt đầu bằng hoa (vì nằm giữa câu)
        if fact2 and fact2[0].isupper():
            fact2 = fact2[0].lower() + fact2[1:]

        template = rng.choice(AI_TEMPLATES)
        text = template.format(
            topic=topic,
            fact1=fact1 if fact1.endswith('.') else fact1 + '.',
            fact2=fact2 if fact2.endswith('.') else fact2 + '.',
        )
        text = re.sub(r'\s+', ' ', text).strip()

        if MIN_LEN <= len(text) <= MAX_LEN and text not in results:
            results.append(text)

    return results[:target]


def main():
    print("=" * 60)
    print("Build dataset tiếng Việt v2 — cùng chủ đề, khác văn phong")
    print("=" * 60)

    try:
        import wikipediaapi
    except ImportError:
        print("\n❌ Chạy: pip install wikipedia-api")
        return

    # 1. Crawl Wikipedia → Human + lấy câu để sinh AI
    print(f"\n[1/2] Crawl Wikipedia (target: {TARGET_PER_CLASS} đoạn)...")
    human_texts, topic_sentences = crawl_wikipedia_vi(WIKI_TOPICS, TARGET_PER_CLASS, max_per_topic=80)
    print(f"   → {len(human_texts)} đoạn Human | {len(topic_sentences)} chủ đề")

    if len(human_texts) < TARGET_PER_CLASS * 0.5:
        print("❌ Quá ít dữ liệu. Kiểm tra kết nối internet.")
        return

    # 2. Sinh AI texts cùng chủ đề
    print(f"\n[2/2] Sinh văn bản AI cùng chủ đề (target: {TARGET_PER_CLASS})...")
    ai_texts = generate_ai_texts_from_topics(topic_sentences, TARGET_PER_CLASS)
    print(f"   → {len(ai_texts)} đoạn AI")

    if len(ai_texts) < 100:
        print("❌ Quá ít mẫu AI. Kiểm tra topic_sentences.")
        return

    # 3. Cân bằng
    n = min(len(human_texts), len(ai_texts))
    human_texts = random.sample(human_texts, n)
    ai_texts    = random.sample(ai_texts, n)
    print(f"   Cân bằng: {n} mẫu/nhãn (tổng {n*2})")

    # 4. Ghi file
    rows = (
        [{'text': t, 'label': 0} for t in human_texts] +
        [{'text': t, 'label': 1} for t in ai_texts]
    )
    random.shuffle(rows)

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Đã lưu {len(rows)} mẫu vào '{OUTPUT_PATH}'")
    print("   Chạy 'python train.py' rồi 'python check_vi_model.py' để kiểm tra lại.")

    # 5. Kiểm tra nhanh — so sánh cặp cùng chủ đề
    print("\n--- Kiểm tra cặp cùng chủ đề ---")
    sample_topic = list(topic_sentences.keys())[0]
    print(f"Chủ đề: {sample_topic}")
    h = [t for t in human_texts if True]  # bất kỳ human
    a = [t for t in ai_texts if sample_topic in t]
    if h:
        print(f"Human: {h[0][:180]}...")
    if a:
        print(f"AI:    {a[0][:180]}...")


if __name__ == "__main__":
    main()