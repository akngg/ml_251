## Bài tập lớn Học máy (CO3117)

Học kỳ: 1, Năm học: 2025–2026

---

## 1. Giới thiệu

Bài toán: Nhận diện chữ số viết tay (0–9) trên tập dữ liệu MNIST (phiên bản Kaggle Digit Recognizer). Ảnh xám 28×28 được trải phẳng thành 784 chiều. Mục tiêu là xây dựng, huấn luyện và đánh giá ít nhất hai mô hình học máy khác nhau, so sánh hiệu năng bằng các chỉ số: Accuracy, Precision, Recall, F1-score, và phân tích nhầm lẫn bằng Confusion Matrix.

Trong đồ án này, nhóm áp dụng hai phương pháp:
- Logistic Regression (LR) truyền thống (sklearn)
- Mạng neural nhiều lớp (MLP) đơn giản (PyTorch)

---

## 2. Dữ liệu và tiền xử lý

- Nguồn dữ liệu: Kaggle Digit Recognizer (data/train.csv, data/test.csv trong repo)
- Định dạng: mỗi dòng là một mẫu gồm 1 nhãn (label) và 784 pixel (0–255)
- Tiền xử lý:
	- Chuẩn hóa: chia cho 255.0 để đưa về [0, 1]
	- Tách tập: train/test = 80%/20% với stratify theo nhãn để giữ phân bố lớp
	- Với MLP: giữ dạng vector 784 chiều (không cần chuẩn hóa Z-score vì đầu vào là pixel đã scale)

Kích thước (in ra từ notebook khi chạy):
- Train: len(train_df)
- Test: len(test_df)

---

## 3. Thuật toán và kiến trúc mô hình

### 3.1 Logistic Regression (sklearn)
- Mục tiêu: phân loại đa lớp 10 nhãn
- Cấu hình chính (theo notebook `Logistic_Regression.ipynb`):
	- solver = 'lbfgs'
	- max_iter = 1000
	- n_jobs = -1 (có thể không ảnh hưởng với solver 'lbfgs')
- Hàm mất mát: log-loss (mặc định)
- Đầu ra: xác suất/điểm số cho 10 lớp, lấy argmax làm dự đoán

### 3.2 MLP (PyTorch)
- Kiến trúc (theo notebook `NeuralNetwork.ipynb`):
	- fc1: 784 → 512, ReLU, Dropout(0.2)
	- fc2: 512 → 256, ReLU, Dropout(0.2)
	- fc3: 256 → 128, ReLU, Dropout(0.1)
	- Skip connection: Linear(512 → 128) cộng tắt từ đầu ra fc1 sang tầng 128
	- fc4: 128 → 10 (logits)
	- Kích hoạt: ReLU cho các tầng ẩn, đầu ra là logits (dùng CrossEntropyLoss)
- Huấn luyện:
	- Tối ưu: Adam, lr = 5e-4
	- Batch size: 512
	- Epochs: 10
	- Thiết bị: CUDA nếu có, không thì CPU

Ghi chú thiết kế: MLP dùng dropout và một skip connection đơn giản nhằm cải thiện khả năng lan truyền gradient và giảm overfitting.

---

## 4. Thiết kế thực nghiệm

- Metric đánh giá: Accuracy, Precision (macro), Recall (macro), F1-score (macro)
- Confusion Matrix: để phân tích chi tiết lỗi nhầm giữa các lớp (đặc biệt các cặp 4–9, 3–5, 5–6, …)
- Thời gian: đo thời gian huấn luyện và thời gian dự đoán (tổng và trung bình/mẫu)
- Tài nguyên mô hình:
	- LR: số tham số = size(coef_) + size(intercept_), kích thước ≈ nbytes của trọng số + bias
	- MLP: tổng số tham số = sum(p.numel() for p in model.parameters()); kích thước tệp mlp.pth

Tập train/test được cố định bằng `random_state=42` (đối với LR) và stratify để giữ phân bố lớp.

---

## 5. Kết quả thực nghiệm

Lưu ý: Các chỉ số dưới đây được lấy từ output khi chạy notebook. Vui lòng chạy lại notebook để cập nhật con số chính xác trên môi trường của bạn.

### 5.1 Logistic Regression
- Thời gian huấn luyện (s): t_train = …
- Thời gian dự đoán (s): t_pred = …
- Thời gian dự đoán trung bình/mẫu (s): …
- Accuracy: …
- Precision (macro): …
- Recall (macro): …
- F1-score (macro): …
- Confusion Matrix (rút gọn, xem đầy đủ trong notebook):
	- Hàng: nhãn thật, Cột: nhãn dự đoán
	- …

### 5.2 MLP (PyTorch)
- Thời gian huấn luyện (s): training_time = …
- Thời gian dự đoán (s): testing_time = …
- Thời gian dự đoán trung bình/mẫu (s): …
- Accuracy: …
- Precision (macro): …
- Recall (macro): …
- F1-score (macro): …
- Confusion Matrix (rút gọn, xem đầy đủ/tới heatmap trong notebook):
	- Hàng: nhãn thật, Cột: nhãn dự đoán
	- …
- Số tham số mô hình: model_params = …
- Kích thước tệp lưu (mlp.pth): … MB

### 5.3 Bảng so sánh tổng hợp

| Mô hình | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Train time (s) | Predict time (s) | Params | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | … | … | … | … | … | … | n_params=… | ≈ … KB |
| MLP (PyTorch) | … | … | … | … | … | … | … | … MB |

---

## 6. Phân tích và thảo luận

1) Hiệu năng: 
- LR thường cho baseline tốt, huấn luyện nhanh, suy luận nhanh, mô hình nhỏ. Tuy nhiên có thể giới hạn trong việc học biên quyết định phi tuyến mạnh.
- MLP học được biểu diễn phi tuyến nên thường đạt độ chính xác cao hơn LR trên MNIST sau vài epoch, nhưng chi phí tính toán và kích thước mô hình lớn hơn.

2) Phân tích nhầm lẫn (Confusion Matrix):
- Các cặp dễ nhầm: 4–9, 3–5, 5–6, 7–9… do hình dạng nét tương tự hoặc nhiễu/viết xấu.
- Với MLP, các lỗi nhầm thường giảm so với LR, đặc biệt ở lớp có biên phức tạp.

3) Ảnh hưởng siêu tham số:
- LR: `C` (mức regularization) và `solver` có thể ảnh hưởng đáng kể; hiện tại dùng 'lbfgs' tổng quát, có thể thử 'saga' với l1/l2.
- MLP: số tầng, số neuron, dropout, lr, batch size, epochs đều quan trọng; tăng epochs có thể cải thiện thêm nhưng cần theo dõi overfitting.

4) Hạn chế:
- Chưa khai thác augmentation ảnh (random shift/rotation) hay normalization nâng cao.
- MLP chưa dùng early stopping hay scheduler.
- Chưa thực hiện hyperparameter tuning có hệ thống.

5) Gợi ý cải tiến:
- Thử chuẩn hóa z-score theo kênh (mặc dù dữ liệu dạng pixel đã scale).
- Thêm data augmentation nhẹ (shift/rotate) để tăng tính khái quát.
- Áp dụng early stopping, ReduceLROnPlateau cho MLP.
- Tối ưu siêu tham số (GridSearch/Optuna) cho cả LR và MLP.
- Thử CNN đơn giản (LeNet-like) làm baseline DL mạnh hơn cho dữ liệu ảnh.

---

## 7. Kết luận

Đồ án đã xây dựng hai mô hình cho nhận diện chữ số MNIST: Logistic Regression và MLP. Kết quả cho thấy MLP có tiềm năng đạt độ chính xác cao hơn nhờ khả năng học phi tuyến, trong khi Logistic Regression là baseline nhẹ, nhanh và dễ triển khai. Việc mở rộng với CNN/augmentation và tuning siêu tham số hứa hẹn nâng cao thêm hiệu năng.

---

## 8. Hướng dẫn chạy và tái hiện kết quả

1) Môi trường:
- Python 3.x; yêu cầu chính: numpy, pandas, scikit-learn, matplotlib, torch, tqdm, seaborn (nếu vẽ heatmap confusion matrix)

2) Chạy notebook:
- Mở và chạy tuần tự `Logistic_Regression.ipynb` để thu được chỉ số LR (bao gồm Confusion Matrix và số tham số/kích thước mô hình).
- Mở và chạy tuần tự `NeuralNetwork.ipynb` để huấn luyện MLP, đánh giá và lưu mô hình `mlp.pth`. Cell cuối notebook đã in tổng hợp thời gian, metric và kích thước mô hình.

3) Điền kết quả vào bảng (mục 5.3) nếu cần đưa vào báo cáo in giấy.

---

## 9. Tài liệu tham khảo

- Y. LeCun et al., “MNIST handwritten digit database”
- Scikit-learn: LogisticRegression — https://scikit-learn.org/
- PyTorch: https://pytorch.org/

