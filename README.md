# Nhận diện chữ số viết tay MNIST

## Cấu trúc thư mục

- `data/` : Chứa file train.csv, test.csv (tải từ Kaggle)
- `Logistic_Regression.ipynb` : Notebook Logistic Regression
- `NeuralNetwork.ipynb` : Notebook Neural Network
- `models/` : Chứa các file mô hình đã lưu (`logistic_regression_model.joblib`, `neural_network_model.pth`)
- `report.pdf` : Báo cáo bài tập lớn

## Yêu cầu môi trường

- Python == 3.11.0
- Các thư viện:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - torch
  - tqdm
  - joblib

Cài đặt nhanh bằng pip:
```bash
python -m pip install -r requirements.txt
```

## Hướng dẫn chạy

1. Chạy lần lượt các notebook:
   - `Logistic_Regression.ipynb`: Huấn luyện, đánh giá, lưu mô hình Logistic Regression.
   - `NeuralNetwork.ipynb`: Huấn luyện, đánh giá, lưu mô hình Neural Network.
3. Kết quả, confusion matrix, số liệu sẽ được in ra cuối mỗi notebook.
4. Có thể nạp lại mô hình đã lưu để dự đoán nhanh trên dữ liệu mới.
