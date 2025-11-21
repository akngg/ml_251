import time
import numpy as np
import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Đọc dữ liệu
data = pd.read_csv("data/train.csv")

X = data.drop("label", axis=1).values
y = data["label"].values

# Chuẩn hóa pixel về [0,1]
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Kích thước Train dataset: ", X_train.shape)
print("Kích thước Test dataset : ", X_test.shape)

# Khởi tạo mô hình Logistic Regression
logreg = LogisticRegression (
    # multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1
)

# Đo thời gian huấn luyện
time0 = time.time()
logreg.fit(X_train, y_train)
t_train = time.time() - time0

print("\n ===== Thời gian huấn luyện =====")
print(f"Thời gian huấn luyện: {t_train:.3f}s")


# Đo thời gian dự đoán
time1 = time.time()
y_pred = logreg.predict(X_test)
t_pred = time.time() - time1

print("\n ===== Thời gian dự đoán =====")
print(f"Thời gian dự đoán trên Test data  : {t_pred:.4f}s")
print(f"Thời gian dự đoán trung bình / mẫu: {t_pred/ len(y_test):.6f}s")


# Độ chính xác mô hình
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("\n ===== Độ chính xác mô hình =====")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision(macro): {prec:.4f}")
print(f"Recall(macro)   : {recall:.4f}")
print(f"F1-score(macro) : {f1:.4f}")


# Phân bố lỗi và độ nhầm lẫn (Confussion Matrix)
cm = confusion_matrix(y_test, y_pred)

print("\n ===== Confusion Matrix =====")
print("Hàng: nhãn thật, Cột: nhãn dự đoán")
print(cm)


# Tài nguyên và độ phức tạp
# Số tham số = số phần tử của ma trận trọng số + bias
n_params = logreg.coef_.size + logreg.intercept_.size


# Kích thước mô hình = số byte của trọng số + bias
model_size_bytes = logreg.coef_.nbytes + logreg.intercept_.nbytes
model_size_kb = model_size_bytes / 1024
model_size_mb = model_size_kb / 1024


print("\n ===== Tài nguyên và độ phức tạp =====")
print(f"Số tham số của mô hình: {n_params}")
print(f"Kích thước của mô hình: {model_size_kb:.2f} KB (~{model_size_mb:.2}MB)")