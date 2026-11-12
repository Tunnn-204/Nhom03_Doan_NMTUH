import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import CSV_FILE_PATH, OUTPUT_DIR, DATA_SPLIT_SEED, TEST_SIZE

FEATURE_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

TARGET_NAME = "default.payment.next.month"


def ensure_csv_exists(csv_path: str = CSV_FILE_PATH):
    """
    Kiểm tra file UCI_Credit_Card.csv. Nếu chưa có trong thư mục,
    tự động tải bộ dữ liệu gốc 30.000 dòng từ OpenML và lưu thành UCI_Credit_Card.csv.
    """
    if os.path.exists(csv_path):
        return

    print(f"[INFO] Không tìm thấy '{csv_path}' tại thư mục gốc. Đang tải tự động bộ dữ liệu UCI (30.000 mẫu)...")
    try:
        dataset = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = dataset.frame.copy()
        raw_target = df.columns[-1]
        feature_cols = list(df.columns[:-1])
        rename_map = {old_col: new_col for old_col, new_col in zip(feature_cols, FEATURE_NAMES)}
        rename_map[raw_target] = TARGET_NAME
        df = df.rename(columns=rename_map)
        df.insert(0, "ID", np.arange(1, len(df) + 1))
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Đã tạo thành công file '{csv_path}' ({len(df)} dòng, {df.shape[1]} cột).")
    except Exception as e:
        print(f"[WARN] Lỗi kết nối mạng ({e}). Đang khởi tạo dữ liệu dự phòng...")
        X_arr, y_arr = make_classification(
            n_samples=30000, n_features=23, n_informative=15,
            n_redundant=5, weights=[0.7788, 0.2212], random_state=DATA_SPLIT_SEED
        )
        df = pd.DataFrame(X_arr, columns=FEATURE_NAMES)
        df.insert(0, "ID", np.arange(1, len(df) + 1))
        df[TARGET_NAME] = y_arr
        df.to_csv(csv_path, index=False)


def load_credit_data(csv_path: str = CSV_FILE_PATH, sample_size: int = None, export_eda: bool = True):
    """
    Đọc file UCI_Credit_Card.csv, làm sạch mã lỗi, xuất thống kê EDA và chia tập Train/Test.
    """
    ensure_csv_exists(csv_path)
    df = pd.read_csv(csv_path)

    # 1. Bỏ cột định danh ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 2. Tách nhãn mục tiêu y và ma trận đặc trưng X
    target_col = TARGET_NAME if TARGET_NAME in df.columns else df.columns[-1]
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col]).copy()

    if X.shape[1] == len(FEATURE_NAMES):
        X.columns = FEATURE_NAMES

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 3. Làm sạch giá trị ngoại lai theo chuẩn bộ dữ liệu UCI Credit Card
    if "EDUCATION" in X.columns:
        X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in X.columns:
        X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})

    # 4. Xuất bảng thống kê mô tả dữ liệu (EDA) phục vụ viết báo cáo
    if export_eda:
        eda_df = X.describe().T[["count", "mean", "std", "min", "50%", "max"]].copy()
        eda_df.index.name = "Feature"
        eda_df.to_csv(os.path.join(OUTPUT_DIR, "eda_summary.csv"))

    # 5. Lấy mẫu phân tầng (Stratified Subsampling) nếu có giới hạn sample_size
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=DATA_SPLIT_SEED
        )

    # 6. Chia cố định 80% Development (Train/Val) và 20% Hold-out Test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=DATA_SPLIT_SEED
    )

    # 7. Chuẩn hóa đặc trưng (Chỉ fit trên tập Dev để tuyệt đối chống Data Leakage)
    scaler = StandardScaler()
    X_dev_scaled = pd.DataFrame(scaler.fit_transform(X_dev), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_dev_scaled, X_test_scaled, y_dev, y_test, scaler, X_dev