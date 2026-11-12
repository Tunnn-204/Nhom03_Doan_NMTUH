import os
import json

CSV_FILE_PATH = "UCI_Credit_Card.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. THIẾT LẬP THỰC NGHIỆM TỐC ĐỘ CAO (ĐỦ 10 SEEDS TRONG ~45 GIÂY)
# ==============================================================================
DATA_SPLIT_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 3                # Giảm từ 5 xuống 3 folds giúp giảm 40% thời gian chạy

# Nếu muốn chạy siêu tốc trong 15 giây (3 seeds, 10 trials), đổi ULTRA_FAST = True
ULTRA_FAST = False

SAMPLE_SIZE = 3000 if ULTRA_FAST else 4500
N_TRIALS = 10 if ULTRA_FAST else 15
SEEDS = [101, 102, 103] if ULTRA_FAST else [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
CONVERGENCE_THRESHOLD = 0.550

# ==============================================================================
# 2. MIỀN XÁC ĐỊNH CỦA BIẾN QUYẾT ĐỊNH x (Gọn nhẹ, hội tụ nhanh)
# ==============================================================================
SEARCH_SPACE = {
    "n_estimators": (50, 200),         # Giới hạn trần 200 cây để chạy cực nhanh
    "max_depth": (3, 8),               # Độ sâu 3-8 tránh cây quá lớn gây chậm
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_lambda": (1e-3, 10.0),
    "scale_pos_weight": (1.0, 4.0),
}

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
}

# ==============================================================================
# 3. CÁC RÀNG BUỘC KỸ THUẬT g_i(x) <= 0
# ==============================================================================
MAX_CV_RUNTIME_SEC = 5.0    # Ràng buộc thời gian cho 3-fold CV (giây)
MAX_OVERFIT_GAP = 0.08      # Ràng buộc chống quá khớp
MAX_MODEL_SIZE_MB = 3.0     # Ràng buộc dung lượng mô hình tối đa (MB)
PENALTY_WEIGHT = 0.50

# ==============================================================================
# 4. THAM SỐ KINH DOANH NGÂN HÀNG
# ==============================================================================
RECALL_AT_K_RATIO = 0.20
AVG_DEFAULT_LOSS_VND = 30_000_000
REVIEW_COST_VND = 400_000
PREVENTION_RATE = 0.80


def export_config_json():
    cfg_dict = {
        "csv_file_path": CSV_FILE_PATH,
        "data_split_seed": DATA_SPLIT_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "ultra_fast_mode": ULTRA_FAST,
        "sample_size": SAMPLE_SIZE,
        "n_trials_budget": N_TRIALS,
        "seeds": SEEDS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "search_space": SEARCH_SPACE,
        "default_params": DEFAULT_PARAMS,
        "constraints": {
            "max_cv_runtime_sec": MAX_CV_RUNTIME_SEC,
            "max_overfit_gap": MAX_OVERFIT_GAP,
            "max_model_size_mb": MAX_MODEL_SIZE_MB,
            "penalty_weight": PENALTY_WEIGHT,
        },
    }
    config_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)
    return config_path