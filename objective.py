import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from config import (
    CV_FOLDS, DATA_SPLIT_SEED, MAX_CV_RUNTIME_SEC,
    MAX_OVERFIT_GAP, MAX_MODEL_SIZE_MB, PENALTY_WEIGHT,
    RECALL_AT_K_RATIO, AVG_DEFAULT_LOSS_VND, REVIEW_COST_VND, PREVENTION_RATE
)


def compute_recall_at_k(y_true: np.ndarray, y_probs: np.ndarray, k_ratio: float = RECALL_AT_K_RATIO) -> float:
    k_count = max(1, int(len(y_true) * k_ratio))
    top_k_indices = np.argsort(y_probs)[::-1][:k_count]
    total_positives = max(1, int(np.sum(y_true)))
    return float(np.sum(y_true[top_k_indices]) / total_positives)


class CreditHPOObjective:
    def __init__(self, X_dev, y_dev, algorithm_name="Unknown", seed=42):
        self.X_dev = X_dev.values if hasattr(X_dev, "values") else X_dev
        self.y_dev = y_dev
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=DATA_SPLIT_SEED)
        self.splits = list(self.cv.split(self.X_dev, self.y_dev))
        self.trials_log = []
        self.best_feasible_score = -np.inf
        self.best_params = None
        self.total_model_fits = 0

    def evaluate(self, params: dict) -> float:
        trial_id = len(self.trials_log) + 1
        clean_params = {
            "n_estimators": int(round(params["n_estimators"])),
            "max_depth": int(round(params["max_depth"])),
            "learning_rate": float(params["learning_rate"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "reg_lambda": float(params["reg_lambda"]),
            "scale_pos_weight": float(params["scale_pos_weight"]),
        }

        start_time = time.perf_counter()
        train_praucs, val_praucs, val_briers, val_recalls = [], [], [], []
        last_model = None

        for train_idx, val_idx in self.splits:
            X_tr, y_tr = self.X_dev[train_idx], self.y_dev[train_idx]
            X_va, y_va = self.X_dev[val_idx], self.y_dev[val_idx]

            # Sử dụng tree_method="hist" giúp huấn luyện cực nhanh
            model = XGBClassifier(
                **clean_params,
                tree_method="hist",
                eval_metric="logloss",
                random_state=DATA_SPLIT_SEED,
                n_jobs=4
            )
            model.fit(X_tr, y_tr)
            self.total_model_fits += 1
            last_model = model

            prob_tr = model.predict_proba(X_tr)[:, 1]
            prob_va = model.predict_proba(X_va)[:, 1]

            train_praucs.append(average_precision_score(y_tr, prob_tr))
            val_praucs.append(average_precision_score(y_va, prob_va))
            val_briers.append(brier_score_loss(y_va, prob_va))
            val_recalls.append(compute_recall_at_k(y_va, prob_va, RECALL_AT_K_RATIO))

        runtime_sec = time.perf_counter() - start_time
        mean_train_prauc = float(np.mean(train_praucs))
        mean_val_prauc = float(np.mean(val_praucs))
        mean_val_brier = float(np.mean(val_briers))
        mean_val_recall = float(np.mean(val_recalls))
        overfit_gap = mean_train_prauc - mean_val_prauc

        # Chỉ lấy kích thước booster 1 lần ở fold cuối để tiết kiệm thời gian
        mean_size_mb = len(last_model.get_booster().save_raw()) / (1024 * 1024)

        g1_runtime = runtime_sec - MAX_CV_RUNTIME_SEC
        g2_overfit = overfit_gap - MAX_OVERFIT_GAP
        g3_size = mean_size_mb - MAX_MODEL_SIZE_MB

        total_violation = (
            max(0.0, g1_runtime / MAX_CV_RUNTIME_SEC)
            + max(0.0, g2_overfit / MAX_OVERFIT_GAP)
            + max(0.0, g3_size / MAX_MODEL_SIZE_MB)
        )
        is_feasible = (g1_runtime <= 0.0) and (g2_overfit <= 0.0) and (g3_size <= 0.0)
        penalized_obj = mean_val_prauc - PENALTY_WEIGHT * total_violation

        if is_feasible and mean_val_prauc > self.best_feasible_score:
            self.best_feasible_score = mean_val_prauc
            self.best_params = clean_params.copy()
        elif self.best_params is None:
            self.best_params = clean_params.copy()

        current_best = (
            self.best_feasible_score
            if self.best_feasible_score > -np.inf
            else mean_val_prauc
        )

        self.trials_log.append({
            "algorithm": self.algorithm_name,
            "seed": self.seed,
            "trial_id": trial_id,
            **clean_params,
            "val_pr_auc": mean_val_prauc,
            "val_brier": mean_val_brier,
            "val_recall_at_k": mean_val_recall,
            "train_pr_auc": mean_train_prauc,
            "overfit_gap": overfit_gap,
            "runtime_sec": runtime_sec,
            "model_size_mb": mean_size_mb,
            "g1_runtime_viol": max(0.0, g1_runtime),
            "g2_overfit_viol": max(0.0, g2_overfit),
            "g3_size_viol": max(0.0, g3_size),
            "total_violation": total_violation,
            "is_feasible": int(is_feasible),
            "penalized_objective": penalized_obj,
            "best_so_far": current_best,
        })

        return penalized_obj


def evaluate_on_holdout_test(best_params: dict, X_dev, y_dev, X_test, y_test) -> dict:
    X_dev_arr = X_dev.values if hasattr(X_dev, "values") else X_dev
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test

    model = XGBClassifier(
        **best_params,
        tree_method="hist",
        eval_metric="logloss",
        random_state=DATA_SPLIT_SEED,
        n_jobs=4
    )
    t0 = time.perf_counter()
    model.fit(X_dev_arr, y_dev)
    fit_time = time.perf_counter() - t0

    t_inf = time.perf_counter()
    probs = model.predict_proba(X_test_arr)[:, 1]
    latency_ms_per_1k = ((time.perf_counter() - t_inf) / len(X_test_arr)) * 1000.0 * 1000.0

    model_size_mb = len(model.get_booster().save_raw()) / (1024 * 1024)

    test_pr_auc = float(average_precision_score(y_test, probs))
    test_brier = float(brier_score_loss(y_test, probs))
    test_recall_k = compute_recall_at_k(y_test, probs, RECALL_AT_K_RATIO)

    k_count = max(1, int(len(y_test) * RECALL_AT_K_RATIO))
    top_k_indices = np.argsort(probs)[::-1][:k_count]
    preds_top_k = np.zeros_like(y_test)
    preds_top_k[top_k_indices] = 1

    tp = int(np.sum((preds_top_k == 1) & (y_test == 1)))
    fp = int(np.sum((preds_top_k == 1) & (y_test == 0)))

    prevented_loss_vnd = tp * PREVENTION_RATE * AVG_DEFAULT_LOSS_VND
    processing_cost_vnd = (tp + fp) * REVIEW_COST_VND
    net_benefit_vnd = prevented_loss_vnd - processing_cost_vnd

    return {
        "test_pr_auc": test_pr_auc,
        "test_brier": test_brier,
        "test_recall_at_20pct": test_recall_k,
        "fit_time_sec": fit_time,
        "latency_ms_per_1k": latency_ms_per_1k,
        "model_size_mb": model_size_mb,
        "tp_caught": tp,
        "fp_false_alarm": fp,
        "prevented_loss_m_vnd": prevented_loss_vnd / 1e6,
        "processing_cost_m_vnd": processing_cost_vnd / 1e6,
        "net_benefit_m_vnd": net_benefit_vnd / 1e6,
        "model": model,
        "test_probs": probs,
    }