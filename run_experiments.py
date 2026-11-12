import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    OUTPUT_DIR, SAMPLE_SIZE, N_TRIALS, SEEDS,
    CONVERGENCE_THRESHOLD, export_config_json
)
from data_loader import load_credit_data
from objective import CreditHPOObjective, evaluate_on_holdout_test
from optimizers import (
    run_default_baseline, run_grid_search, run_random_search,
    run_tpe_optuna, run_genetic_algorithm
)
sns.set_theme(style="whitegrid")
def compute_evals_to_threshold(trials_log: list, threshold: float) -> int:
    for row in trials_log:
        if row["is_feasible"] == 1 and row["val_pr_auc"] >= threshold:
            return int(row["trial_id"])
    return len(trials_log)
def run_all_experiments():
    t_global = time.perf_counter()
    print("CHẠY THỰC NGHIỆM TỐC ĐỘ CAO - TỐI ƯU HÓA SIÊU THAM SỐ XGBOOST")
    cfg_path = export_config_json()
    X_dev, X_test, y_dev, y_test, _, _ = load_credit_data(sample_size=SAMPLE_SIZE, export_eda=True)
    base_data_ram_mb = (X_dev.memory_usage().sum() + X_test.memory_usage().sum()) / (1024 * 1024)
    print(f"[1/4] Đã tải dữ liệu: Dev = {len(X_dev)} mẫu | Test = {len(X_test)} mẫu.")
    all_trials_records = []
    seed_level_summaries = []

    deterministic_methods = [
        ("Default_XGB", lambda obj: run_default_baseline(obj)),
        ("Grid_Search", lambda obj: run_grid_search(obj, N_TRIALS)),
    ]

    for alg_name, runner in deterministic_methods:
        t0 = time.perf_counter()
        obj = CreditHPOObjective(X_dev, y_dev, algorithm_name=alg_name, seed=0)
        best_params, logs = runner(obj)
        total_runtime = time.perf_counter() - t0

        all_trials_records.extend(logs)
        test_res = evaluate_on_holdout_test(best_params, X_dev, y_dev, X_test, y_test)
        feas_rate = float(np.mean([r["is_feasible"] for r in logs]) * 100.0)
        mean_viol = float(np.mean([r["total_violation"] for r in logs]))
        evals_thresh = compute_evals_to_threshold(logs, CONVERGENCE_THRESHOLD)
        peak_ram_est = base_data_ram_mb + max(r["model_size_mb"] for r in logs) * 2.5

        seed_level_summaries.append({
            "algorithm": alg_name,
            "seed": 0,
            "best_val_pr_auc": obj.best_feasible_score,
            "feasibility_rate_pct": feas_rate,
            "mean_constraint_violation": mean_viol,
            "evals_to_threshold": evals_thresh,
            "total_runtime_sec": total_runtime,
            "model_fits": obj.total_model_fits,
            "peak_ram_mb": peak_ram_est,
            **{k: v for k, v in test_res.items() if k not in ["model", "test_probs"]},
            "best_params_json": json.dumps(best_params),
        })
        print(f"  -> {alg_name:<18} | Val PR-AUC = {obj.best_feasible_score:.4f} | Time = {total_runtime:.2f}s")

    stochastic_methods = [
        ("Random_Search", run_random_search),
        ("TPE_Optuna", run_tpe_optuna),
        ("Genetic_Algorithm", run_genetic_algorithm),
    ]

    for alg_name, func in stochastic_methods:
        t_alg = time.perf_counter()
        print(f"  -> Đang chạy {alg_name} ({len(SEEDS)} seeds x {N_TRIALS} trials)...", end=" ", flush=True)
        for seed in SEEDS:
            t0 = time.perf_counter()
            obj = CreditHPOObjective(X_dev, y_dev, algorithm_name=alg_name, seed=seed)
            best_params, logs = func(obj, N_TRIALS, seed)
            total_runtime = time.perf_counter() - t0

            all_trials_records.extend(logs)
            test_res = evaluate_on_holdout_test(best_params, X_dev, y_dev, X_test, y_test)
            feas_rate = float(np.mean([r["is_feasible"] for r in logs]) * 100.0)
            mean_viol = float(np.mean([r["total_violation"] for r in logs]))
            evals_thresh = compute_evals_to_threshold(logs, CONVERGENCE_THRESHOLD)
            peak_ram_est = base_data_ram_mb + max(r["model_size_mb"] for r in logs) * 2.5

            seed_level_summaries.append({
                "algorithm": alg_name,
                "seed": seed,
                "best_val_pr_auc": obj.best_feasible_score,
                "feasibility_rate_pct": feas_rate,
                "mean_constraint_violation": mean_viol,
                "evals_to_threshold": evals_thresh,
                "total_runtime_sec": total_runtime,
                "model_fits": obj.total_model_fits,
                "peak_ram_mb": peak_ram_est,
                **{k: v for k, v in test_res.items() if k not in ["model", "test_probs"]},
                "best_params_json": json.dumps(best_params),
            })
        print(f"Hoàn tất trong {time.perf_counter() - t_alg:.2f}s!")

    df_trials = pd.DataFrame(all_trials_records)
    df_runs = pd.DataFrame(seed_level_summaries)
    df_trials.to_csv(os.path.join(OUTPUT_DIR, "trials_history.csv"), index=False)
    df_runs.to_csv(os.path.join(OUTPUT_DIR, "seed_runs_summary.csv"), index=False)

    agg_rows = []
    for alg, grp in df_runs.groupby("algorithm", sort=False):
        q75, q25 = np.percentile(grp["best_val_pr_auc"], [75, 25])
        mean_val = grp["best_val_pr_auc"].mean()
        std_val = grp["best_val_pr_auc"].std(ddof=0)

        agg_rows.append({
            "Algorithm": alg,
            "Val_PR_AUC_Mean_Std": f"{mean_val:.4f} ± {std_val:.4f}",
            "Val_PR_AUC_Mean": mean_val,
            "Val_PR_AUC_Std": std_val,
            "Val_PR_AUC_Median": grp["best_val_pr_auc"].median(),
            "Val_PR_AUC_IQR": q75 - q25,
            "Val_PR_AUC_Best": grp["best_val_pr_auc"].max(),
            "Val_PR_AUC_Worst": grp["best_val_pr_auc"].min(),
            "Test_PR_AUC_Mean": grp["test_pr_auc"].mean(),
            "Test_Brier_Mean": grp["test_brier"].mean(),
            "Test_Recall@20%_Mean": grp["test_recall_at_20pct"].mean(),
            "Feasibility_Rate_%": grp["feasibility_rate_pct"].mean(),
            "Constraint_Violation_Mean": grp["mean_constraint_violation"].mean(),
            "Evals_to_Threshold": grp["evals_to_threshold"].mean(),
            "Runtime_Sec_Mean": grp["total_runtime_sec"].mean(),
            "Model_Fits_Mean": grp["model_fits"].mean(),
            "Peak_RAM_MB_Mean": grp["peak_ram_mb"].mean(),
            "Latency_ms_per_1k": grp["latency_ms_per_1k"].mean(),
            "Model_Size_MB": grp["model_size_mb"].mean(),
            "Prevented_Loss_M_VND": grp["prevented_loss_m_vnd"].mean(),
            "Processing_Cost_M_VND": grp["processing_cost_m_vnd"].mean(),
            "Net_Benefit_M_VND": grp["net_benefit_m_vnd"].mean(),
        })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "final_report_table.csv"), index=False)
    print("\n[2/4] BẢNG TỔNG HỢP KẾT QUẢ:")
    cols_show = [
        "Algorithm", "Val_PR_AUC_Mean_Std", "Test_PR_AUC_Mean",
        "Feasibility_Rate_%", "Runtime_Sec_Mean", "Net_Benefit_M_VND"
    ]
    print(df_agg[cols_show].to_string(index=False))

    print("\n[3/4] Đang đo khả năng mở rộng (Scalability)...")
    scale_records = []
    for frac in [0.25, 0.50, 1.00]:
        n_sub = int(len(X_dev) * frac)
        X_sub, y_sub = X_dev.iloc[:n_sub], y_dev[:n_sub]
        t_start = time.perf_counter()
        obj_scale = CreditHPOObjective(X_sub, y_sub, algorithm_name="TPE_Scale", seed=SEEDS[0])
        run_tpe_optuna(obj_scale, max(8, N_TRIALS // 2), SEEDS[0])
        scale_records.append({
            "data_fraction": f"{int(frac * 100)}%",
            "n_samples": n_sub,
            "best_val_pr_auc": obj_scale.best_feasible_score,
            "runtime_sec": time.perf_counter() - t_start,
        })
    df_scale = pd.DataFrame(scale_records)
    df_scale.to_csv(os.path.join(OUTPUT_DIR, "scalability_results.csv"), index=False)

    print("[4/4] Đang xuất 5 biểu đồ PNG...")
    generate_all_plots(df_trials, df_runs, df_scale)
    print(f"\nHOÀN TẤT TOÀN BỘ TRONG {time.perf_counter() - t_global:.1f} GIÂY!")


def generate_all_plots(df_trials: pd.DataFrame, df_runs: pd.DataFrame, df_scale: pd.DataFrame):
    stoch_algs = ["Random_Search", "TPE_Optuna", "Genetic_Algorithm"]
    df_stoch_trials = df_trials[df_trials["algorithm"].isin(stoch_algs)]

    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=df_stoch_trials, x="trial_id", y="best_so_far",
        hue="algorithm", errorbar="sd", linewidth=2.2
    )
    plt.title("Đường cong hội tụ Best-so-far Validation PR-AUC (Mean ± 1 Std)")
    plt.xlabel("Số lần đánh giá hàm mục tiêu (Trials)")
    plt.ylabel("Best-so-far Validation PR-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_convergence_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 5))
    df_stoch_runs = df_runs[df_runs["algorithm"].isin(stoch_algs)]
    sns.boxplot(
        data=df_stoch_runs, x="algorithm", y="best_val_pr_auc",
        hue="algorithm", width=0.45, palette="Set2", legend=False
    )
    sns.stripplot(
        data=df_stoch_runs, x="algorithm", y="best_val_pr_auc",
        color="black", alpha=0.65, size=6
    )
    plt.title("Phân phối chất lượng nghiệm cuối cùng qua các Seeds (Boxplot)")
    plt.xlabel("Thuật toán tối ưu hóa")
    plt.ylabel("Best Validation PR-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_stability_boxplot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    sns.barplot(
        data=df_runs, x="algorithm", y="net_benefit_m_vnd",
        hue="algorithm", errorbar="sd", palette="viridis", legend=False
    )
    plt.title("Lợi ích kinh doanh ròng trên tập Hold-out Test (Triệu VNĐ)")
    plt.xlabel("Phương pháp")
    plt.ylabel("Lợi ích ròng (Triệu VNĐ)")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_business_net_benefit.png"), dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(df_scale["data_fraction"], df_scale["runtime_sec"], "o-", color="tab:blue", linewidth=2.2)
    ax2.plot(df_scale["data_fraction"], df_scale["best_val_pr_auc"], "s--", color="tab:orange", linewidth=2.2)
    ax1.set_xlabel("Quy mô dữ liệu huấn luyện (% tập Development)")
    ax1.set_ylabel("Thời gian chạy (Giây)", color="tab:blue")
    ax2.set_ylabel("Best Validation PR-AUC", color="tab:orange")
    plt.title("Đánh giá khả năng mở rộng (Scalability theo quy mô mẫu N)")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_scalability_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    sns.scatterplot(
        data=df_stoch_trials, x="overfit_gap", y="val_pr_auc",
        hue="algorithm", style="is_feasible", alpha=0.75, s=60
    )
    plt.axvline(0.08, color="red", linestyle="--", linewidth=1.8, label="Ngưỡng Overfit (0.08)")
    plt.title("Đánh đổi giữa Chất lượng nghiệm và Ràng buộc Chống quá khớp")
    plt.xlabel("Độ lệch Overfitting (Train PR-AUC - Val PR-AUC)")
    plt.ylabel("Validation PR-AUC")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_constraint_tradeoff.png"), dpi=300)
    plt.close()
if __name__ == "__main__":
    run_all_experiments()