import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load_credit_data, FEATURE_NAMES
from config import (
    OUTPUT_DIR, DEFAULT_PARAMS, SEARCH_SPACE,
    MAX_CV_RUNTIME_SEC, MAX_OVERFIT_GAP, MAX_MODEL_SIZE_MB,
    AVG_DEFAULT_LOSS_VND, REVIEW_COST_VND, PREVENTION_RATE
)
from objective import evaluate_on_holdout_test

# ==============================================================================
# CẤU HÌNH TRANG & CSS GIAO DIỆN TRẮNG SÁNG (HIGH-CONTRAST LIGHT THEME)
# ==============================================================================
st.set_page_config(
    page_title="Demo Đồ án Tối ưu hóa - XGBoost Credit Risk HPO",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Ép toàn bộ nền trang và thanh bên về màu trắng / xám trắng sáng */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    /* Chữ tiêu đề và văn bản rõ nét cho máy chiếu */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0F172A;
    }
    /* Khung thẻ thông tin (Academic Card) */
    .light-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #1D4ED8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-box {
        background-color: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        color: #475569 !important;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #1D4ED8 !important;
    }
    .kpi-sub {
        font-size: 12px;
        color: #15803D !important;
        font-weight: 600;
    }
    /* Làm rõ các Tab */
    button[data-baseweb="tab"] {
        background-color: #F1F5F9 !important;
        border-radius: 6px 6px 0 0 !important;
        margin-right: 6px !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# THANH BÊN (SIDEBAR): THÔNG TIN ĐỒ ÁN & TRẠNG THÁI DỮ LIỆU
# ==============================================================================
with st.sidebar:
    st.markdown("### THÔNG TIN ĐỒ ÁN")
    st.markdown("""
    <div class="light-card" style="padding: 12px;">
        <b>Môn học:</b> Nhập môn Tối ưu hóa<br>
        <b>Đề tài:</b> Tối ưu hóa Siêu tham số XGBoost trong Dự báo Rủi ro Vỡ nợ Thẻ tín dụng<br>
        <b>Bộ dữ liệu:</b> UCI Credit Card (30.000 mẫu, 23 đặc trưng)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ THIẾT LẬP THỰC NGHIỆM")
    cfg_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            exp_cfg = json.load(f)
        st.write(f"- **Số Seeds lặp:** `{len(exp_cfg.get('seeds', []))}` seeds")
        st.write(f"- **Ngân sách (Budget):** `{exp_cfg.get('n_trials_budget', 15)}` trials/seed")
        st.write(f"- **Kiểm định chéo:** `{exp_cfg.get('cv_folds', 3)}-Fold Stratified CV`")
        st.write(f"- **Tỷ lệ Hold-out Test:** `{int(exp_cfg.get('test_size', 0.2)*100)}%` (Khóa độc lập)")
    else:
        st.info("Chưa có file cấu hình JSON. Hãy chạy `python run_experiments.py`.")

# ==============================================================================
# TIÊU ĐỀ CHÍNH (HEADER BANNER)
# ==============================================================================
st.markdown("""
<div style="background-color: #EFF6FF; border: 1.5px solid #93C5FD; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;">
    <h2 style="margin: 0; color: #1E3A8A !important;">🏦 HỆ THỐNG TỐI ƯU HÓA SIÊU THAM SỐ MÔ HÌNH XGBOOST</h2>
    <p style="margin: 6px 0 0 0; font-size: 16px; color: #1E40AF !important; font-weight: 500;">
        Bài toán: Dự báo Rủi ro Vỡ nợ Thẻ tín dụng Khách hàng Cá nhân (UCI Credit Card Default Dataset)
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Tab 1: Mô hình Toán học & Biến Quyết định",
    "Tab 2: Kết quả Thực nghiệm 10 Seeds & Biểu đồ",
    "Tab 3: Mô phỏng Chấm điểm Tín dụng & Chi phí"
])

# ==============================================================================
# TAB 1: MÔ HÌNH TOÁN HỌC (PROBLEM FORMULATION) & DỮ LIỆU
# ==============================================================================
with tab1:
    col_math1, col_math2 = st.columns([1.15, 1])

    with col_math1:
        st.subheader("1. Phát biểu Bài toán Tối ưu hóa Lồng (Min–Max)")
        st.markdown("""
        <div class="light-card">
            <b>Tầng ngoài (Outer Loop - Cực đại hóa PR-AUC trên Validation):</b><br>
            Tìm vector siêu tham số <b>x ∈ Ω</b> để cực đại hóa hàm mục tiêu có xét phạt ràng buộc:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"x^* = \arg\max_{x \in \Omega} f(x) = \frac{1}{K} \sum_{k=1}^{K} \text{PR-AUC}\left(\mathcal{M}(\Theta^*(x), \mathcal{D}_{\text{val}}^{(k)})\right)")

        st.markdown("""
        <div class="light-card">
            <b>Tầng trong (Inner Loop - Cực tiểu hóa hàm mất mát của XGBoost):</b><br>
            Ứng với mỗi cấu hình <b>x</b> cố định, tìm tập cây quyết định tối ưu:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\Theta^*(x) = \arg\min_{\Theta} \mathcal{L}_{\text{XGB}}(\Theta; \mathcal{D}_{\text{train}}^{(k)}, x)")

        st.subheader("2. Hệ thống 3 Ràng buộc Kỹ thuật g_i(x) ≤ 0")
        st.latex(r"\begin{aligned} g_1(x) &= T_{\text{cv}}(x) - T_{\max} \le 0 \quad (\text{Thời gian chạy CV } \le " + f"{MAX_CV_RUNTIME_SEC}" + r"\text{ giây}) \\ g_2(x) &= (\text{PR-AUC}_{\text{train}} - \text{PR-AUC}_{\text{val}}) - \Delta_{\max} \le 0 \quad (\text{Chống quá khớp } \le " + f"{MAX_OVERFIT_GAP}" + r") \\ g_3(x) &= \text{Size}(x) - S_{\max} \le 0 \quad (\text{Dung lượng mô hình } \le " + f"{MAX_MODEL_SIZE_MB}" + r"\text{ MB}) \end{aligned}")

    with col_math2:
        st.subheader("3. Không gian Biến Quyết định x ∈ Ω")
        var_rows = [
            {"Ký hiệu": "x1", "Siêu tham số": "n_estimators", "Kiểu biến": "Nguyên (Int)", "Biên dưới (x_min)": SEARCH_SPACE["n_estimators"][0], "Biên trên (x_max)": SEARCH_SPACE["n_estimators"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["n_estimators"]},
            {"Ký hiệu": "x2", "Siêu tham số": "max_depth", "Kiểu biến": "Nguyên (Int)", "Biên dưới (x_min)": SEARCH_SPACE["max_depth"][0], "Biên trên (x_max)": SEARCH_SPACE["max_depth"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["max_depth"]},
            {"Ký hiệu": "x3", "Siêu tham số": "learning_rate", "Kiểu biến": "Thực (Log-Float)", "Biên dưới (x_min)": SEARCH_SPACE["learning_rate"][0], "Biên trên (x_max)": SEARCH_SPACE["learning_rate"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["learning_rate"]},
            {"Ký hiệu": "x4", "Siêu tham số": "subsample", "Kiểu biến": "Thực (Float)", "Biên dưới (x_min)": SEARCH_SPACE["subsample"][0], "Biên trên (x_max)": SEARCH_SPACE["subsample"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["subsample"]},
            {"Ký hiệu": "x5", "Siêu tham số": "colsample_bytree", "Kiểu biến": "Thực (Float)", "Biên dưới (x_min)": SEARCH_SPACE["colsample_bytree"][0], "Biên trên (x_max)": SEARCH_SPACE["colsample_bytree"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["colsample_bytree"]},
            {"Ký hiệu": "x6", "Siêu tham số": "reg_lambda", "Kiểu biến": "Thực (Log-Float)", "Biên dưới (x_min)": SEARCH_SPACE["reg_lambda"][0], "Biên trên (x_max)": SEARCH_SPACE["reg_lambda"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["reg_lambda"]},
            {"Ký hiệu": "x7", "Siêu tham số": "scale_pos_weight", "Kiểu biến": "Thực (Float)", "Biên dưới (x_min)": SEARCH_SPACE["scale_pos_weight"][0], "Biên trên (x_max)": SEARCH_SPACE["scale_pos_weight"][1], "Mặc định (Baseline)": DEFAULT_PARAMS["scale_pos_weight"]},
        ]
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("4. Thống kê Mô tả Tập dữ liệu gốc (UCI_Credit_Card.csv)")
    eda_path = os.path.join(OUTPUT_DIR, "eda_summary.csv")
    if os.path.exists(eda_path):
        df_eda = pd.read_csv(eda_path)
        st.dataframe(df_eda.style.format(precision=2), use_container_width=True, hide_index=True)
    else:
        st.warning("Chưa có file `outputs/eda_summary.csv`. Hãy chạy `python run_experiments.py` trước.")

# ==============================================================================
# TAB 2: KẾT QUẢ THỰC NGHIỆM 10 SEEDS & BIỂU ĐỒ BÁO CÁO
# ==============================================================================
with tab2:
    report_path = os.path.join(OUTPUT_DIR, "final_report_table.csv")
    trials_path = os.path.join(OUTPUT_DIR, "trials_history.csv")

    if not os.path.exists(report_path):
        st.error("Chưa tìm thấy kết quả thực nghiệm! Vui lòng chạy `python run_experiments.py` trước.")
    else:
        df_report = pd.read_csv(report_path)
        df_trials = pd.read_csv(trials_path)

        # Lấy thuật toán tốt nhất theo Validation PR-AUC Mean
        best_row = df_report.sort_values("Val_PR_AUC_Mean", ascending=False).iloc[0]
        base_row = df_report[df_report["Algorithm"] == "Default_XGB"].iloc[0]
        gain_prauc = (best_row["Test_PR_AUC_Mean"] - base_row["Test_PR_AUC_Mean"]) * 100
        gain_vnd = best_row["Net_Benefit_M_VND"] - base_row["Net_Benefit_M_VND"]

        # 4 Thẻ KPI nổi bật trên nền sáng
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">THUẬT TOÁN THẮNG CUỘC (VALIDATION)</div>
                <div class="kpi-value">{best_row['Algorithm']}</div>
                <div class="kpi-sub">Đánh giá qua 10 Seeds độc lập</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            val_str = best_row["Val_PR_AUC_Mean_Std"] if "Val_PR_AUC_Mean_Std" in best_row else f"{best_row['Val_PR_AUC_Mean']:.4f}"
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">VALIDATION PR-AUC (MEAN ± STD)</div>
                <div class="kpi-value">{val_str}</div>
                <div class="kpi-sub">Best: {best_row['Val_PR_AUC_Best']:.4f} | Worst: {best_row['Val_PR_AUC_Worst']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">HOLD-OUT TEST PR-AUC</div>
                <div class="kpi-value">{best_row['Test_PR_AUC_Mean']:.4f}</div>
                <div class="kpi-sub">+{gain_prauc:.2f}% so với Default XGB</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">LỢI ÍCH RÒNG TRÊN TẬP TEST</div>
                <div class="kpi-value">{best_row['Net_Benefit_M_VND']:,.1f} Tr.đ</div>
                <div class="kpi-sub">+{gain_vnd:,.1f} Triệu VNĐ so với Baseline</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("1. Bảng Tổng hợp Hệ thống Chỉ số Đánh giá (Qua 10 Seeds)")
        st.dataframe(df_report, use_container_width=True, hide_index=True)

        st.subheader("2. Hệ thống Biểu đồ Phân tích Thực nghiệm")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            img1 = os.path.join(OUTPUT_DIR, "1_convergence_curve.png")
            if os.path.exists(img1):
                st.image(img1, caption="Hình 1: Đường cong hội tụ Best-so-far PR-AUC (Mean ± 1 Std)", use_container_width=True)
        with r1c2:
            img2 = os.path.join(OUTPUT_DIR, "2_stability_boxplot.png")
            if os.path.exists(img2):
                st.image(img2, caption="Hình 2: Phân phối độ ổn định qua 10 Seeds (Boxplot)", use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            img3 = os.path.join(OUTPUT_DIR, "3_business_net_benefit.png")
            if os.path.exists(img3):
                st.image(img3, caption="Hình 3: Lợi ích kinh doanh ròng trên tập Hold-out Test (Triệu VNĐ)", use_container_width=True)
        with r2c2:
            img5 = os.path.join(OUTPUT_DIR, "5_constraint_tradeoff.png")
            if os.path.exists(img5):
                st.image(img5, caption="Hình 5: Đánh đổi chất lượng nghiệm và ràng buộc chống quá khớp", use_container_width=True)

        img4 = os.path.join(OUTPUT_DIR, "4_scalability_plot.png")
        if os.path.exists(img4):
            _, mid_col, _ = st.columns([0.2, 0.6, 0.2])
            with mid_col:
                st.image(img4, caption="Hình 4: Khả năng mở rộng (Scalability) theo quy mô mẫu dữ liệu N", use_container_width=True)

        st.markdown("---")
        st.subheader("3. Tra cứu Nhật ký Trials & Kiểm tra Ràng buộc g_i(x) ≤ 0")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_alg = st.selectbox("Chọn thuật toán tối ưu:", df_trials["algorithm"].unique())
        with col_f2:
            avail_seeds = sorted(df_trials[df_trials["algorithm"] == selected_alg]["seed"].unique())
            selected_seed = st.selectbox("Chọn Seed chạy:", avail_seeds)

        filtered_trials = df_trials[
            (df_trials["algorithm"] == selected_alg) & (df_trials["seed"] == selected_seed)
        ]
        st.dataframe(filtered_trials, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: MÔ PHỎNG CHẤM ĐIỂM TÍN DỤNG & QUYẾT ĐỊNH KINH DOANH
# ==============================================================================
with tab3:
    st.subheader("Mô phỏng Trực tiếp: Default XGBoost (Baseline) vs TPE Optimized XGBoost")
    st.markdown("""
    <div class="light-card">
        Nhập thông số hồ sơ khách hàng bên dưới để quan sát sự khác biệt về <b>xác suất dự báo vỡ nợ</b> và <b>quyết định kinh doanh</b> giữa mô hình cấu hình mặc định và mô hình đã được tối ưu hóa siêu tham số bằng <b>TPE (Optuna)</b>.
    </div>
    """, unsafe_allow_html=True)

    # Tải dữ liệu an toàn (tương thích với mọi phiên bản hàm load_credit_data)
    data_tuple = load_credit_data(sample_size=4000)
    X_dev, X_test, y_dev, y_test, scaler = data_tuple[:5]

    runs_path = os.path.join(OUTPUT_DIR, "seed_runs_summary.csv")
    if os.path.exists(runs_path):
        df_runs = pd.read_csv(runs_path)
        tpe_sub = df_runs[df_runs["algorithm"] == "TPE_Optuna"].sort_values("best_val_pr_auc", ascending=False)
        best_tpe_params = json.loads(tpe_sub.iloc[0]["best_params_json"]) if len(tpe_sub) > 0 else DEFAULT_PARAMS
    else:
        best_tpe_params = {
            "n_estimators": 160, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 1.8, "scale_pos_weight": 2.4
        }

    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        limit_bal = st.number_input("1. Hạn mức tín dụng - LIMIT_BAL (NT$)", 10000, 1000000, 60000, step=10000)
        age = st.slider("2. Độ tuổi khách hàng - AGE", 21, 75, 31)
    with c_in2:
        pay_0 = st.selectbox("3. Trạng thái trả nợ tháng gần nhất (PAY_0)", [-1, 0, 1, 2, 3, 4], index=3, help="2: Trễ hạn 2 tháng; 0: Đúng hạn")
        pay_2 = st.selectbox("4. Trạng thái trả nợ 2 tháng trước (PAY_2)", [-1, 0, 1, 2, 3, 4], index=2)
    with c_in3:
        bill_amt1 = st.number_input("5. Dư nợ sao kê gần nhất - BILL_AMT1", 0, 500000, 58000, step=5000)
        pay_amt1 = st.number_input("6. Số tiền thực trả kỳ gần nhất - PAY_AMT1", 0, 500000, 1000, step=500)

    if st.button("CHẠY MÔ PHỎNG SO SÁNH & TÍNH CHI PHÍ RỦI RO", type="primary", use_container_width=True):
        res_base = evaluate_on_holdout_test(DEFAULT_PARAMS, X_dev, y_dev, X_test, y_test)
        res_opt = evaluate_on_holdout_test(best_tpe_params, X_dev, y_dev, X_test, y_test)

        raw_vec = np.zeros((1, len(FEATURE_NAMES)))
        raw_vec[0, 0] = limit_bal
        raw_vec[0, 1] = 2
        raw_vec[0, 2] = 2
        raw_vec[0, 3] = 1
        raw_vec[0, 4] = age
        raw_vec[0, 5] = pay_0
        raw_vec[0, 6] = pay_2
        raw_vec[0, 11] = bill_amt1
        raw_vec[0, 17] = pay_amt1

        scaled_vec = pd.DataFrame(scaler.transform(raw_vec), columns=FEATURE_NAMES)
        prob_base = float(res_base["model"].predict_proba(scaled_vec)[0, 1])
        prob_opt = float(res_opt["model"].predict_proba(scaled_vec)[0, 1])

        expected_loss_vnd = prob_opt * AVG_DEFAULT_LOSS_VND
        net_saving_vnd = (prob_opt * PREVENTION_RATE * AVG_DEFAULT_LOSS_VND) - REVIEW_COST_VND

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="kpi-box" style="border-top: 4px solid #64748B;">
                <div class="kpi-title">MÔ HÌNH BASELINE (DEFAULT XGBOOST)</div>
                <div class="kpi-value" style="color: #334155 !important;">Xác suất vỡ nợ: {prob_base*100:.2f}%</div>
                <div class="kpi-sub" style="color: #475569 !important;">Test PR-AUC: {res_base['test_pr_auc']:.4f} | Brier: {res_base['test_brier']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Bộ siêu tham số mặc định (chưa tối ưu):")
            st.dataframe(pd.DataFrame([DEFAULT_PARAMS]), use_container_width=True, hide_index=True)

        with col_res2:
            st.markdown(f"""
            <div class="kpi-box" style="border-top: 4px solid #1D4ED8; background-color: #F8FAFC;">
                <div class="kpi-title">MÔ HÌNH TỐI ƯU HÓA (TPE OPTUNA XGBOOST)</div>
                <div class="kpi-value" style="color: #1D4ED8 !important;">Xác suất vỡ nợ: {prob_opt*100:.2f}% ({(prob_opt-prob_base)*100:+.2f}%)</div>
                <div class="kpi-sub">Test PR-AUC: {res_opt['test_pr_auc']:.4f} | Brier: {res_opt['test_brier']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Bộ siêu tham số tối ưu x* tìm được bởi TPE:")
            st.dataframe(pd.DataFrame([best_tpe_params]), use_container_width=True, hide_index=True)

        # Khung kết luận kinh doanh
        decision_color = "#DC2626" if net_saving_vnd > 0 else "#16A34A"
        decision_bg = "#FEF2F2" if net_saving_vnd > 0 else "#F0FDF4"
        decision_text = (
            "⚠️ CẢNH BÁO RỦI RO VỠ NỢ CAO — Đưa vào danh sách thẩm định & thu hồi nợ sớm!"
            if net_saving_vnd > 0
            else "KHÁCH HÀNG AN TOÀN — Duyệt tự động, không tốn chi phí thẩm định."
        )

        st.markdown(f"""
        <div style="background-color: {decision_bg}; border: 2px solid {decision_color}; border-radius: 10px; padding: 18px; margin-top: 12px;">
            <h4 style="margin: 0 0 8px 0; color: {decision_color} !important;">{decision_text}</h4>
            <ul style="margin: 0; font-size: 15px; color: #0F172A !important;">
                <li><b>Tổn thất rủi ro kỳ vọng (nếu bỏ sót):</b> {expected_loss_vnd:,.0f} VNĐ</li>
                <li><b>Chi phí nhân sự thẩm định / nhắc nợ:</b> {REVIEW_COST_VND:,.0f} VNĐ</li>
                <li><b>Lợi ích kinh doanh ròng kỳ vọng khi can thiệp:</b> <b>{net_saving_vnd:,.0f} VNĐ</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)