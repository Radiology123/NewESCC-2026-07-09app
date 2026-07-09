import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# 0. 页面基础设置
# =======================
st.set_page_config(
    page_title="ESCC Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =======================
# 1. 文件路径
# 确保 RF.pkl、zscore_params.pkl 和 app.py 在同一目录
# =======================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

MODEL_PATH = BASE_DIR / "RF.pkl"
ZPARAMS_PATH = BASE_DIR / "zscore_params.pkl"

feature_names = ["Asparagine", "Choline", "Glutamate", "Sarcosine"]

# =======================
# 2. 加载模型与预处理参数
# =======================
@st.cache_resource
def load_model_and_params():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

    if not ZPARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing parameter file not found: {ZPARAMS_PATH.resolve()}"
        )

    model = joblib.load(MODEL_PATH)
    zparams = joblib.load(ZPARAMS_PATH)

    offset = float(zparams.get("offset", 0.0))

    mean = pd.Series(zparams["mean"], dtype=float)
    std = pd.Series(zparams["std"], dtype=float)

    return model, offset, mean, std


try:
    model, offset, mean, std = load_model_and_params()
except Exception as e:
    st.error(f"Failed to load the model or preprocessing parameters: {e}")
    st.stop()

# =======================
# 3. 页面标题
# =======================
st.title("ESCC Prediction System")

# =======================
# 4. 输入特征：显示在网页上方
# =======================
st.markdown("## Input Raw Metabolite Values")

with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Asparagine = st.number_input(
            "Asparagine",
            value=1.0,
            format="%.6f"
        )

    with col2:
        Choline = st.number_input(
            "Choline",
            value=1.0,
            format="%.6f"
        )

    with col3:
        Glutamate = st.number_input(
            "Glutamate",
            value=1.0,
            format="%.6f"
        )

    with col4:
        Sarcosine = st.number_input(
            "Sarcosine",
            value=1.0,
            format="%.6f"
        )

    submitted = st.form_submit_button("Generate Prediction")

raw_df = pd.DataFrame(
    [{
        "Asparagine": Asparagine,
        "Choline": Choline,
        "Glutamate": Glutamate,
        "Sarcosine": Sarcosine
    }]
)

# =======================
# 5. 预测
# =======================
if submitted:

    # 检查 mean/std 是否包含 4 个代谢物
    missing = [
        c for c in feature_names
        if c not in mean.index or c not in std.index
    ]

    if missing:
        st.error(
            "The zscore_params.pkl file does not contain the mean and standard "
            "deviation values for the following metabolites:\n\n"
            f"{missing}\n\n"
            "Please confirm that the feature names in the training dataset are "
            "identical to those used in this application."
        )
        st.stop()

    # 检查标准差是否为 0
    zero_std = [
        c for c in feature_names
        if std[c] == 0
    ]

    if zero_std:
        st.error(
            "The standard deviation of the following metabolites is 0, "
            "which prevents Z-score standardization:\n\n"
            f"{zero_std}"
        )
        st.stop()

    # 检查 log2 是否可计算
    min_allowed = -offset + 1e-12

    if (raw_df[feature_names] <= min_allowed).any().any():
        st.error(
            f"One or more input values are less than or equal to {-offset}, "
            "which prevents calculation of log2(x + offset).\n\n"
            f"Please ensure that each metabolite value satisfies x > {-offset} "
            f"(offset = {offset})."
        )
        st.stop()

    # 1）log2 transformation
    log2_df = np.log2(raw_df[feature_names].astype(float) + offset)

    # 2）Z-score standardization using training-cohort parameters
    z_df = (log2_df - mean[feature_names]) / std[feature_names]

    # 3）输入模型
    input_values = z_df[feature_names].values

    # 4）预测
    pred = int(model.predict(input_values)[0])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_values)[0]
    else:
        st.error(
            "The loaded model does not support predict_proba(). "
            "Please confirm that the model is a probabilistic classifier."
        )
        st.stop()

    prob_non_escc = probas[0]
    prob_escc = probas[1]

    # =======================
    # 6. 展示预测结果
    # =======================
    st.markdown("---")
    st.markdown("## Prediction Result")

    if pred == 1:
        st.markdown("### 🩺 Prediction Result: **ESCC**")
    else:
        st.markdown("### 🩺 Prediction Result: **Non-ESCC**")

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Non-ESCC Probability",
            value=f"{prob_non_escc:.4f}"
        )

    with col_result2:
        st.metric(
            label="ESCC Probability",
            value=f"{prob_escc:.4f}"
        )

    st.write(
        f"**Predicted probabilities:** "
        f"Non-ESCC (0) = {prob_non_escc:.4f}, "
        f"ESCC (1) = {prob_escc:.4f}"
    )

    # =======================
    # 7. 结果解释
    # =======================
    if pred == 1:
        st.info(
            f"The model generated an **ESCC-positive prediction**, with an "
            f"estimated ESCC probability of **{prob_escc * 100:.2f}%**. "
            "Further evaluation using endoscopy, histopathology, and relevant "
            "clinical information is recommended."
        )
    else:
        st.info(
            f"The model generated a **Non-ESCC prediction**, with an estimated "
            f"ESCC probability of **{prob_escc * 100:.2f}%**. "
            "The result should still be interpreted together with clinical risk "
            "factors and appropriate follow-up examinations."
        )

    # =======================
    # 8. 可视化预测概率
    # =======================
    st.markdown("## Predicted Probability Plot")

    fig, ax = plt.subplots(figsize=(5, 2.5))

    labels = ["Non-ESCC (0)", "ESCC (1)"]
    values = [prob_non_escc, prob_escc]

    ax.barh(
        labels,
        values,
        color=["#2E86C1", "#E74C3C"],
        height=0.6
    )

    ax.set_xlabel("Predicted probability")
    ax.set_xlim(0, 1)

    for i, v in enumerate(values):
        ax.text(
            min(v + 0.02, 0.95),
            i,
            f"{v:.3f}",
            va="center",
            fontweight="bold",
            fontsize=9
        )

    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    left_col, middle_col, right_col = st.columns([1, 2, 1])
    with middle_col:
        st.pyplot(fig)

    plt.close(fig)