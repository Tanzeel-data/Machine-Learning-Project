import streamlit as st
import numpy as np
import joblib
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="🩺",
    layout="wide"
)

# ── Load model, scaler, features ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("rf_best_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    with open("classification_report.json") as f:
        report = json.load(f)
    return model, scaler, features, report

model, scaler, FEATURES, report = load_artifacts()

LABEL_MAP = {1: "🟢 Normal", 2: "🟡 Suspect", 3: "🔴 Pathologic"}
LABEL_COLOR = {1: "success", 2: "warning", 3: "error"}

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🩺 Fetal Health Classification")
st.markdown(
    "**Cardiotocography (CTG) Dataset** — Predict fetal state from FHR & UC signals.  \n"
    "Model: *Random Forest (tuned)* · Classes: Normal / Suspect / Pathologic"
)
st.divider()

# ── Sidebar: Model Performance ────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    if "Random Forest" in report:
        rf = report["Random Forest"]
        st.metric("Accuracy",  f"{rf['accuracy']:.2%}")
        st.metric("F1 (macro)", f"{rf['macro avg']['f1-score']:.2%}")
    st.markdown("---")
    st.caption("Dataset: UCI CTG (2126 samples, 17 features)")

# ── Input panel ───────────────────────────────────────────────────────────────
st.subheader("📝 Enter CTG Signal Values")

# Feature metadata: (label, min, max, default, step)
FEATURE_META = {
    "LB":       ("FHR Baseline (bpm)",                  50,  200, 120,  1.0),
    "AC":       ("Accelerations / second",                0,  1.0, 0.003, 0.001),
    "FM":       ("Fetal Movements / second",              0,  1.0, 0.0,  0.001),
    "UC":       ("Uterine Contractions / second",         0,  1.0, 0.004, 0.001),
    "ASTV":     ("% Abnormal Short-Term Variability",     0,  100, 14.0, 1.0),
    "MSTV":     ("Mean Short-Term Variability",           0,  20,  1.4,  0.1),
    "ALTV":     ("% Abnormal Long-Term Variability",      0,  100, 0.0,  1.0),
    "MLTV":     ("Mean Long-Term Variability",            0,  100, 8.0,  0.1),
    "Width":    ("FHR Histogram Width",                   0,  300, 70.0, 1.0),
    "Min":      ("FHR Histogram Min",                    50,  200, 62.0, 1.0),
    "Max":      ("FHR Histogram Max",                    50,  300, 126.0,1.0),
    "Nmax":     ("Number of Histogram Peaks",             0,  30,  4.0,  1.0),
    "Nzeros":   ("Number of Histogram Zeros",             0,  20,  0.0,  1.0),
    "Mode":     ("FHR Histogram Mode",                   50,  200, 120.0,1.0),
    "Mean":     ("FHR Histogram Mean",                   50,  200, 137.0,1.0),
    "Variance": ("FHR Variance",                          0,  500, 24.0, 1.0),
    "Tendency": ("FHR Trend",                            -1,   1,  0.0,  1.0),
}

cols = st.columns(3)
values = {}
for i, feat in enumerate(FEATURES):
    label, fmin, fmax, fdef, fstep = FEATURE_META[feat]
    with cols[i % 3]:
        values[feat] = st.number_input(
            label=f"**{feat}** — {label}",
            min_value=float(fmin),
            max_value=float(fmax),
            value=float(fdef),
            step=float(fstep),
            key=feat
        )

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Fetal Health", use_container_width=True, type="primary"):
    input_arr = np.array([[values[f] for f in FEATURES]])
    input_sc  = scaler.transform(input_arr)

    prediction   = model.predict(input_sc)[0]
    probabilities = model.predict_proba(input_sc)[0]

    label    = LABEL_MAP[prediction]
    color_fn = getattr(st, LABEL_COLOR[prediction])
    color_fn(f"### Prediction: {label}")

    st.markdown("#### Class Probabilities")
    prob_cols = st.columns(3)
    class_names = ["Normal", "Suspect", "Pathologic"]
    for j, (cls, prob) in enumerate(zip(class_names, probabilities)):
        prob_cols[j].metric(cls, f"{prob:.1%}")

    with st.expander("ℹ️ What does this mean?"):
        descriptions = {
            1: "**Normal**: Fetal heart rate pattern appears healthy. Routine monitoring is recommended.",
            2: "**Suspect**: Some abnormal patterns detected. Clinical review is advised.",
            3: "**Pathologic**: Significant abnormalities detected. Immediate medical evaluation is required."
        }
        st.markdown(descriptions[prediction])

st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")
