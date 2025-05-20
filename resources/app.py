# -------------------------------------------------------------
# Streamlit app: Sickle‑Cell Pain‑Crisis Forecaster
# -------------------------------------------------------------
# Requirements: streamlit, pandas, numpy, torch, scikit‑learn, joblib
# -------------------------------------------------------------
import joblib, base64, pathlib, numpy as np, pandas as pd, streamlit as st, torch
from torch.utils.data import TensorDataset, DataLoader
from ode_torch import ODEModel
from sign_vector import sign_vector

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
FEATURES = [
    "Pain Intensity", "Pain Location_No Pain", "Oxygen Saturation (%)",
    "Body Temperature (°C)", "Pain Crises Frequency (past year)",
    "Respiratory Rate (bpm)", "Heart Rate (bpm)", "Acute Chest Syndrome",
    "Hospitalization History (past year)", "Dizziness", "Pain Location_Head",
    "Nausea", "Fever", "Jaundice", "Shortness of Breath",
    "Swelling", "Headache", "Fatigue"
]
BOOL_FEATURES = {"Pain Location_No Pain", "Pain Location_Head"}
LABELS = {
    "Pain Intensity": "Pain intensity (VAS 0‑10)",
    "Oxygen Saturation (%)": "SpO₂ (%)",
    "Body Temperature (°C)": "Body temperature (°C)",
    "Pain Crises Frequency (past year)": "Crises in past year",
    "Respiratory Rate (bpm)": "Respiratory rate (bpm)",
    "Heart Rate (bpm)": "Heart rate (bpm)",
    "Acute Chest Syndrome": "Hx acute chest syndrome (count)",
    "Hospitalization History (past year)": "Hospitalisations (past year)",
}

# -------------------------------------------------------------
# UI CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="SCD Pain‑Crisis Forecaster", layout="centered")

# ---- sidebar branding --------------------------------------
with st.sidebar:
    st.image("logo.png", caption="v1.0", use_container_width=True)
    st.header("Forecast settings")
    method = st.selectbox("Select engine", ["Mechanistic ODE", "Exponential shortcut"])
    horizons = st.multiselect("Prediction horizons (h)", [1, 3, 6, 12, 24, 48, 72], default=[1, 6, 12, 24])
    st.markdown("---")
    st.caption("Made with ❤️  &  PyTorch + Streamlit")

st.title("🩸 Sickle‑Cell Pain‑Crisis Risk Forecaster")
st.markdown("Enter current vital signs and symptom data; the app returns both a **class prediction** and **time‑resolved crisis probabilities**.")

# -------------------------------------------------------------
# LOAD ARTEFACTS (cache)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    base = pathlib.Path("models")
    ensemble = joblib.load(base / "ensemble_model.pkl")
    sign_vec = sign_vector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ode = ODEModel(sign_vec).to(device)
    ode.load_state_dict(torch.load(base / "ode_theta_torch.pt", map_location=device))
    ode.eval()
    return ensemble, ode, device

ensemble_model, ode_model, device = load_models()

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def survival_shortcut(p_cls: float, delta_t_h: float, tref_h: float = 24):
    hazard = -np.log(max(1e-8, 1 - p_cls)) / tref_h
    return 1 - np.exp(-hazard * delta_t_h)

@torch.no_grad()
def ode_probs(x_scaled: np.ndarray, horizon_h: float):
    tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    return ode_model(tensor, horizon_h)[0].cpu().numpy()

# -------------------------------------------------------------
# INPUT FORM  (with bin-aware widgets)
# -------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("1️⃣ Patient features")
    cols = st.columns(2)
    inputs = {}

    # --- categorical bins -----------------------------------
    body_temp_bins = ["Hypothermia (<35)", "Normal (35–37.5)", "Fever (>37.5)"]
    hr_bins        = ["Bradycardia (<60)", "Normal (60–100)", "Tachycardia (>100)"]
    rr_bins        = ["Bradypnea (<12)", "Normal (12–20)", "Tachypnea (>20)"]
    bin_to_code    = {0:0, 1:1, 2:2}   # ordinal encoding already used in training

    for i, feat in enumerate(FEATURES):
        c = cols[i % 2]
        # Boolean flags first
        if feat in BOOL_FEATURES:
            inputs[feat] = c.checkbox(feat.replace("_", " "), value=False)
        # Binned physiologic variables
        elif feat == "Body Temperature (°C)":
            sel = c.selectbox("Body temperature", body_temp_bins, index=1)
            inputs[feat] = float(bin_to_code[body_temp_bins.index(sel)])
        elif feat == "Heart Rate (bpm)":
            sel = c.selectbox("Heart rate", hr_bins, index=1)
            inputs[feat] = float(bin_to_code[hr_bins.index(sel)])
        elif feat == "Respiratory Rate (bpm)":
            sel = c.selectbox("Respiratory rate", rr_bins, index=1)
            inputs[feat] = float(bin_to_code[rr_bins.index(sel)])
        else:
            label = LABELS.get(feat, feat)
            if "Intensity" in feat:
                inputs[feat] = c.slider(label, 0.0, 10.0, 5.0, 0.1)
            elif "Oxygen" in feat:
                inputs[feat] = c.slider(label, 80.0, 100.0, 96.0, 0.1)
            else:
                inputs[feat] = c.number_input(label, value=0.0, step=0.1)

    submitted = st.form_submit_button("🔍 Run forecast")

if not submitted:
    st.stop()

# -------------------------------------------------------------
# Assemble numeric feature vector exactly as used in training
# -------------------------------------------------------------
row = np.array([inputs[f] for f in FEATURES], dtype=np.float32).reshape(1, -1)
row_df = pd.DataFrame(row, columns=FEATURES)

# ---------------- 2️⃣  CLASS PREDICTION (24 h)  ----------------
static_probs = ensemble_model.predict_proba(row_df)[0]
static_pred  = np.argmax(static_probs)
class_map = {0: "No crisis", 1: "Acute crisis", 2: "Chronic crisis"}

st.subheader("2️⃣ 24‑h Class prediction (ensemble)")
st.markdown(f"**Predicted class:** :blue[{class_map[static_pred]}]")
st.write(pd.Series(static_probs, index=["No", "Acute", "Chronic"]).round(3))

# ---------------- 3️⃣  TIME‑RESOLVED FORECAST  ----------------
engine_ode = method == "Mechanistic ODE"
results = {}
for h in horizons:
    if engine_ode:
        probs = ode_probs(row, float(h))
    else:
        probs = np.zeros(3, dtype=np.float32)
        for cls in (1, 2):
            probs[cls] = survival_shortcut(static_probs[cls], float(h))
        probs[0] = 1.0 - probs[1] - probs[2]
    results[h] = np.clip(probs, 0, 1)

st.subheader("3️⃣ Time‑resolved risk forecast")
res_df = pd.DataFrame(results, index=["No crisis", "Acute", "Chronic"]).T.round(5)
st.dataframe(res_df, use_container_width=True, height=200)
st.line_chart(res_df, x_label="Time Horizon (h)", 
              y_label= "Probability of crisis", 
              use_container_width=True, )

st.caption(f"*Engine:* {method}. Probabilities cumulative to horizon.")
