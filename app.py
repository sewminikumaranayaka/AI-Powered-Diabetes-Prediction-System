import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# ---------------- MODERN UI CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Nunito+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Nunito Sans', sans-serif;
    background: #0d172a;
    color: #e8f0fe;
}

/* Keep header elements visible, remove black stripe background */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,210,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,255,0.055) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Ambient glow orbs ── */
.stApp::after {
    content: '';
    position: fixed;
    width: 700px;
    height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,136,255,0.11) 0%, transparent 70%);
    top: -200px;
    right: -200px;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
    padding: 2.5rem 3rem 3rem;
    max-width: 1200px;
}

/* ── Typography ── */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em;
    word-spacing: 0.22em;
    text-align: left !important;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 30%, #00d2ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
    padding-bottom: 0;
}

h2 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #e8f0fe !important;
}

h3 {
    font-family: 'Nunito Sans', sans-serif !important;
    font-weight: 300 !important;
    font-size: 1.1rem !important;
    color: #60a5fa !important;
    text-align: left !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.25rem !important;
}

h4 { font-family: 'Outfit', sans-serif !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090f1e !important;
    border-right: 1px solid rgba(0,210,255,0.12) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #00d2ff !important;
    -webkit-text-fill-color: #00d2ff !important;
    background: none !important;
    margin-bottom: 1.5rem;
}

/* Sidebar subheaders */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #4b6cb7 !important;
    font-weight: 500 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(0,210,255,0.1) !important;
}

/* Slider labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00d2ff !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 12px rgba(0,210,255,0.6) !important;
}

.stSlider [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #0066ff, #00d2ff) !important;
}

/* Number inputs */
.stNumberInput input {
    background: #0d1929 !important;
    border: 1px solid rgba(0,210,255,0.2) !important;
    border-radius: 8px !important;
    color: #e8f0fe !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stNumberInput input:focus {
    border-color: #00d2ff !important;
    box-shadow: 0 0 0 2px rgba(0,210,255,0.15) !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0066ff 0%, #00d2ff 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    border: none !important;
    box-shadow: 0 4px 24px rgba(0,102,255,0.4) !important;
    transition: all 0.25s ease !important;
    margin-top: 1.5rem !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(0,210,255,0.5) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1929 0%, #0a1525 100%) !important;
    border: 1px solid rgba(0,210,255,0.15) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #e8f0fe !important;
}

/* Force metric text visibility across Streamlit nested spans/labels */
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] *,
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

/* Streamlit version-safe metric text selectors */
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

/* ── Alerts / Result boxes ── */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: none !important;
    font-family: 'Nunito Sans', sans-serif !important;
}

/* Success */
div[data-testid="stAlert"][kind="success"],
.element-container .stSuccess {
    background: linear-gradient(135deg, rgba(16,185,129,0.24), rgba(6,78,59,0.32)) !important;
    border-left: 4px solid #34d399 !important;
    color: #ecfdf5 !important;
    box-shadow: 0 8px 24px rgba(16,185,129,0.16) !important;
}

/* Error */
div[data-testid="stAlert"][kind="error"],
.element-container .stError {
    background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(127,29,29,0.30)) !important;
    border-left: 4px solid #f87171 !important;
    color: #fff1f2 !important;
    box-shadow: 0 8px 24px rgba(239,68,68,0.14) !important;
}

/* Warning */
div[data-testid="stAlert"][kind="warning"],
.element-container .stWarning {
    background: linear-gradient(135deg, rgba(245,158,11,0.24), rgba(120,53,15,0.30)) !important;
    border-left: 4px solid #fbbf24 !important;
    color: #fffbeb !important;
    box-shadow: 0 8px 24px rgba(245,158,11,0.14) !important;
}

/* Info */
div[data-testid="stAlert"][kind="info"],
.element-container .stInfo {
    background: linear-gradient(135deg, rgba(59,130,246,0.24), rgba(30,58,138,0.30)) !important;
    border-left: 4px solid #60a5fa !important;
    color: #eff6ff !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.14) !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,210,255,0.1) !important;
    margin: 2rem 0 !important;
}

/* ── Subheaders in main area ── */
.stSubheader, [data-testid="stSubheader"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    color: #bfdbfe !important;
    text-shadow: 0 0 14px rgba(96,165,250,0.20) !important;
}

/* ── Code block ── */
.stCodeBlock {
    background: #0d1929 !important;
    border: 1px solid rgba(0,210,255,0.15) !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #60a5fa !important;
    background: #0d1929 !important;
    border-radius: 10px !important;
}

/* ── Info idle cards ── */
.stApp .stInfo p {
    color: #93c5fd !important;
    font-family: 'Nunito Sans', sans-serif !important;
}

/* ── Plotly chart container ── */
.js-plotly-plot {
    border-radius: 16px;
    overflow: hidden;
}

/* ── Header accent bar ── */
.header-accent {
    width: 60px;
    height: 4px;
    background: linear-gradient(90deg, #0066ff, #00d2ff);
    border-radius: 2px;
    margin-bottom: 1.5rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.badge-live {
    background: rgba(0,200,120,0.12);
    color: #00c878;
    border: 1px solid rgba(0,200,120,0.25);
}

.badge-live::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00c878;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4b6cb7;
    font-weight: 600;
    margin-bottom: 0.75rem;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None


# ── Header ──
st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
st.title("DIABETES PREDICTION SYSTEM")
st.markdown("### AI-Powered Risk Assessment Tool")

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please run:
    ```
    python diabetes_prediction.py
    ```
    """)
    st.stop()

# ── Sidebar ──
st.sidebar.title("⚙️ Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider('Age', 21, 100, 30)
pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 0)

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 130, 70)
skin = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)
insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 900, 80)
bmi = st.sidebar.number_input('BMI', 10.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, 0.01)

predict_btn = st.sidebar.button("🔮 Predict", use_container_width=True)

# ── Main ──
if predict_btn:

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_std = scaler.transform(input_data)

    prediction = model.predict(input_std)[0]

    try:
        probability = model.predict_proba(input_std)[0]
        prob_negative = probability[0] * 100
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive

    st.markdown("---")
    st.header("🎯 Prediction Results")

    col1, col2 = st.columns([2, 1])

    # ── Result ──
    with col1:

        if prediction == 0:
            if prob_positive < 30:
                st.success("### ✅ LOW RISK — Not Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK — Not Diabetic")
        else:
            if prob_positive > 70:
                st.error("### 🔴 HIGH RISK — Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK — Diabetic")

        st.subheader("Probability Breakdown")

        c1, c2 = st.columns(2)
        c1.metric("Non-Diabetic", f"{prob_negative:.1f}%")
        c2.metric("Diabetic", f"{prob_positive:.1f}%")

    # ── Gauge ──
    with col2:
        # Determine accent color based on risk
        if prob_positive < 30:
            bar_color = "#00e396"
        elif prob_positive < 70:
            bar_color = "#ffb400"
        else:
            bar_color = "#ff4560"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            title={
                'text': "RISK LEVEL",
                'font': {'color': '#7ea8d4', 'size': 12, 'family': 'Outfit'}
            },
            number={
                'font': {'color': '#ffffff', 'size': 42, 'family': 'Outfit'},
                'suffix': '%'
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': '#2a4a6b',
                    'tickfont': {'color': '#7ea8d4', 'size': 10, 'family': 'Outfit'},
                    'nticks': 6
                },
                'bar': {'color': bar_color, 'thickness': 0.28},
                'bgcolor': "#0d1929",
                'borderwidth': 2,
                'bordercolor': '#1a3050',
                'steps': [
                    {'range': [0, 30],  'color': '#0a2a1a'},
                    {'range': [30, 70], 'color': '#1f1a00'},
                    {'range': [70, 100], 'color': '#2a0a0a'}
                ],
                'threshold': {
                    'line': {'color': bar_color, 'width': 3},
                    'thickness': 0.8,
                    'value': prob_positive
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor='#090f1e',
            plot_bgcolor='#090f1e',
            font={'family': 'Outfit'},
        )
        fig.update_layout(
            shapes=[dict(
                type='rect', xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='#1a3050', width=1),
                fillcolor='rgba(0,0,0,0)'
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk Factor Analysis ──
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis")

    risk_factors = []
    positive_factors = []

    if glucose > 125:
        risk_factors.append("🔴 High Glucose Level (>125 mg/dL)")
    elif glucose < 100:
        positive_factors.append("🟢 Normal Glucose Level")

    if bmi > 30:
        risk_factors.append("🔴 High BMI - Obesity (>30)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 Healthy BMI (18.5-24.9)")

    if age > 45:
        risk_factors.append("🟡 Age Factor (>45)")

    if bp > 80:
        risk_factors.append("🔴 High Blood Pressure (>80 mm Hg)")
    elif 60 <= bp <= 80:
        positive_factors.append("🟢 Normal Blood Pressure")

    if dpf > 0.5:
        risk_factors.append("🟡 Genetic Predisposition")

    if risk_factors:
        st.warning("**Identified Risk Factors:**")
        for r in risk_factors:
            st.markdown(f"- {r}")

    if positive_factors:
        st.success("**Positive Health Indicators:**")
        for p in positive_factors:
            st.markdown(f"- {p}")

    # ── Recommendations ──
    st.markdown("---")
    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.error("""
        - Consult healthcare professional
        - Regular glucose monitoring
        - Lifestyle changes required
        """)
    else:
        st.success("""
        - Maintain healthy diet
        - Exercise regularly
        - Routine checkups
        """)

    # ── Disclaimer ──
    st.markdown("---")
    st.warning("""
    ⚠️ This tool is for educational purposes only and not a medical diagnosis.
    """)

else:
    st.markdown("---")
    st.info("👈 Enter patient information in the sidebar and click **Predict**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Type", "SVM")
    c2.metric("Accuracy", "~78%")
    c3.metric("Dataset", "768 samples")
