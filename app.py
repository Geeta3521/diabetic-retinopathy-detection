"""
app.py — Streamlit Web Application
AI-Based Multi-Disease Eye Disease Detection System
Run with: streamlit run app.py
"""

import os
import sys
import io
import time
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI Eye Disease Detector",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-primary: #020817;
    --bg-card: #0f172a;
    --bg-card2: #1e293b;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.3);
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #8b5cf6;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg-primary);
    color: var(--text);
  }

  .stApp { background: var(--bg-primary); }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
  }

  /* Cards */
  .card {
    background: var(--bg-card);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent);
    margin-bottom: 0.75rem;
  }

  /* Result badge */
  .result-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
  }

  /* Confidence bar */
  .conf-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
  }

  /* Stat grid */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-top: 0.5rem;
  }
  .stat-item {
    background: #1e293b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(59,130,246,0.15);
  }
  .stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .stat-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 2px;
  }

  /* Disease pill */
  .disease-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
  }

  /* Streamlit tweaks */
  .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
  }

  div[data-testid="stFileUploadDropzone"] {
    background: #0f172a !important;
    border: 2px dashed rgba(59,130,246,0.4) !important;
    border-radius: 12px !important;
  }

  .stTextInput > div > div > input,
  .stSelectbox > div > div > select {
    background: #1e293b !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid rgba(59,130,246,0.15) !important;
  }

  [data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem;
  }

  .stMarkdown h3 {
    color: var(--accent) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
  }

  /* Info box */
  .info-box {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--muted);
  }

  /* Warning box */
  .warn-box {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.8rem;
    color: #fcd34d;
  }

  /* Step indicator */
  .step-dot {
    display: inline-block;
    width: 26px; height: 26px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    text-align: center;
    line-height: 26px;
    margin-right: 8px;
  }

  hr { border-color: rgba(59,130,246,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Lazy Imports ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached():
    from predict import load_model
    return load_model()


def run_prediction(pil_image):
    from predict import predict_from_pil
    return predict_from_pil(pil_image)


def run_gradcam(pil_image, result):
    from utils.gradcam import generate_gradcam_figure
    from utils.preprocessing import preprocess_pil_image
    img_array = result["preprocessed_array"]
    model = load_model_cached()
    return generate_gradcam_figure(
        original_pil=pil_image,
        preprocessed_array=img_array,
        model=model,
        predicted_class=result["display_name"],
        confidence=result["confidence"],
        pred_index=result["pred_index"]
    )


def run_report(patient_name, patient_age, patient_id, pil_image, gradcam_fig, result):
    from utils.report_generator import generate_report
    return generate_report(
        patient_name=patient_name,
        patient_age=patient_age,
        patient_id=patient_id,
        original_pil=pil_image,
        gradcam_fig=gradcam_fig,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        all_probabilities=result["probabilities"]
    )


# ─── Disease Config ───────────────────────────────────────────────────────────

DISEASE_COLORS = {
    "normal": "#22c55e",
    "cataract": "#f59e0b",
    "diabetic_retinopathy": "#ef4444",
    "glaucoma": "#8b5cf6"
}

DISEASE_DISPLAY = {
    "normal": "Normal",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma"
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
      <div style="font-size: 2.5rem;">👁</div>
      <div style="font-size: 1rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.3px;">AI Eye Diagnostics</div>
      <div style="font-size: 0.7rem; color: #64748b; margin-top: 2px;">Powered by EfficientNetB0 + XAI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Diseases Detected")
    diseases = [
        ("🔴", "Diabetic Retinopathy", "#ef4444"),
        ("🟡", "Cataract", "#f59e0b"),
        ("🟣", "Glaucoma", "#8b5cf6"),
        ("🟢", "Normal", "#22c55e"),
    ]
    for icon, name, color in diseases:
        st.markdown(
            f'<div style="padding: 0.4rem 0.75rem; margin: 3px 0; background: rgba(255,255,255,0.03); '
            f'border-radius: 8px; border-left: 3px solid {color}; font-size: 0.85rem;">'
            f'{icon} {name}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Patient Info")
    patient_name = st.text_input("Patient Name", placeholder="e.g. Arjun Sharma")
    patient_age  = st.text_input("Age", placeholder="e.g. 45")
    patient_id   = st.text_input("Patient ID", placeholder="e.g. PT-001",
                                  value=f"PT-{datetime.now().strftime('%H%M%S')}")

    st.markdown("---")
    st.markdown("""
    <div class="warn-box">
      ⚠️ For educational & screening purposes only. Not a substitute for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1rem; font-size: 0.72rem; color: #475569; text-align: center;">
      Model: EfficientNetB0 · XAI: Grad-CAM<br>
      Framework: TensorFlow · UI: Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─── Main Page ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>👁 AI-Based Eye Disease Detection</h1>
  <p>Upload a retinal fundus image for instant AI-powered screening of Diabetic Retinopathy, 
  Cataract, and Glaucoma — with Explainable AI visualization.</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="card" style="text-align:center">
      <div style="font-size:1.6rem;font-weight:700;color:#3b82f6">4</div>
      <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Classes Detected</div></div>""",
      unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="card" style="text-align:center">
      <div style="font-size:1.6rem;font-weight:700;color:#22c55e">~94%</div>
      <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Avg Test Accuracy</div></div>""",
      unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="card" style="text-align:center">
      <div style="font-size:1.6rem;font-weight:700;color:#8b5cf6">Grad-CAM</div>
      <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Explainable AI</div></div>""",
      unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="card" style="text-align:center">
      <div style="font-size:1.6rem;font-weight:700;color:#f59e0b">EfficientNetB0</div>
      <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Base Architecture</div></div>""",
      unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload + Results layout ──
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.markdown('<div class="card-title"><span class="step-dot">1</span>Upload Fundus Image</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a retinal fundus photograph (JPG/PNG)"
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        st.markdown('<div class="card-title" style="margin-top:1rem"><span class="step-dot">2</span>Run Analysis</div>',
                    unsafe_allow_html=True)

        analyze_btn = st.button("🔍 Analyze Eye Image", key="analyze")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2rem; color: #64748b;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem">📷</div>
          <div style="font-size: 0.9rem">Upload a fundus image to begin</div>
          <div style="font-size: 0.75rem; margin-top: 0.4rem">Supports JPG, PNG, BMP</div>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

with right_col:
    if uploaded_file and analyze_btn:

        # ── Load model first ──
        with st.spinner("Loading AI model..."):
            try:
                model = load_model_cached()
                model_loaded = True
            except FileNotFoundError as e:
                st.error(str(e))
                model_loaded = False

        if model_loaded:
            # ── Prediction ──
            with st.spinner("Analyzing retinal image..."):
                result = run_prediction(pil_image)

            predicted_class = result["predicted_class"]
            display_name    = result["display_name"]
            confidence      = result["confidence"]
            probs           = result["probabilities"]
            class_info      = result["class_info"]
            color           = DISEASE_COLORS.get(predicted_class, "#3b82f6")

            # ── Main Result ──
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Diagnosis Result</div>
              <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px">
                <div class="result-badge" style="background:{color}22; color:{color}; border: 1.5px solid {color}">
                  {display_name}
                </div>
                <div style="font-size:0.85rem; color:#94a3b8">Detected with {confidence:.1f}% confidence</div>
              </div>
              <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:12px">
                {class_info.get('description', '')}
              </div>
              <div style="background: rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))}, 0.1);
                          border: 1px solid {color}33; border-radius:8px; padding:0.7rem 1rem;
                          font-size:0.85rem; color:#e2e8f0;">
                📋 <strong>Recommendation:</strong> {class_info.get('recommendation', 'Consult a doctor.')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence Breakdown ──
            st.markdown('<div class="card-title" style="margin-top:0.5rem">Confidence Breakdown</div>',
                        unsafe_allow_html=True)

            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            bar_colors = [DISEASE_COLORS.get(k, "#3b82f6") for k, _ in sorted_probs]
            bar_labels = [DISEASE_DISPLAY.get(k, k) for k, _ in sorted_probs]
            bar_values = [v for _, v in sorted_probs]

            fig = go.Figure(go.Bar(
                x=bar_values,
                y=bar_labels,
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0)", width=0)
                ),
                text=[f"{v:.1f}%" for v in bar_values],
                textposition="inside",
                textfont=dict(color="white", size=12, family="Space Grotesk")
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e293b",
                height=200,
                margin=dict(l=0, r=10, t=10, b=0),
                xaxis=dict(
                    range=[0, 105],
                    showgrid=True, gridcolor="#334155",
                    tickfont=dict(color="#94a3b8", size=10),
                    title=dict(text="Confidence (%)", font=dict(color="#94a3b8", size=11))
                ),
                yaxis=dict(
                    tickfont=dict(color="#e2e8f0", size=11, family="Space Grotesk"),
                    showgrid=False
                ),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Grad-CAM ──
            st.markdown('<div class="card-title"><span class="step-dot">3</span>Explainable AI — Grad-CAM</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating Grad-CAM heatmap..."):
                try:
                    gradcam_fig = run_gradcam(pil_image, result)
                    # Convert to image buffer for Streamlit
                    buf = io.BytesIO()
                    gradcam_fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                                        facecolor=gradcam_fig.get_facecolor())
                    buf.seek(0)
                    st.image(buf, caption="Grad-CAM: Highlighted diagnostic regions", use_column_width=True)
                    gradcam_ok = True
                except Exception as e:
                    st.warning(f"Grad-CAM visualization unavailable: {e}")
                    gradcam_fig = None
                    gradcam_ok = False

            # ── PDF Report ──
            st.markdown('<div class="card-title" style="margin-top:0.5rem"><span class="step-dot">4</span>Download Report</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating PDF report..."):
                try:
                    report_path = run_report(
                        patient_name, patient_age, patient_id,
                        pil_image, gradcam_fig if gradcam_ok else None, result
                    )
                    if report_path and os.path.exists(report_path):
                        with open(report_path, "rb") as f:
                            report_bytes = f.read()
                        st.download_button(
                            label="📄 Download Medical Report (PDF)",
                            data=report_bytes,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )
                    else:
                        st.info("PDF generation requires 'reportlab'. Install it to enable reports.")
                except Exception as e:
                    st.info(f"PDF report: {e}")

    elif not uploaded_file:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:380px; color:#334155;">
          <div style="font-size:4rem; margin-bottom:1rem; opacity:0.4">🔬</div>
          <div style="font-size:1rem; color:#475569; text-align:center;">
            Results will appear here after analysis
          </div>
          <div style="font-size:0.8rem; color:#334155; margin-top:0.5rem">
            Upload an image and click Analyze
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── How It Works ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### How It Works")

how_cols = st.columns(4)
steps = [
    ("📤", "Upload Image", "Upload a retinal fundus photograph (JPG/PNG/BMP)"),
    ("⚙️", "Preprocessing", "CLAHE enhancement, resize to 224×224, normalization"),
    ("🧠", "EfficientNetB0", "Transfer learning model classifies into 4 categories"),
    ("🔥", "Grad-CAM XAI", "Heatmap shows which regions drove the prediction"),
]
for col, (icon, title, desc) in zip(how_cols, steps):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center; height:130px">
          <div style="font-size:1.8rem">{icon}</div>
          <div style="font-weight:600; margin:0.3rem 0; font-size:0.9rem">{title}</div>
          <div style="font-size:0.75rem; color:#94a3b8">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1.5rem 0 1rem; color:#334155; font-size:0.75rem">
  AI-Based Multi-Disease Eye Disease Detection System &nbsp;·&nbsp;
  EfficientNetB0 + Grad-CAM &nbsp;·&nbsp;
  Built with TensorFlow & Streamlit<br>
  <span style="color:#1e3a5f">⚠️ Not for clinical use — educational project</span>
</div>
""", unsafe_allow_html=True)
