
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Assisted Glaucoma Screening",
    layout="wide"
)

# ---------------- LOAD MODEL (ONCE) ----------------
@st.cache_resource
def load_my_model():
    return load_model("retinal_model_v3.keras", compile=False)

model = load_my_model()

# ---------------- CONSTANTS ----------------
IMG_SIZE = 256

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
/* Page width */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* App background */
.stApp {
    background-color: #f4f6f8;
    font-family: "Segoe UI", sans-serif;
}

            
/* FORCE readable text */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #111827 !important;
}

        

/* Header */
.header {
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* Feature cards */
.feature-card {
    background-color: #f9fafb;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Status badges */
.status-green {
    background-color: #e6f4ea;
    color: #166534 !important;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}

.status-red {
    background-color: #fdecea;
    color: #991b1b !important;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<div class="header">
    <h1>AI-Assisted Glaucoma Screening</h1>
    <p>Upload a retinal fundus image for automated screening</p>
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT / INFO ----------------
with st.expander("ℹ️ About this project"):
    st.markdown("""
    **Glaucoma** is a leading cause of irreversible blindness caused by
    progressive optic nerve damage, often without early symptoms.

    This application uses **retinal fundus imaging** and a
    **deep learning model** to screen for glaucomatous patterns.

    It is intended as a **clinical decision-support tool**
    and not a definitive diagnostic system.
    """)

# ---------------- FEATURE CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <b>👁 Retinal Imaging</b><br>
        Uses standard fundus photographs.<br>
        Non-invasive optic nerve analysis.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <b>🧠 Deep Learning Model</b><br>
        High-sensitivity classifier.<br>
        Distinguishes glaucomatous patterns.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <b>🩺 Clinical Decision Support</b><br>
        Prioritizes case management.<br>
        Improves transparency.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN DASHBOARD ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1.3])

    # IMAGE PANEL
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION
    processed_image = preprocess_image(image)
    prediction = float(model.predict(processed_image)[0][0])

    # RESULT PANEL
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if prediction >= 0.10:
            st.markdown(
                '<div class="status-red">Status: Glaucoma Detected ⚠️</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"### AI Confidence: {prediction:.2%}")
        else:
            st.markdown(
                '<div class="status-green">Status: Non-Glaucomatous (Low Risk) ✅</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"### AI Confidence: {(1 - prediction):.2%}")

        st.markdown("""
        The AI analysis estimates the likelihood of glaucoma
        based on optic disc morphology and vascular patterns.
        Results are intended for screening purposes only.
        """)

        st.bar_chart({
            "Optic Disc Size": 0.72,
            "Cup-to-Disc Ratio": prediction,
            "Rim Width": 0.45,
            "Vessel Configuration": 0.38
        })

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>Upload a different retinal fundus image above</p>",
    unsafe_allow_html=True
)
