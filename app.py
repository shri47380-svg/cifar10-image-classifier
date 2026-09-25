import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionAI | CIFAR-10",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.18), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(6,182,212,0.14), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(168,85,247,0.12), transparent 30%),
        #070b17;
    color: #f8fafc;
}

/* Remove default top padding */
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 35px 20px 25px 20px;
}

.badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 50px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.35);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 62px;
    font-weight: 800;
    margin: 0;
    line-height: 1.05;

    background: linear-gradient(
        90deg,
        #818cf8,
        #22d3ee,
        #c084fc,
        #818cf8
    );

    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: gradientMove 8s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero p {
    color: #94a3b8;
    font-size: 18px;
    margin-top: 15px;
}

/* =========================================================
   GLASS CARDS
   ========================================================= */

.glass {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 24px;
    padding: 28px;
    box-shadow:
        0 20px 50px rgba(0,0,0,0.25),
        inset 0 1px rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    margin-bottom: 20px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 15px;
}

.card-subtitle {
    color: #64748b;
    font-size: 13px;
}

/* =========================================================
   UPLOAD AREA
   ========================================================= */

[data-testid="stFileUploader"] {
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.7),
        rgba(15,23,42,0.7)
    );

    border: 2px dashed rgba(129,140,248,0.45);
    border-radius: 22px;
    padding: 22px;
    transition: 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #818cf8;
    box-shadow: 0 0 30px rgba(99,102,241,0.15);
}

/* =========================================================
   PREDICTION CARD
   ========================================================= */

.prediction-card {
    background:
        linear-gradient(
            145deg,
            rgba(79,70,229,0.22),
            rgba(6,182,212,0.10)
        );

    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 24px;
    padding: 30px;
    text-align: center;

    box-shadow:
        0 0 40px rgba(99,102,241,0.10);
}

.prediction-label {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.prediction-name {
    font-size: 42px;
    font-weight: 800;
    margin: 10px 0;

    background: linear-gradient(
        90deg,
        #818cf8,
        #22d3ee
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.confidence-number {
    font-size: 28px;
    font-weight: 700;
    color: #34d399;
}

/* =========================================================
   STAT CARDS
   ========================================================= */

.stat {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #e2e8f0;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section {
    margin-top: 35px;
    margin-bottom: 18px;
}

.section h2 {
    font-size: 28px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section p {
    color: #64748b;
}

/* =========================================================
   TOP PREDICTION
   ========================================================= */

.top-prediction {
    background: rgba(30,41,59,0.55);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 16px;
    padding: 15px 18px;
    margin-bottom: 10px;
}

.top-name {
    font-weight: 700;
    color: #e2e8f0;
}

.top-value {
    color: #818cf8;
    font-weight: 700;
}

/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 20px;
    padding: 25px;
    height: 150px;
}

.info-icon {
    font-size: 28px;
}

.info-title {
    font-weight: 700;
    margin-top: 8px;
}

.info-text {
    color: #64748b;
    font-size: 13px;
    margin-top: 6px;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #475569;
    padding: 35px 0 10px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")


model = load_model()


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="badge">
● AI POWERED • CIFAR-10 VISION
</div>

<h1>VisionAI</h1>

<p>
Intelligent image classification powered by
Deep Learning & Convolutional Neural Networks
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# STATS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">10</div>
        <div class="stat-label">IMAGE CLASSES</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">32×32</div>
        <div class="stat-label">INPUT SIZE</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">CNN</div>
        <div class="stat-label">MODEL TYPE</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">64%</div>
        <div class="stat-label">TEST ACCURACY</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("""
<div class="section">
<h2>🔮 Analyze an Image</h2>
<p>Upload an image and let the neural network identify what it sees.</p>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_resized = image.resize((32, 32))

    image_array = np.array(image_resized)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100

    sorted_indices = np.argsort(predictions)[::-1]


    # =====================================================
    # IMAGE + RESULT
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:

        st.markdown("""
        <div class="glass">
        <div class="card-title">🖼️ Input Image</div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


    with right:

        st.markdown(f"""
        <div class="prediction-card">

        <div class="prediction-label">
        AI PREDICTION
        </div>

        <div class="prediction-name">
        {predicted_class}
        </div>

        <div class="prediction-label">
        CONFIDENCE
        </div>

        <div class="confidence-number">
        {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if confidence >= 70:
            st.success("High confidence prediction")

        elif confidence >= 40:
            st.info("Moderate confidence prediction")

        else:
            st.warning("Low confidence prediction")


    # =====================================================
    # TOP 3
    # =====================================================

    st.markdown("""
    <div class="section">
    <h2>🏆 Top Predictions</h2>
    <p>Most likely classes detected by the neural network.</p>
    </div>
    """, unsafe_allow_html=True)

    for rank, index in enumerate(sorted_indices[:3], start=1):

        class_name = class_names[index]

        probability = predictions[index] * 100

        st.markdown(f"""
        <div class="top-prediction">

        <span class="top-name">
        #{rank} &nbsp; {class_name}
        </span>

        <span style="float:right" class="top-value">
        {probability:.2f}%
        </span>

        </div>
        """, unsafe_allow_html=True)

        st.progress(float(predictions[index]))


    # =====================================================
    # ALL PROBABILITIES
    # =====================================================

    st.markdown("""
    <div class="section">
    <h2>📊 Full Probability Analysis</h2>
    <p>Confidence distribution across all CIFAR-10 classes.</p>
    </div>
    """, unsafe_allow_html=True)

    for index in sorted_indices:

        class_name = class_names[index]

        probability = predictions[index] * 100

        st.write(
            f"**{class_name}** — {probability:.2f}%"
        )

        st.progress(float(predictions[index]))


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("""
<div class="section">
<h2>⚡ How VisionAI Works</h2>
<p>Three simple steps from image to prediction.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="info-card">

    <div class="info-icon">📤</div>

    <div class="info-title">
    01 — Upload
    </div>

    <div class="info-text">
    Upload a JPG, JPEG or PNG image.
    </div>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-card">

    <div class="info-icon">🧠</div>

    <div class="info-title">
    02 — Analyze
    </div>

    <div class="info-text">
    The CNN processes the image and extracts visual features.
    </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-card">

    <div class="info-icon">🎯</div>

    <div class="info-title">
    03 — Predict
    </div>

    <div class="info-text">
    The model selects the class with the highest probability.
    </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("""
<div class="section">
<h2>🧬 About the Model</h2>
</div>

<div class="glass">

<b>Dataset:</b> CIFAR-10<br><br>

<b>Architecture:</b> Convolutional Neural Network (CNN)<br><br>

<b>Input:</b> 32 × 32 RGB image<br><br>

<b>Classes:</b> Airplane, Automobile, Bird, Cat, Deer,
Dog, Frog, Horse, Ship and Truck<br><br>

<b>Data Augmentation:</b> Horizontal Flip, Rotation and Zoom

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🤖 VisionAI &nbsp; • &nbsp;
TensorFlow &nbsp; • &nbsp;
CIFAR-10 &nbsp; • &nbsp;
Streamlit

<br><br>

Built with Deep Learning

</div>
""", unsafe_allow_html=True)
