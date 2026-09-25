import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 AI Vision",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
        color: white;
    }

    /* Main container */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Prediction */
    .prediction-title {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 5px;
    }

    .prediction-class {
        font-size: 36px;
        font-weight: 800;
        color: #60a5fa;
    }

    .confidence {
        font-size: 24px;
        font-weight: 700;
        color: #34d399;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 40px;
        font-size: 14px;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(96,165,250,0.5);
        border-radius: 15px;
        padding: 15px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CIFAR-10 CLASSES
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 CIFAR-10 AI Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Image Classification using TensorFlow & CNN'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# PROJECT INTRO
# --------------------------------------------------

st.markdown("""
<div class="card">

<h3>🧠 About the AI</h3>

<p>
This application uses a Convolutional Neural Network (CNN)
trained on the CIFAR-10 dataset to recognize objects in images.
</p>

<p>
<strong>10 Classes:</strong>
Airplane • Automobile • Bird • Cat • Deer • Dog • Frog • Horse • Ship • Truck
</p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📤 Upload an Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 1])

    # --------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">🖼️ Your Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    image_resized = image.resize((32, 32))

    image_array = np.array(image_resized)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">🤖 AI Prediction</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">

            <div class="prediction-title">
            Predicted Class
            </div>

            <div class="prediction-class">
            {predicted_class}
            </div>

            <br>

            <div class="prediction-title">
            Confidence
            </div>

            <div class="confidence">
            {confidence:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# PROBABILITIES
# --------------------------------------------------

if uploaded_file is not None:

    st.markdown(
        '<div class="section-title">📊 Class Probabilities</div>',
        unsafe_allow_html=True
    )

    # Sort probabilities from highest to lowest
    sorted_indices = np.argsort(predictions)[::-1]

    for index in sorted_indices:

        class_name = class_names[index]

        probability = predictions[index] * 100

        col1, col2 = st.columns([2, 5])

        with col1:
            st.write(f"**{class_name}**")

        with col2:
            st.progress(
                float(predictions[index])
            )

        st.write(
            f"{probability:.2f}%"
        )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="card">

    <h3>1️⃣ Upload</h3>

    <p>
    Upload a JPG, JPEG or PNG image.
    </p>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <h3>2️⃣ Process</h3>

    <p>
    The image is resized to 32×32 pixels
    and normalized before prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="card">

    <h3>3️⃣ Predict</h3>

    <p>
    The CNN predicts which CIFAR-10
    class is most likely.
    </p>

    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

CIFAR-10 AI Vision • TensorFlow • CNN • Streamlit

</div>
""", unsafe_allow_html=True)
