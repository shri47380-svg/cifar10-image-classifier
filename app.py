import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️"
)

# CIFAR-10 class names
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

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")

model = load_model()

# Website title
st.title("🖼️ CIFAR-10 Image Classification")

st.write(
    "Upload an image and the AI model will predict its class."
)

st.info(
    "Classes: Airplane, Automobile, Bird, Cat, Deer, "
    "Dog, Frog, Horse, Ship and Truck."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your uploaded image",
        use_container_width=True
    )

    # Resize image to CIFAR-10 size
    image_resized = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.array(image_resized)

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100

    # Display prediction
    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # Display probabilities
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = predictions[i] * 100

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            float(predictions[i])
        )

# Project information
st.divider()

st.subheader("About This Project")

st.write(
    """
    This project uses a Convolutional Neural Network (CNN)
    trained on the CIFAR-10 dataset.

    Data augmentation techniques such as horizontal flipping,
    rotation and zooming were used during training to provide
    more varied training images and help reduce overfitting.
    """
)