import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from utils.preprocess import preprocess_image


st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="🛍️",
    layout="centered"
)


CLASS_NAMES = [
    "Dresses",
    "Jeans",
    "Shirts",
    "Shoes",
    "Tshirts"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/E-commerce.h5")

model = load_model()


st.title("🛍️ E-Commerce Clothing Classifier")
st.write("Upload a clothing image and get its category.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "jfif", "bmp", "avif"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

    st.success(f"Prediction: **{CLASS_NAMES[predicted_index]}**")
    st.info(f"Confidence: **{confidence:.2%}**")

  
    st.write("Raw prediction vector:", predictions)
    st.write("Predicted index:", predicted_index)
