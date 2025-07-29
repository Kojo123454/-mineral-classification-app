import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model


# ✅ Corrected line to load the model
model = tf.keras.models.load_model("densenet_mineral_classifier.keras")

# Class names
class_names = ['Biotite', 'Illite', 'Kaolinite', 'Muscovite', 'Plagioclase', 'Quartz', 'Smectite']

# Set up the Streamlit UI
st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("Mineral Classification in Thin-Section Sandstone")
st.write("Upload a petrographic image (thin section), and the model will predict the most likely mineral.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Thin Section", use_container_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Display prediction
    st.success(f"**Predicted Mineral: {predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.barh(class_names, predictions, color='skyblue')
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probability")
    ax.invert_yaxis()
    st.pyplot(fig)
