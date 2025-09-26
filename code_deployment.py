# Streamlit deployment
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

# Define class names
class_names = ['Center', 'Donut', 'Edge Local', 'Edge Ring', 'Local', 'near full', 'none', 'random', 'Scratch']

# Load models
vgg_pretrained = load_model('model_vgg16_pretrained.h5') 
deep_cnn = load_model('Deep_CNN.h5') 

# Model options dictionary
model_options = {
    "VGG16 Pretrained": vgg_pretrained,
    "Deep CNN": deep_cnn
}

# Streamlit UI
st.title("Silicon Wafer Defect Classification")

# Model selection
selected_model_name = st.selectbox("Select a model:", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Image uploader
uploaded_file = st.file_uploader("Upload a silicon wafer image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded wafer image", use_container_width=True)

    # Preprocess the image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)  # VGG-style preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = selected_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show prediction
    st.markdown(f"### 🧠 Model Used: `{selected_model_name}`")
    st.markdown(f"### 🔍 Prediction: `{predicted_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

