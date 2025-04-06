import streamlit as st
import os
import zipfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Title
st.title("🔍 Deepfake Image Detection using YOLOv8")

# Extract model if not extracted
model_dir = "yolo_model"
zip_path = "yolo_trained_model.zip"

if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    st.success("✅ Model unzipped successfully.")

# Load model
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
if model_files:
    model_path = os.path.join(model_dir, model_files[0])
    model = YOLO(model_path)
    st.success("✅ YOLOv8 Model loaded!")
else:
    st.error("❌ No .pt file found in the unzipped model folder.")

# Upload image
uploaded_image = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    # Prediction button
    if st.button("Detect Deepfake"):
        with st.spinner("Analyzing..."):
            results = model.predict(image)

        # Draw boxes on the image
        result_image = results[0].plot()

        # Convert to PIL Image and display
        result_pil = Image.fromarray(result_image[..., ::-1])  # BGR to RGB
        st.image(result_pil, caption="Detection Result", width=200)

        # Removed label display line 👇
        # st.write("🔎 Detected Labels:", results[0].names)
