import os
import io
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown

# Initialize logging
logging.basicConfig(level=logging.INFO)

MODEL_ID = "1FMvhvLE2ikEmeIgNFMM8jlnpI_iRSTIn"  # Replace with your actual model ID
MODEL_PATH = "./model.tflite"
CLASS_NAMES = ['Organic', 'Recycleable']  # Replace with your class names

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model file '{MODEL_PATH}' not found. Downloading...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
            file_size = os.path.getsize(MODEL_PATH)
            logging.info(f"Model downloaded. Size: {file_size} bytes")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            raise
    else:
        logging.info(f"Model file '{MODEL_PATH}' already exists. Skipping download.")

download_model()

# Load TFLite model
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    st.error("Failed to load the model. Please check the model file or try again.")
    interpreter = None

# Image preprocessing function (unchanged)
def preprocess_image(file):
    try:
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB')
        input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
        img = img.resize(input_shape)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        st.error("Error processing image. Please try again with a valid image file.")
        return None

# Streamlit UI
st.set_page_config(page_title="Waste Classification", page_icon=":recycle:")  # Set page title and icon

st.title("Waste Classification using CNN")
st.write("Upload an image to classify it as either 'Organic' or 'Recyclable'.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

CLASS_NAMES = ['Organic', 'Recyclable']

if uploaded_file is not None:
    if interpreter is not None:
        img_array = preprocess_image(uploaded_file)

        if img_array is not None:
            with st.spinner("Classifying..."):  # Add a spinner while processing
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_class_index]
                predicted_score = prediction[0][predicted_class_index] * 100

                col1, col2 = st.columns([1, 1]) # Equal columns for image and results
                with col1:
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                with col2:
                    st.write(f"**Prediction:** {predicted_class}")
                    st.write(f"**Confidence:** {predicted_score:.2f}%")

    else:
        st.error("Model is not loaded. Please restart the app or check the model file.")

st.markdown("---")  # Separator line
st.markdown("This project uses a CNN model to classify waste. Thanks for visiting!")
